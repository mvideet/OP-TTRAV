# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single
GPU model. Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model
to perform generation.
"""

import contextlib

import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.torch_functional import get_response_mask

from .base import BaseRollout

__all__ = ["HFRollout"]


class HFRollout(BaseRollout):
    def __init__(self, module: nn.Module, config, tokenizer=None):
        super().__init__()
        self.config = config
        self.module = module
        self.tokenizer = tokenizer  # Optional tokenizer for debug decoding

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        batch_size = prompts.batch.batch_size[0]
        num_chunks = max(batch_size // self.config.get("micro_batch_size", batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
        return output

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        # make sampling args can be overridden by inputs
        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        is_validate = prompts.meta_info.get("validate", False)

        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = max(0, prompts.meta_info.get("top_k", self.config.get("top_k", 0)))  # to be compatible with vllm

        if not do_sample:
            # do_sample==False -> greedy decoding
            kwargs = {
                "do_sample": False,
                "num_beams": 1,
            }
        elif is_validate:
            # do validate and do sample -> use val_kwargs
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_k": max(0, self.config.val_kwargs.top_k),  # to be compatible with vllm
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "num_return_sequences": 1,  # if validate, already repeat in ray_trainer
            }
        else:
            # do_sample -> use rollout config
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                "num_return_sequences": self.config.n,
            }

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        prompt_length = idx.size(1)
        attention_mask = prompts.batch["attention_mask"]  # left-padded attention_mask
        position_ids = prompts.batch["position_ids"]

        multi_modal_inputs = {}
        if "multi_modal_inputs" in prompts.non_tensor_batch:
            mm_inputs_list = prompts.non_tensor_batch["multi_modal_inputs"]
            if mm_inputs_list is not None and len(mm_inputs_list) > 0:
                target_device = idx.device
                target_dtype = torch.bfloat16  # Model runs in bfloat16
                
                for key in mm_inputs_list[0].keys():
                    if key == "use_audio_in_video":
                        multi_modal_inputs[key] = mm_inputs_list[0][key]
                    else:
                        tensors = [inputs[key] for inputs in mm_inputs_list if inputs.get(key) is not None]
                        if tensors:
                            concatenated = torch.cat(tensors, dim=0)
                            concatenated = concatenated.to(target_device)
                            if concatenated.is_floating_point():
                                concatenated = concatenated.to(target_dtype)
                            multi_modal_inputs[key] = concatenated

        # used to construct attention_mask
        # Get eos_token_id and pad_token_id from meta_info, with fallback to model config
        eos_token_id = prompts.meta_info.get("eos_token_id")
        pad_token_id = prompts.meta_info.get("pad_token_id")
        
        # Fallback to model config if not in meta_info
        if eos_token_id is None:
            eos_token_id = getattr(self.module.config, "eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.module.config, "pad_token_id", None)
        
        # Final fallback - eos_token_id is required
        if eos_token_id is None:
            raise ValueError("eos_token_id is None - not found in prompts.meta_info or model.config")

        self.module.eval()
        param_ctx = contextlib.nullcontext()

        if isinstance(self.module, FSDP):
            # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        
        # Get the underlying model class name (handle FSDP wrapping)
        model_to_check = self.module
        while hasattr(model_to_check, '_fsdp_wrapped_module'):
            model_to_check = model_to_check._fsdp_wrapped_module
        model_class_name = model_to_check.__class__.__name__
        
        # For Qwen Omni models with thinker-talker architecture, disable talker if present
        # Note: If we're using the thinker model directly (recommended), this won't have disable_talker
        if hasattr(model_to_check, 'disable_talker'):
            model_to_check.disable_talker()
        
        with param_ctx, torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            # Build generate kwargs
            generate_kwargs = dict(
                input_ids=idx,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
                max_new_tokens=response_length,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                output_scores=False,  # this is potentially very large
                return_dict_in_generate=True,
                use_cache=True,
            )

            # Add multimodal inputs (video/audio features) if present
            if multi_modal_inputs:
                generate_kwargs.update(multi_modal_inputs)
            
            # For Qwen Omni models with full conditional generation wrapper,
            # disable audio output. Skip if using thinker model directly (standard causal LM).
            if hasattr(model_to_check, 'talker') or 'OmniForConditionalGeneration' in model_class_name:
                generate_kwargs["return_audio"] = False
            
            output = self.module.generate(**generate_kwargs)

        # TODO: filter out the seq with no answers like ds-chat
        seq = output.sequences
        generated_batch_size = seq.size(0)  # bs * num_return_sequences

        # huggingface generate will stop generating when all the batch reaches [EOS].
        # Pad or truncate to match the expected sequence_length.
        sequence_length = prompt_length + response_length
        delta_length = sequence_length - seq.shape[1]

        if delta_length > 0:
            # Sequence is too short, pad with pad_token_id
            delta_tokens = torch.ones(size=(generated_batch_size, delta_length), device=seq.device, dtype=seq.dtype)
            delta_tokens = pad_token_id * delta_tokens
            seq = torch.cat((seq, delta_tokens), dim=1)
        elif delta_length < 0:
            # Sequence is too long, truncate
            seq = seq[:, :sequence_length]
        assert seq.shape[1] == sequence_length, f"seq.shape[1]={seq.shape[1]} != sequence_length={sequence_length}"

        # make necessary reputations if num_return_sequences > 1
        num_return_sequences = kwargs.get("num_return_sequences", 1)
        if num_return_sequences > 1:
            position_ids = position_ids.repeat_interleave(num_return_sequences, dim=0)
            attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)

        prompt = seq[:, :prompt_length]  # (generated_batch_size, prompt_length)
        response = seq[:, prompt_length:]  # (generated_batch_size, response_length)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(generated_batch_size, 1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": prompt,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=generated_batch_size,
        )

        # empty cache before compute old_log_prob
        get_torch_device().empty_cache()

        self.module.train()
        return DataProto(batch=batch)
