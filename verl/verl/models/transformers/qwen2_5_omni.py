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
PPO-specific forward passes for Qwen2.5-Omni models.

Delegates the full model forward (multimodal embedding, rope, attention) to the
original HF Qwen2_5OmniThinkerForConditionalGeneration.forward, then applies a
fused PPO head (log-probs + entropy) on top of the hidden states.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from transformers.modeling_outputs import ModelOutput


# ---------------------------------------------------------------------------
# Original HF forward — saved before monkey-patching so we can delegate to it.
# ---------------------------------------------------------------------------
_original_hf_forward = None


def set_original_forward(forward_fn):
    """Save the real HF forward before monkey-patching replaces it."""
    global _original_hf_forward
    _original_hf_forward = forward_fn


@dataclass
class Qwen2_5OmniThinkerForCausalLMOutputForPPO(ModelOutput):
    """Output class for PPO training with Qwen2.5-Omni thinker model."""
    log_probs: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


def forward_base_model(
    self,
    input_ids=None,
    input_features=None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    attention_mask=None,
    feature_attention_mask=None,
    audio_feature_lengths=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    rope_deltas=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    use_audio_in_video=None,
    cache_position=None,
    video_second_per_grid=None,
    **kwargs,
):
    """
    Delegate to the original HF Qwen2_5OmniThinkerForConditionalGeneration.forward.

    We always request ``output_hidden_states=True`` so callers can extract the
    last decoder hidden state for the PPO head, and force ``labels=None`` so HF
    skips its own cross-entropy loss (we compute PPO log-probs ourselves).
    """
    assert _original_hf_forward is not None, (
        "Original HF forward not saved. "
        "Call set_original_forward() before monkey-patching (see monkey_patch.py)."
    )
    return _original_hf_forward(
        self,
        input_ids=input_ids,
        input_features=input_features,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
        feature_attention_mask=feature_attention_mask,
        audio_feature_lengths=audio_feature_lengths,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        rope_deltas=rope_deltas,
        labels=None,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=True,
        return_dict=True,
        use_audio_in_video=use_audio_in_video,
        cache_position=cache_position,
        video_second_per_grid=video_second_per_grid,
    )


def forward_with_torch_backend(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    input_features: Optional[torch.FloatTensor] = None,
    feature_attention_mask: Optional[torch.Tensor] = None,
    audio_feature_lengths: Optional[torch.LongTensor] = None,
    use_audio_in_video: Optional[bool] = None,
    video_second_per_grid: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **loss_kwargs,
) -> Union[Tuple, Qwen2_5OmniThinkerForCausalLMOutputForPPO]:
    """
    Forward pass with PyTorch backend for PPO training.
    Computes log probabilities and entropy for policy optimization.
    """
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    if labels is not None and torch.any(labels < 0):
        raise ValueError(
            "The fused PPO forward path does not support labels with ignore indices. "
            "Pass `input_ids` for PPO log-prob computation, or sanitize `labels` before calling this wrapper."
        )

    outputs = forward_base_model(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        input_features=input_features,
        feature_attention_mask=feature_attention_mask,
        audio_feature_lengths=audio_feature_lengths,
        use_audio_in_video=use_audio_in_video,
        video_second_per_grid=video_second_per_grid,
        rope_deltas=rope_deltas,
        cache_position=cache_position,
    )

    # outputs.hidden_states is a tuple of all decoder layers; [-1] is post-norm
    # (same tensor HF feeds to lm_head).
    hidden_states = outputs.hidden_states[-1]

    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_torch_backend, either labels or input_ids must be provided.")

    fused_linear_for_ppo = FusedLinearForPPO()
    log_probs, entropy = fused_linear_for_ppo.forward(
        hidden_states=hidden_states,
        vocab_weights=self.lm_head.weight,
        input_ids=rolled_labels,
        temperature=temperature,
    )

    return Qwen2_5OmniThinkerForCausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        past_key_values=getattr(outputs, 'past_key_values', None),
        hidden_states=getattr(outputs, 'hidden_states', None),
        attentions=getattr(outputs, 'attentions', None),
    )


def forward_with_triton_backend(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    input_features: Optional[torch.FloatTensor] = None,
    feature_attention_mask: Optional[torch.Tensor] = None,
    audio_feature_lengths: Optional[torch.LongTensor] = None,
    use_audio_in_video: Optional[bool] = None,
    video_second_per_grid: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **loss_kwargs,
) -> Union[Tuple, Qwen2_5OmniThinkerForCausalLMOutputForPPO]:
    """
    Forward pass with Triton backend for PPO training.
    Uses fused kernels for better performance.
    """
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

    if labels is not None and torch.any(labels < 0):
        raise ValueError(
            "The fused PPO forward path does not support labels with ignore indices. "
            "Pass `input_ids` for PPO log-prob computation, or sanitize `labels` before calling this wrapper."
        )

    outputs = forward_base_model(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        input_features=input_features,
        feature_attention_mask=feature_attention_mask,
        audio_feature_lengths=audio_feature_lengths,
        use_audio_in_video=use_audio_in_video,
        video_second_per_grid=video_second_per_grid,
        rope_deltas=rope_deltas,
        cache_position=cache_position,
    )

    hidden_states = outputs.hidden_states[-1]

    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_triton_backend, either labels or input_ids must be provided.")

    log_probs, entropy = linear_cross_entropy(
        hidden_states,
        self.lm_head.weight,
        rolled_labels,
        temperature,
        "none",
    )

    return Qwen2_5OmniThinkerForCausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        past_key_values=getattr(outputs, 'past_key_values', None),
        hidden_states=getattr(outputs, 'hidden_states', None),
        attentions=getattr(outputs, 'attentions', None),
    )


def _get_llm_pos_ids_for_vision(
    start_idx, vision_idx, spatial_merge_size, t_index, grid_hs, grid_ws,
):
    """Compute 3D position IDs for a single image/video block. Mirrors HF get_llm_pos_ids_for_vision."""
    grid_h = grid_hs[vision_idx] // spatial_merge_size
    grid_w = grid_ws[vision_idx] // spatial_merge_size
    grid_t = len(t_index)
    h_index = torch.arange(grid_h).view(1, -1, 1).expand(grid_t, -1, grid_w).flatten()
    w_index = torch.arange(grid_w).view(1, 1, -1).expand(grid_t, grid_h, -1).flatten()
    t_expanded = torch.Tensor(t_index).view(-1, 1).expand(-1, grid_h * grid_w).flatten().long()
    return torch.stack([t_expanded, h_index, w_index]) + start_idx


def _get_chunked_index(token_indices, tokens_per_chunk, remove_index):
    """Split token indices into chunks by value ranges. Mirrors HF get_chunked_index."""
    chunks = []
    start_idx = 0
    current_chunk = 1
    for i in range(len(token_indices)):
        if token_indices[i] - remove_index >= current_chunk * tokens_per_chunk:
            chunks.append((start_idx, i))
            start_idx = i
            current_chunk += 1
    chunks.append((start_idx, len(token_indices)))
    return chunks


def get_rope_index(
    processor,
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    use_audio_in_video: bool = False,
    audio_seqlens: Optional[torch.Tensor] = None,
    second_per_grids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute 3D M-RoPE position IDs for Qwen2.5-Omni (single sample, no batch dim).

    Mirrors HF Qwen2_5OmniPreTrainedModelForConditionalGeneration.get_rope_index
    but works standalone using processor attributes. Called from the dataset.

    Args:
        processor: Qwen2_5OmniProcessor instance
        input_ids: 1D token IDs (seq_len,)
        image_grid_thw: (num_images, 3) or None
        video_grid_thw: (num_videos, 3) or None
        attention_mask: 1D mask (seq_len,) or None
        use_audio_in_video: whether audio is interleaved in video
        audio_seqlens: (num_audios,) raw audio feature lengths before downsampling
        second_per_grids: (num_videos,) seconds per temporal grid

    Returns:
        position_ids: (3, seq_len) tensor
    """
    spatial_merge_size = processor.image_processor.merge_size
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    video_token_id = processor.tokenizer.convert_tokens_to_ids(processor.video_token)
    audio_token_id = processor.tokenizer.convert_tokens_to_ids(processor.audio_token)
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids(processor.vision_bos_token)
    audio_start_token_id = processor.tokenizer.convert_tokens_to_ids(processor.audio_bos_token)

    position_id_per_seconds = 25
    seconds_per_chunk = 2.0

    has_multimodal = image_grid_thw is not None or video_grid_thw is not None

    if input_ids is not None and has_multimodal:
        position_ids = torch.ones(3, input_ids.size(0), dtype=input_ids.dtype, device=input_ids.device)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        active_ids = input_ids[attention_mask == 1]
        input_tokens = active_ids.tolist()

        vision_start_indices = torch.argwhere(active_ids == vision_start_token_id).squeeze(1)
        vision_tokens = active_ids[vision_start_indices + 1]
        audio_nums = int(torch.sum(active_ids == audio_start_token_id).item())
        image_nums = int((vision_tokens == image_token_id).sum().item())
        video_nums = int(
            (vision_tokens == audio_start_token_id).sum().item()
            if use_audio_in_video
            else (vision_tokens == video_token_id).sum().item()
        )

        image_idx, video_idx, audio_idx = 0, 0, 0
        llm_pos_ids_list: list = []
        st = 0
        remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums
        multimodal_nums = (
            image_nums + audio_nums if use_audio_in_video else image_nums + video_nums + audio_nums
        )

        for _ in range(multimodal_nums):
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0

            ed_image = input_tokens.index(image_token_id, st) if (image_token_id in input_tokens[st:] and remain_images > 0) else len(input_tokens) + 1
            ed_video = input_tokens.index(video_token_id, st) if (video_token_id in input_tokens[st:] and remain_videos > 0) else len(input_tokens) + 1
            ed_audio = input_tokens.index(audio_token_id, st) if (audio_token_id in input_tokens[st:] and remain_audios > 0) else len(input_tokens) + 1

            min_ed = min(ed_image, ed_video, ed_audio)

            if min_ed == ed_audio:
                text_len = min_ed - st - 1
                if text_len != 0:
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                llm_pos_ids_list.append(torch.arange(1).view(1, -1).expand(3, -1) + st_idx)

                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                audio_len = int(((audio_seqlens[audio_idx] - 1) // 2 + 1 - 2) // 2 + 1)
                llm_pos_ids_list.append(torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx)

                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                llm_pos_ids_list.append(torch.arange(1).view(1, -1).expand(3, -1) + st_idx)

                st += text_len + 1 + audio_len + 1
                audio_idx += 1
                remain_audios -= 1

            elif min_ed == ed_image:
                text_len = min_ed - st - 1
                if text_len != 0:
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                llm_pos_ids_list.append(torch.arange(1).view(1, -1).expand(3, -1) + st_idx)

                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                grid_t = image_grid_thw[image_idx][0]
                t_index = (torch.arange(grid_t) * 1 * position_id_per_seconds).long()
                llm_pos_ids = _get_llm_pos_ids_for_vision(
                    st_idx, image_idx, spatial_merge_size, t_index,
                    image_grid_thw[:, 1], image_grid_thw[:, 2],
                )
                image_len = int(image_grid_thw[image_idx].prod() // (spatial_merge_size ** 2))
                llm_pos_ids_list.append(llm_pos_ids)

                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                llm_pos_ids_list.append(torch.arange(1).view(1, -1).expand(3, -1) + st_idx)

                st += text_len + 1 + image_len + 1
                image_idx += 1
                remain_images -= 1

            elif min_ed == ed_video and not use_audio_in_video:
                text_len = min_ed - st - 1
                if text_len != 0:
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                llm_pos_ids_list.append(torch.arange(1).view(1, -1).expand(3, -1) + st_idx)

                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                grid_t = video_grid_thw[video_idx][0]
                spg = float(second_per_grids[video_idx]) if second_per_grids is not None else 1.0
                t_index = (torch.arange(grid_t) * spg * position_id_per_seconds).long()
                llm_pos_ids = _get_llm_pos_ids_for_vision(
                    st_idx, video_idx, spatial_merge_size, t_index,
                    video_grid_thw[:, 1], video_grid_thw[:, 2],
                )
                video_len = int(video_grid_thw[video_idx].prod() // (spatial_merge_size ** 2))
                llm_pos_ids_list.append(llm_pos_ids)

                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                llm_pos_ids_list.append(torch.arange(1).view(1, -1).expand(3, -1) + st_idx)

                st += text_len + 1 + video_len + 1
                video_idx += 1
                remain_videos -= 1

            elif min_ed == ed_video and use_audio_in_video:
                text_len = min_ed - st - 2
                if text_len != 0:
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                llm_pos_ids_list.append(torch.arange(1).view(1, -1).expand(3, -1) + st_idx)
                llm_pos_ids_list.append(torch.arange(1).view(1, -1).expand(3, -1) + st_idx)

                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                audio_len = int(((audio_seqlens[audio_idx] - 1) // 2 + 1 - 2) // 2 + 1)
                audio_llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx

                grid_t = video_grid_thw[video_idx][0]
                spg = float(second_per_grids[video_idx]) if second_per_grids is not None else 1.0
                t_index = (torch.arange(grid_t) * spg * position_id_per_seconds).long()
                video_llm_pos_ids = _get_llm_pos_ids_for_vision(
                    st_idx, video_idx, spatial_merge_size, t_index,
                    video_grid_thw[:, 1], video_grid_thw[:, 2],
                )

                t_ntoken_per_chunk = int(position_id_per_seconds * seconds_per_chunk)
                video_chunk_indexes = _get_chunked_index(video_llm_pos_ids[0], t_ntoken_per_chunk, st_idx)
                audio_chunk_indexes = _get_chunked_index(audio_llm_pos_ids[0], t_ntoken_per_chunk, st_idx)

                for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
                    if j < len(video_chunk_indexes):
                        ci = video_chunk_indexes[j]
                        llm_pos_ids_list.append(video_llm_pos_ids[:, ci[0]:ci[1]])
                    if j < len(audio_chunk_indexes):
                        ci = audio_chunk_indexes[j]
                        llm_pos_ids_list.append(audio_llm_pos_ids[:, ci[0]:ci[1]])

                video_len = int(video_grid_thw[video_idx].prod() // (spatial_merge_size ** 2))

                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                llm_pos_ids_list.append(torch.arange(1).view(1, -1).expand(3, -1) + st_idx)
                llm_pos_ids_list.append(torch.arange(1).view(1, -1).expand(3, -1) + st_idx)

                st += text_len + 2 + audio_len + video_len + 2
                audio_idx += 1
                video_idx += 1
                remain_videos -= 1
                remain_audios -= 1

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., attention_mask == 1] = llm_positions.to(position_ids.device)

    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1).to(input_ids.device)
        else:
            position_ids = torch.arange(input_ids.shape[0], device=input_ids.device).view(1, -1).expand(3, -1)

    return position_ids
