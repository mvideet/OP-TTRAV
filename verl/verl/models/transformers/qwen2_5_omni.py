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
Adapted from Qwen3-Omni implementation for multimodal (audio+video+image+text) RL training.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from transformers.modeling_outputs import ModelOutput

try:
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniThinkerForCausalLM,
    )
except ImportError:
    Qwen2_5OmniThinkerForCausalLM = None


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
    Forward pass for Qwen2.5-Omni thinker model with multimodal inputs.
    This function implements the forward logic directly to avoid infinite recursion
    when the forward method is monkey-patched.
    
    Based on the Qwen2.5-Omni thinker model's forward implementation.
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
    if use_audio_in_video is None:
        use_audio_in_video = False

    # 1. Get input embeddings
    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    # 2. Process audio features if present
    if input_features is not None and hasattr(self, 'get_audio_features'):
        audio_features = self.get_audio_features(
            input_features,
            feature_attention_mask=feature_attention_mask,
            audio_feature_lengths=audio_feature_lengths,
        )
        if hasattr(audio_features, 'last_hidden_state'):
            audio_features = audio_features.last_hidden_state
        audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
        
        if hasattr(self, 'get_placeholder_mask'):
            _, _, audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

    # 3. Process image features if present
    if pixel_values is not None and hasattr(self, 'get_image_features'):
        image_outputs = self.get_image_features(pixel_values, image_grid_thw)
        if hasattr(image_outputs, 'pooler_output'):
            image_embeds = image_outputs.pooler_output
        else:
            image_embeds = image_outputs
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        
        if hasattr(self, 'get_placeholder_mask'):
            image_mask, _, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    # 4. Process video features if present
    if pixel_values_videos is not None and hasattr(self, 'get_video_features'):
        video_outputs = self.get_video_features(pixel_values_videos, video_grid_thw)
        if hasattr(video_outputs, 'pooler_output'):
            video_embeds = video_outputs.pooler_output
        else:
            video_embeds = video_outputs
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        
        if hasattr(self, 'get_placeholder_mask'):
            _, video_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    # 5. Handle position_ids for multimodal inputs
    if attention_mask is not None and position_ids is None:
        if hasattr(self, 'get_rope_index'):
            if feature_attention_mask is not None:
                audio_feature_lengths_for_rope = torch.sum(feature_attention_mask, dim=1)
            else:
                audio_feature_lengths_for_rope = None
            
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask,
                use_audio_in_video,
                audio_feature_lengths_for_rope,
                video_second_per_grid,
            )
            self.rope_deltas = rope_deltas
        else:
            # Fallback: create position_ids from attention_mask
            batch_size, seq_length = inputs_embeds.shape[:2]
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

    # 6. Call the underlying transformer model
    outputs = self.model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **kwargs,
    )
    
    return outputs


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
        return_dict=True,  # Always use return_dict for consistency
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

    hidden_states = outputs[0]

    # Get labels for computing log_probs
    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_torch_backend, either labels or input_ids must be provided.")

    # Use FusedLinearForPPO to compute log_probs and entropy efficiently
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
        past_key_values=outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
        hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
        attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
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
        return_dict=True,
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

    hidden_states = outputs[0]

    # Get labels for computing log_probs
    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_triton_backend, either labels or input_ids must be provided.")

    # Use Triton-optimized linear_cross_entropy
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
        past_key_values=outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
        hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
        attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
    )
