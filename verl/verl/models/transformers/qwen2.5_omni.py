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

import torch
from torch import nn
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput

try:
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniThinkerCausalLMOutputWithPast
except ImportError:
    Qwen2_5OmniThinkerForConditionalGeneration = None
    Qwen2_5OmniThinkerCausalLMOutputWithPast = None

@dataclass
class Qwen2_5OmniThinkerForConditionalGenerationOutputForPPO(ModelOutput):
    """Output class for PPO training with Qwen2.5-Omni model (text output only, no voice)."""
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
        output_router_logits: bool | None = None,
        use_audio_in_video=None,
        cache_position=None,
        video_second_per_grid=None,
        **kwargs,
    ) -> tuple | Qwen2_5OmniThinkerCausalLMOutputWithPast:
    """
    Forward pass for Qwen2.5-Omni base model with multimodal inputs.
    
    Args:
        self: Qwen2_5OmniThinkerForConditionalGeneration instance
        input_ids: Text token IDs
        input_features: Audio mel-spectrogram features
        pixel_values: Image pixel values
        pixel_values_videos: Video pixel values
        image_grid_thw: Image grid (temporal, height, width)
        video_grid_thw: Video grid (temporal, height, width)
        attention_mask: Attention mask for text
        feature_attention_mask: Attention mask for audio features
        audio_feature_lengths: Lengths of audio features
        position_ids: Position IDs for rotary embeddings
        past_key_values: Cached key-value pairs
        inputs_embeds: Input embeddings (if provided instead of input_ids)
        rope_deltas: RoPE deltas for multimodal inputs
        labels: Labels for language modeling loss
        use_cache: Whether to use KV cache
        output_router_logits: Whether to output router logits (for MoE models)
        use_audio_in_video: Whether to extract audio from video
        cache_position: Cache position for generation
        video_second_per_grid: Seconds per grid for video temporal encoding
        **kwargs: Additional arguments
    
    Returns:
        Qwen2_5OmniThinkerCausalLMOutputWithPast with logits, past_key_values, hidden_states, attentions
    """
    output_attentions = kwargs.get('output_attentions', False)
    output_hidden_states = kwargs.get('output_hidden_states', False)
    return_dict = kwargs.get('return_dict', True)
    
    if output_router_logits is None:
        output_router_logits = False
    
    if use_audio_in_video is None:
        use_audio_in_video = False
        
    outputs = self.forward(
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
        labels=labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        use_audio_in_video=use_audio_in_video,
        cache_position=cache_position,
        video_second_per_grid=video_second_per_grid,
        **kwargs,
    )
    
    return outputs


def forward_with_torch_backend(
    self: Qwen2_5OmniThinkerForConditionalGeneration,
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
    output_router_logits: Optional[bool] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    **loss_kwargs,
) -> Union[Tuple, Qwen2_5OmniThinkerForConditionalGenerationOutputForPPO]:
    """
    Forward pass with PyTorch backend for PPO training.
    Computes log probabilities and entropy for policy optimization.
    """
    # Get base model outputs
    outputs = forward_base_model(
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
        labels=labels,
        use_cache=use_cache,
        output_router_logits=output_router_logits,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        use_audio_in_video=use_audio_in_video,
        cache_position=cache_position,
        video_second_per_grid=video_second_per_grid,
        **loss_kwargs,
    )
    
    logits = outputs.logits
    
    # Apply temperature scaling
    if temperature != 1.0:
        logits = logits / temperature
    
    # Compute log probabilities
    log_probs = torch.log_softmax(logits, dim=-1)
    
    # Get log probs for actual tokens (labels)
    if labels is not None:
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_log_probs = log_probs[..., :-1, :].contiguous()
        
        # Gather log probs for the actual tokens
        token_log_probs = torch.gather(
            shift_log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Compute entropy: -sum(p * log(p))
        probs = torch.softmax(shift_logits, dim=-1)
        entropy = -(probs * shift_log_probs).sum(dim=-1)
    else:
        token_log_probs = None
        entropy = None
    
    return Qwen2_5OmniThinkerForConditionalGenerationOutputForPPO(
        log_probs=token_log_probs,
        entropy=entropy,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def forward_with_triton_backend(
    self: Qwen2_5OmniThinkerForConditionalGeneration,
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
    output_router_logits: Optional[bool] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    **loss_kwargs,
) -> Union[Tuple, Qwen2_5OmniThinkerForConditionalGenerationOutputForPPO]:
    """
    Forward pass with Triton backend for PPO training.
    Uses fused kernels for better performance.
    
    Note: Currently falls back to torch backend as Triton implementation is WIP.
    """
    # For now, use torch backend
    # TODO: Implement Triton-optimized version with fused softmax/cross-entropy
    return forward_with_torch_backend(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        input_features=input_features,
        feature_attention_mask=feature_attention_mask,
        audio_feature_lengths=audio_feature_lengths,
        use_audio_in_video=use_audio_in_video,
        video_second_per_grid=video_second_per_grid,
        output_router_logits=output_router_logits,
        rope_deltas=rope_deltas,
        cache_position=cache_position,
        second_per_grid_ts=second_per_grid_ts,
        temperature=temperature,
        **loss_kwargs,
    )


def ulysses_flash_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Ulysses sequence parallel flash attention forward pass.
    
    This is used for long-context multimodal sequences.
    Currently a placeholder - defaults to standard attention.
    """
    # TODO: Implement Ulysses sequence parallelism for Qwen2.5-Omni
    # For now, use the standard forward
    return self._original_forward(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
