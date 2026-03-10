"""Compatibility shim for the legacy dotted filename.

The PPO runtime monkey-patch uses `verl.models.transformers.qwen2_5_omni`.
Keeping the implementation in one place avoids divergence in multimodal forward
logic and prevents recursive calls if this file is loaded by mistake.
"""

from verl.models.transformers.qwen2_5_omni import (
    Qwen2_5OmniThinkerForCausalLMOutputForPPO,
    forward_base_model,
    forward_with_torch_backend,
    forward_with_triton_backend,
    get_rope_index,
)


Qwen2_5OmniThinkerForConditionalGenerationOutputForPPO = Qwen2_5OmniThinkerForCausalLMOutputForPPO
