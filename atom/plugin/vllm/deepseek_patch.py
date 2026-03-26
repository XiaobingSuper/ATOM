import functools
import logging
from atom.utils import envs
from atom.utils.custom_register import direct_register_custom_op
import torch
from atom.config import get_current_atom_config
logger = logging.getLogger("atom")


def _current_vllm_forward_uses_full_decode_cudagraph() -> bool:
    """True only for vLLM uniform-decode forwards running with FULL cudagraph."""
    try:
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        if not is_forward_context_available():
            return False

        forward_context = get_forward_context()
        runtime_mode = forward_context.cudagraph_runtime_mode
        batch_descriptor = forward_context.batch_descriptor
        return (
            getattr(runtime_mode, "name", str(runtime_mode)) == "FULL"
            and batch_descriptor is not None
            and bool(getattr(batch_descriptor, "uniform", False))
        )
    except Exception:
        return False



def dual_stream_moe_forward(
    self,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    num_tokens, hidden_dim = hidden_states.shape
    current_stream = torch.cuda.current_stream()
    alt_stream = self.alt_stream

    alt_stream.wait_stream(current_stream)

    with torch.cuda.stream(alt_stream):
        final_hidden_states = self.routed_expert_forward(hidden_states)

    shared_output = self.shared_experts(hidden_states)

    current_stream.wait_stream(alt_stream)

    final_hidden_states = self.combine_outputs(
        final_hidden_states, shared_output, hidden_states
    )
    return final_hidden_states.view(num_tokens, hidden_dim)


def deepseek_v2_moe_forward(
    hidden_states: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    """Dual-stream MoE forward: shared experts on alt stream, routed on main."""
    atom_config = get_current_atom_config()
    self = atom_config.compilation_config.static_forward_context[layer_name]
    DUAL_STREAM_TOKEN_THRESHOLD = envs.ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD
    num_tokens, _ = hidden_states.shape
    if (
        self._use_dual_stream
        and num_tokens > 0
        and num_tokens <= DUAL_STREAM_TOKEN_THRESHOLD
        and _current_vllm_forward_uses_full_decode_cudagraph()
    ):
        return self.dual_stream_moe_forward(hidden_states)
    else:
        return self.single_stream_moe_forward(hidden_states)


def deepseek_v2_moe_forward_fake(
    hidden_states: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="deepseek_v2_moe_forward",
    op_func=deepseek_v2_moe_forward,
    mutates_args=["hidden_states"],
    fake_impl=deepseek_v2_moe_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)

def _patch_deepseek_v2_moe_dual_stream() -> None:
    from atom.models.deepseek_v2 import DeepseekV2MoE

    orig_forward = DeepseekV2MoE.forward
    if getattr(orig_forward, "_atom_vllm_dual_stream_patched", False):
        return

    @functools.wraps(orig_forward)
    def _forward(self, hidden_states):
        if not getattr(self, "_use_dual_stream", False):
            return orig_forward(self, hidden_states)

        return torch.ops.aiter.deepseek_v2_moe_forward(hidden_states, self.prefix)

    setattr(_forward, "_atom_vllm_dual_stream_patched", True)
    DeepseekV2MoE.forward = _forward


def patch_vllm_deepseek_dual_stream() -> None:
    _patch_deepseek_v2_moe_dual_stream()
