"""Patch vLLM GPU model runner to write positions to ATOM buffer before each forward.

MLA attention reads positions from static_forward_context["positions"] because
vLLM's MLA interface does not pass positions as a parameter. In CUDA graph mode,
model.forward is not called (graph replay), so the write in model_wrapper never
runs. This patch adds the write in execute_model before both eager and cudagraph
paths, ensuring positions are always available when attention_mla runs.
"""

import logging

logger = logging.getLogger("atom")


def _write_positions_to_atom_buffer(model_runner, positions, num_tokens: int) -> None:
    """Write positions to ATOM static_forward_context for MLA attention."""
    if positions is None or num_tokens <= 0:
        return
    model = getattr(model_runner, "model", None)
    if model is None:
        return
    atom_config = getattr(model, "atom_config", None)
    if atom_config is None:
        return
    ctx = getattr(atom_config, "compilation_config", None)
    if ctx is None:
        return
    static_ctx = getattr(ctx, "static_forward_context", None)
    if static_ctx is None or "positions" not in static_ctx:
        return
    n = min(num_tokens, positions.numel())
    buf = static_ctx["positions"]
    if n > buf.shape[0]:
        return
    flat = positions.flatten()[:n]
    buf[:n].copy_(flat)


def _patch_v2_model_runner() -> None:
    """Patch vLLM v1 gpu.model_runner.GPUModelRunner.execute_model to write positions."""
    try:
        from vllm.config.compilation import CUDAGraphMode
        from vllm.forward_context import set_forward_context
        from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
        from vllm.v1.worker.gpu.input_batch import InputBatch
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    except ImportError:
        return

    _execute_model_orig = GPUModelRunner.execute_model

    def _execute_model_patched(
        self,
        scheduler_output,
        intermediate_tensors=None,
        dummy_run=False,
    ):
        assert intermediate_tensors is None
        if scheduler_output.total_num_scheduled_tokens == 0 and not dummy_run:
            self.update_states(scheduler_output)
            return EMPTY_MODEL_RUNNER_OUTPUT

        cudagraph_mode, num_tokens_after_padding, num_tokens_across_dp = (
            self.get_cudagraph_and_dp_padding(scheduler_output)
        )
        self.update_states(scheduler_output)
        if num_tokens_after_padding == 0:
            return EMPTY_MODEL_RUNNER_OUTPUT

        if not dummy_run:
            input_batch = self.prepare_inputs(
                scheduler_output,
                num_tokens_after_padding,
            )
            if self.lora_config:
                lora_inputs = self.req_states.make_lora_inputs(
                    input_batch.req_ids,
                    input_batch.idx_mapping_np,
                    input_batch.num_scheduled_tokens,
                )
                self._set_active_loras(*lora_inputs)
        else:
            num_reqs = min(num_tokens_after_padding, self.max_num_reqs)
            input_batch = InputBatch.make_dummy(
                num_reqs=num_reqs,
                num_tokens=num_tokens_after_padding,
                input_buffers=self.input_buffers,
                device=self.device,
            )
            if self.uses_mrope:
                input_batch.mrope_positions = self.mrope_states.mrope_positions[
                    :, :num_tokens_after_padding
                ]
            self.prepare_dummy_attn_metadata(input_batch)

        # Write positions to ATOM buffer before model run (both eager and cudagraph)
        positions = input_batch.positions
        if self.uses_mrope and input_batch.mrope_positions is not None:
            positions = input_batch.mrope_positions
        _write_positions_to_atom_buffer(
            self, positions, input_batch.num_tokens_after_padding
        )

        if cudagraph_mode == CUDAGraphMode.FULL:
            hidden_states = self.cudagraph_manager.run(
                input_batch.num_tokens_after_padding
            )
        else:
            with set_forward_context(
                input_batch.attn_metadata,
                self.vllm_config,
                num_tokens=input_batch.num_tokens_after_padding,
                cudagraph_runtime_mode=cudagraph_mode,
                num_tokens_across_dp=num_tokens_across_dp,
            ):
                hidden_states = self.model(
                    input_ids=input_batch.input_ids,
                    positions=positions,
                )

        self.execute_model_state = hidden_states, input_batch
        return None

    GPUModelRunner.execute_model = _execute_model_patched
    logger.info("ATOM positions patch: patched v2 GPUModelRunner.execute_model")


def apply_positions_patch() -> None:
    """Apply positions buffer write patch to vLLM model runner. Called from platform init."""
    _patch_v2_model_runner()
