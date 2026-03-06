export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER=1
export VLLM_RPC_TIMEOUT=1800000
export VLLM_CACHE_ROOT=/root/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/inductor
export HIP_VISIBLE_DEVICES=4,5,6,7
# quick allreduce
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
model_path=/it-share/models/Kimi-K2-Thinking-MXFP4

export ATOM_PROFILER_MORE=1

python -m atom.entrypoints.openai_server  \
    --model $model_path \
    --trust-remote-code \
    -tp 4 \
    --kv_cache_dtype fp8 \
    --gpu-memory-utilization 0.9 \
    --max-num-batched-tokens 18432 \
    --server-port 8001 \
    --torch-profiler-dir  "/it-share/xiaobing/profiler_trace/atom"
    2>&1 | tee log_atom.log
