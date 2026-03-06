export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER=1
export VLLM_RPC_TIMEOUT=1800000

export VLLM_CACHE_ROOT=/root/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/inductor

# export HIP_VISIBLE_DEVICES=0,1,2,3
# export HIP_VISIBLE_DEVICES=4,5,6,7
# quick allreduce
export AITER_QUICK_REDUCE_QUANTIZATION=INT4

export ATOM_DISABLE_VLLM_PLUGIN=0

export ATOM_PROFILER_MORE=1

export VLLM_TORCH_PROFILER_RECORD_SHAPES=1

model_path=/it-share/models/Kimi-K2-Thinking-MXFP4
# export  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# model_path=/it-share/models/deepseek-ai/DeepSeek-R1-0528

# --max-num-batched-tokens 18432 \
#    --async-scheduling \
#    --enforce-eager \
vllm serve $model_path \
    --host localhost \
    --port 8001 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --trust-remote-code \
    --disable-log-requests \
    --gpu_memory_utilization 0.9 \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --kv-cache-dtype fp8 \
    --max-num-batched-tokens 18432 \
     --async-scheduling \
    --max-model-len 16384 \
    --no-enable-prefix-caching \
    --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/it-share/xiaobing/profiler_trace/oot_new"}' \
    2>&1 | tee log_kimi_k2_vllm.log