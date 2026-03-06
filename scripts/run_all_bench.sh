#!/bin/bash
set -uo pipefail

# =============================================================
# Master Benchmark Script for Kimi-K2-Thinking-MXFP4
# Runs 3 server configs x 6 benchmark configs each
# =============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_SCRIPT="/home/xiaobizh@amd.com/bench_serving/benchmark_serving.py"
MODEL=/it-share/models/Kimi-K2-Thinking-MXFP4
RANGE_RATIO=0.8
OSL=1000
PORT=8001
RESULT_BASE="${SCRIPT_DIR}/perf_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="${RESULT_BASE}/${TIMESTAMP}"

export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER=1
export VLLM_RPC_TIMEOUT=1800000
export VLLM_CACHE_ROOT=/root/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/inductor
export HIP_VISIBLE_DEVICES=4,5,6,7
export AITER_QUICK_REDUCE_QUANTIZATION=INT4

# ======================== Helpers ========================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

wait_for_server() {
    local max_wait=2700
    local elapsed=0
    log "Waiting for server to be ready on port ${PORT}..."
    while [ $elapsed -lt $max_wait ]; do
        if curl -s --max-time 5 http://localhost:${PORT}/health > /dev/null 2>&1; then
            log "Server is ready! (took ${elapsed}s)"
            return 0
        fi
        sleep 15
        elapsed=$((elapsed + 15))
        if (( elapsed % 60 == 0 )); then
            log "  ...still waiting (${elapsed}s elapsed)"
        fi
    done
    log "ERROR: Server did not become ready within ${max_wait}s"
    return 1
}

kill_tree() {
    local pid=$1
    local children
    children=$(pgrep -P "$pid" 2>/dev/null || true)
    for child in $children; do
        kill_tree "$child"
    done
    kill -9 "$pid" 2>/dev/null || true
}

kill_server() {
    log "Stopping server processes..."

    if [ -n "${SERVER_PID:-}" ]; then
        log "  Killing process tree rooted at SERVER_PID=${SERVER_PID}"
        kill_tree "$SERVER_PID"
        SERVER_PID=""
    fi

    local pids
    pids=$(lsof -ti:${PORT} 2>/dev/null || true)
    if [ -n "$pids" ]; then
        log "  Killing PIDs on port ${PORT}: $pids"
        for p in $pids; do kill_tree "$p"; done
    fi

    pkill -9 -f "vllm serve.*Kimi-K2" 2>/dev/null || true
    pkill -9 -f "atom.entrypoints.openai_server" 2>/dev/null || true
    pkill -9 -f "from multiprocessing.spawn import spawn_main" 2>/dev/null || true
    pkill -9 -f "from multiprocessing.resource_tracker import" 2>/dev/null || true
    pkill -9 -f "torch._inductor.compile_worker" 2>/dev/null || true

    sleep 10
    log "Server stopped."
}

run_benchmark() {
    local isl=$1
    local osl=$2
    local con=$3
    local num=$((con * 4))
    local log_file=$4
    local label=$5

    log "============================================"
    log "  ${label}"
    log "  ISL=${isl}  OSL=${osl}  CON=${con}  NUM=${num}"
    log "  Log: ${log_file}"
    log "============================================"

    python "${BENCH_SCRIPT}" \
        --model=${MODEL} \
        --backend=vllm \
        --base-url=http://localhost:${PORT} \
        --dataset-name=random \
        --random-input-len=${isl} \
        --random-output-len=${osl} \
        --random-range-ratio ${RANGE_RATIO} \
        --num-prompts=${num} \
        --max-concurrency=${con} \
        --request-rate=inf \
        --ignore-eos \
        --save-result \
        --percentile-metrics="ttft,tpot,itl,e2el" \
        --result-dir="${cur_result_dir}" \
        --trust-remote-code \
        2>&1 | tee "${log_file}"

    log "Benchmark finished: ${label}"
    echo ""
}

# ======================== Server Launchers ========================

start_atom_server() {
    log "Starting ATOM server..."
    unset ATOM_DISABLE_VLLM_PLUGIN 2>/dev/null || true

    python -m atom.entrypoints.openai_server \
        --model ${MODEL} \
        --trust-remote-code \
        -tp 4 \
        --kv_cache_dtype fp8 \
        --gpu-memory-utilization 0.9 \
        --server-port ${PORT} \
        > "${cur_result_dir}/server.log" 2>&1 &
    SERVER_PID=$!
    log "ATOM server started with PID=${SERVER_PID}"
}

start_vllm_server() {
    log "Starting vLLM server (plugin disabled)..."
    export ATOM_DISABLE_VLLM_PLUGIN=1

    vllm serve ${MODEL} \
        --host localhost \
        --port ${PORT} \
        --tensor-parallel-size 4 \
        --enable-expert-parallel \
        --trust-remote-code \
        --disable-log-requests \
        --gpu_memory_utilization 0.9 \
        --async-scheduling \
        --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
        --kv-cache-dtype fp8 \
        --max-num-batched-tokens 18432 \
        --max-model-len 16384 \
        --no-enable-prefix-caching \
        > "${cur_result_dir}/server.log" 2>&1 &
    SERVER_PID=$!
    log "vLLM server started with PID=${SERVER_PID}"
}

start_vllm_atom_server() {
    log "Starting vLLM+ATOM server (plugin enabled)..."
    export ATOM_DISABLE_VLLM_PLUGIN=0

    vllm serve ${MODEL} \
        --host localhost \
        --port ${PORT} \
        --tensor-parallel-size 4 \
        --enable-expert-parallel \
        --trust-remote-code \
        --disable-log-requests \
        --gpu_memory_utilization 0.9 \
        --async-scheduling \
        --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
        --kv-cache-dtype fp8 \
        --max-num-batched-tokens 18432 \
        --max-model-len 16384 \
        --no-enable-prefix-caching \
        > "${cur_result_dir}/server.log" 2>&1 &
    SERVER_PID=$!
    log "vLLM+ATOM server started with PID=${SERVER_PID}"
}

# ======================== Main ========================

SERVERS=("atom" "vllm" "vllm_atom")

log "######################################################"
log "  Kimi-K2 Performance Benchmark Suite"
log "  Results directory: ${RESULT_DIR}"
log "  Servers to test: ${SERVERS[*]}"
log "######################################################"
echo ""

mkdir -p "${RESULT_DIR}"

for SERVER_NAME in "${SERVERS[@]}"; do
    echo ""
    log "######################################################"
    log "  SERVER: ${SERVER_NAME}"
    log "######################################################"
    echo ""

    cur_result_dir="${RESULT_DIR}/${SERVER_NAME}"
    mkdir -p "${cur_result_dir}"

    kill_server

    case ${SERVER_NAME} in
        atom)       start_atom_server ;;
        vllm)       start_vllm_server ;;
        vllm_atom)  start_vllm_atom_server ;;
    esac

    if ! wait_for_server; then
        log "FAILED to start ${SERVER_NAME}, skipping..."
        kill_server
        continue
    fi

    # --- Warmup (2 rounds for stable results) ---
    log ">>> Running WARMUP round 1/2 (results discarded) <<<"
    run_benchmark 1000 1000 128 "${cur_result_dir}/warmup_1.log" "[WARMUP 1/2] ${SERVER_NAME}"

    log ">>> Running WARMUP round 2/2 (results discarded) <<<"
    run_benchmark 1000 1000 128 "${cur_result_dir}/warmup_2.log" "[WARMUP 2/2] ${SERVER_NAME}"

    # --- ISL=1K, CON=256 ---
    run_benchmark 1000 ${OSL} 256 \
        "${cur_result_dir}/bench_isl1k_osl1k_con256.log" \
        "${SERVER_NAME} | ISL=1K OSL=1K CON=256"

    # --- ISL=1K, CON=128 ---
    run_benchmark 1000 ${OSL} 128 \
        "${cur_result_dir}/bench_isl1k_osl1k_con128.log" \
        "${SERVER_NAME} | ISL=1K OSL=1K CON=128"

    # --- ISL=4K, CON=128 ---
    run_benchmark 4000 ${OSL} 128 \
        "${cur_result_dir}/bench_isl4k_osl1k_con128.log" \
        "${SERVER_NAME} | ISL=4K OSL=1K CON=128"

    # --- ISL=4K, CON=64 ---
    run_benchmark 4000 ${OSL} 64 \
        "${cur_result_dir}/bench_isl4k_osl1k_con64.log" \
        "${SERVER_NAME} | ISL=4K OSL=1K CON=64"

    # --- ISL=10K, CON=64 ---
    run_benchmark 10000 ${OSL} 64 \
        "${cur_result_dir}/bench_isl10k_osl1k_con64.log" \
        "${SERVER_NAME} | ISL=10K OSL=1K CON=64"

    # --- ISL=10K, CON=32 ---
    run_benchmark 10000 ${OSL} 32 \
        "${cur_result_dir}/bench_isl10k_osl1k_con32.log" \
        "${SERVER_NAME} | ISL=10K OSL=1K CON=32"

    kill_server

    log "######################################################"
    log "  COMPLETED: ${SERVER_NAME}"
    log "  Results in: ${cur_result_dir}/"
    log "######################################################"
    echo ""
done

log "======================================================"
log "  ALL BENCHMARKS COMPLETE!"
log "  Results: ${RESULT_DIR}/"
log ""
log "  Directory structure:"
log "    ${RESULT_DIR}/atom/         - ATOM server results"
log "    ${RESULT_DIR}/vllm/         - vLLM server results"
log "    ${RESULT_DIR}/vllm_atom/    - vLLM+ATOM server results"
log "======================================================"
