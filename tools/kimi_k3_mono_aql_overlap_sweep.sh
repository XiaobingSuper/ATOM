#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# The any-order (AQL barrier bit clear) launch study on one gfx1250 GPU:
# builds the HIP microbenchmark, runs the sweeps (every line carries the
# replayed packets' headers, read in-kernel), then the FlyDSL probe. One JSON
# line a run (tagged) on stdout; the first failing or hung run stops it.
#   HIP_VISIBLE_DEVICES=0 tools/kimi_k3_mono_aql_overlap_sweep.sh | tee aql_overlap.jsonl
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=${BIN:-/tmp/aql_overlap_bench}
LIMIT=${LIMIT:-120}
hipcc -O3 --offload-arch=gfx1250 -std=c++17 -Wno-unused-value \
    -c tools/kimi_k3_mono_aql_overlap_bench.hip -o "$BIN.o"
HIP_LIB=$(readlink -f "$(hipconfig --path 2>/dev/null || echo /opt/rocm)/lib/libamdhip64.so.7")
"$(hipconfig --path 2>/dev/null || echo /opt/rocm)/lib/llvm/bin/clang++" "$BIN.o" "$HIP_LIB" \
    -Wl,-rpath,"$(dirname "$HIP_LIB")" -o "$BIN"

run() {
    local tag=$1
    shift
    env "$@" timeout -s INT "$LIMIT" "$BIN" | sed "s/^{/{\"tag\": \"$tag\", /"
}
run launch_gap DEP=none PRO_KB=0 BODY_KB=16
run small BODY_KB=16
run small_no_prefetch BODY_KB=16 PRO_KB=0
run small_dep_one BODY_KB=16 DEP=one
run small_straggler_mem BODY_KB=16 TAIL_CTAS=8 TAIL_KB=64
run small_lds320 BODY_KB=16 LDS_KB=320
run base
run dep_one DEP=one
run all_any ANY=all
run no_prefetch PRO_KB=0
run lds160 LDS_KB=160
run lds320 LDS_KB=320
run straggler_mem TAIL_CTAS=8 TAIL_KB=256
run straggler_alu TAIL_CTAS=8 TAIL_ITERS=20000
run straggler_lds320 TAIL_CTAS=8 TAIL_KB=256 LDS_KB=320
run alu FMA_ITERS=4000
run half_grid GRID=128
run direct STREAM=1

probe() {
    local tag=$1
    shift
    env "$@" timeout -s INT "$LIMIT" python tools/kimi_k3_mono_any_order_probe.py \
        | sed "s/^{/{\"tag\": \"flydsl_$tag\", /"
}
probe base
probe prefetch PRO_KB=16
probe straggler TAIL_US=4
probe dep_one DEP=one
