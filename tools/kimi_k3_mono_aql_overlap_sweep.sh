#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# The any-order (AQL barrier bit clear) launch study on one gfx1250 GPU:
# builds the HIP microbenchmark, checks the replayed packet headers, runs the
# sweeps, then the FlyDSL probe. One JSON line a run (tagged) on stdout.
#   HIP_VISIBLE_DEVICES=0 tools/kimi_k3_mono_aql_overlap_sweep.sh | tee aql_overlap.jsonl
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=${BIN:-/tmp/aql_overlap_bench}
hipcc -O3 --offload-arch=gfx1250 -std=c++17 tools/kimi_k3_mono_aql_overlap_bench.hip -o "$BIN"

echo "# packet headers of one replay of each graph (any-order nodes: barrier=0)" >&2
AMD_LOG_LEVEL=4 AMD_LOG_MASK=0x8 NODES=4 LAYERS=1 REPS=1 REPLAYS=1 "$BIN" 2>&1 \
    | grep -o "Dispatch Header = 0x[0-9a-f]* (type=[0-9]*, barrier=[0-9]*" | tail -n 16 >&2 || true

run() {
    local tag=$1
    shift
    env "$@" "$BIN" | sed "s/^{/{\"tag\": \"$tag\", /"
}
run launch_gap DEP=none PRO_KB=0 BODY_KB=16
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
    env "$@" python tools/kimi_k3_mono_any_order_probe.py | sed "s/^{/{\"tag\": \"flydsl_$tag\", /"
}
probe base
probe prefetch PRO_KB=16
probe straggler TAIL_US=4
probe dep_one DEP=one
