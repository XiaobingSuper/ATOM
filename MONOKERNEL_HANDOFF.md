# Native Decode MonoKernel Handoff

> Current status (2026-09-29): implementation and the focused/device gates in
> this document are complete on `feat/monokernel-integration`. The old cluster
> blocker and continuation checklist later in this file are retained as
> historical context and are superseded by this status block.
>
> - Kimi-K3 real weights: staged DCP1/FP32 and Agentic DCP8/FP16 graph decode,
>   plus explicit mono S4 graph/S8 eager, all returned the expected `333`.
> - GLM-5.2 real weights: TP4 FP8-KV staged MoE passed no-MTP, MTP5, and
>   DCP4+MTP4; graph supersets cover every C2–C48 recipe bucket and returned
>   `333` without forced acceptance.
> - Focused suite: `79 passed`; compileall, `git diff --check`, and IDE lints
>   are clean. The untracked `diff.diff` remains untouched.
> - Performance: Kimi candidate reduced GPU work/launch count in the recorded
>   C32 trace. GLM's first A16W4 stage regressed S8 TPOT by ~49%; profile showed
>   slower expert GEMM plus generic top-k launches. The retained stage uses the
>   production A4W4 expert pipeline and fused FP32 biased top-k instead:
>   dummy S8 is 9.2–9.4 ms vs 9.3 ms baseline, and real Agentic C2 median TPOT
>   is 4.20 ms vs 4.05 ms baseline (within the non-regression gate, not a
>   speedup claim).
> - Remaining performance boundary: the large 2× layer gains from FlyDSL #1204
>   are whole-layer TP8/contiguous-cache results. Agentic GLM keeps FP8 paged
>   attention/IndexShare/MTP/DCP outside the staged MoE segment, so this branch
>   must not claim those whole-layer gains. Kimi MLA layers likewise retain
>   ATOM's paged attention path; KDA layers are the native accelerated segment.

## Goal and ownership decision

Continue the MI355 integration of the fused decode-layer kernels from
[FlyDSL PR #1204](https://github.com/ROCm/FlyDSL/pull/1204) and
[FlyDSL PR #1205](https://github.com/ROCm/FlyDSL/pull/1205) into ATOM for
Kimi-K3 and GLM-5.2. PR #1204 is the source baseline and includes the GLM work
from #1205; the imported device sources track FlyDSL commit `21a3d1ee`.

The settled ownership decision is that these kernels live directly in ATOM under
`atom/model_ops/monokernel/`. Existing AITER APIs remain dependencies, but this
integration makes no AITER source changes and does not require an AITER PR.

## Repository and assets

- Worktree: `/shared_nfs/xiaobizh/monokernel-integration-20260929/atom`
- Base: `aa0c5c3a`
- Branch: `feat/monokernel-integration`
- Intended push target: `XiaobingSuper/ATOM` (configured as remote `fork`)
- GLM weights: `/shared_nfs/models/GLM-5.2-MXFP4`
- Kimi weights: `/shared_nfs/hyperloom/models/Kimi-K3`

This branch is a handoff checkpoint. Commit and push it as requested, but do
not rebase, open a PR, or claim validation until the GPU gates below pass.

## Stable dispatch contract

`ATOM_NATIVE_DECODE_MONOKERNEL` is resolved once during model construction. It
is stable, defaults to `off`, and accepts only:

- `off`: byte-for-byte existing ATOM path.
- `auto`: Kimi selects `staged`; GLM selects TP4 `staged_moe` for Agentic
  deployments and the legacy TP8 `mono` where its stricter gates pass.
- `mono`: request the single-launch backend where supported.
- `staged`: request Kimi staged KDA or GLM TP4 staged MoE.

Unsupported models, layers, forwards, layouts, or deployment modes use the
baseline path. Do not infer enablement from tensor dtype or shape and do not
weaken a gate merely to make a test route.

Relevant dispatch and wiring:

- `atom/utils/envs.py`
- `atom/model_ops/monokernel/dispatch.py`
- `atom/models/kimi_k3.py` and `atom/models/kimi_k3_mono.py`
- `atom/models/deepseek_v2.py` and `atom/models/glm52_mono.py`
- `docs/environment_variables.md`

## Implemented scope

### Shared kernel port

- ATOM-owned FlyDSL sources are compatible with the deployed FlyDSL
  `0.3.4.1` ABI.
- Shared packing, weight mapping, IPC/symmetric buffers, all-reduce, GEMM, and
  model kernels are under `atom/model_ops/monokernel/`.
- The focused contracts are in `tests/test_native_monokernel.py`.

### Kimi-K3

- Native TP8 decode for exactly S=4 or S=8 KDA+MoE layers.
- Both `staged` and `mono` KDA backends are present; `auto` selects `staged`.
- Deployment construction accepts BF16 or FP8 KV cache. The native path handles
  FP16 or FP32 KDA state and TP8+DCP8; full-attention MLA layers stay on the
  baseline ATOM paged-attention path.
- Prefill, ReplaySSM/speculative decode, non-S4/S8 batches, non-TP8, DP/DPA,
  PP, plugins, graph-time construction, and failed rank-local mappings fall
  back.

Primary files are `atom/model_ops/monokernel/k3/` and
`atom/models/kimi_k3_mono.py`. The unchanged production launch authority is
`recipes/Kimi-K3.md`.

### GLM-5.2

- Agentic TP4 `staged_moe` wraps the existing 75 `DeepseekV2MoE` modules
  without changing parameter names. It uses FP32 biased routing, 256 routed
  MXFP4 experts plus the physically fused shared expert, and the production
  A4W4 expert pipeline.
- Existing FP8 KV, full/shared IndexShare, MTP acceptance, DCP/QREP/LSE merge,
  residual, and fused next-layer TP-reduce/RMSNorm contracts remain baseline
  owned. Flattened MoE rows are prepared before graph capture.
- The exact C2–C48 Agentic graph matrix is supported. Plugins, prefill, TP/EP
  layouts outside the declared stage contract, and failed all-rank mappings
  retain the original MoE forward.
- The legacy TP8/BF16 whole-layer S4/S8 path remains available for functional
  comparison. Its external-indexer, padded-row, and physical-cache safeguards
  are unchanged.

Primary files are `atom/model_ops/monokernel/glm/` and
`atom/models/glm52_mono.py`. Recipe context is in `recipes/GLM-5.md` and
`recipes/Agentic-GLM-5.2.md`; the published TP4 FP8-KV/MTP/DCP recipe now
routes its MoE segment through `staged_moe`.

## Historical baseline evidence

These are dummy/single-layer gates, not end-to-end performance claims:

- Kimi on FlyDSL `0.3.4.1`, mono S4: **145.935 us**.
- Kimi on FlyDSL `0.3.4.1`, staged S4: **114.146 us**.
- Kimi on FlyDSL `0.3.4.1`, mono S8: **226.491 us**.
- Kimi on FlyDSL `0.3.4.1`, staged S8: **129.330 us**.
- Kimi output relative L2: **0.003707** at S4 and **0.003652** at S8.
- The upstream GLM dummy harness reported PASS at S4/S8 and **78.7 us /
  117.6 us**. That run predates the final ATOM paged-cache/IndexShare adapter;
  it is explicitly not final GLM evidence.

The last static pass used:

```bash
python3 -m compileall -q \
  atom/model_ops/monokernel \
  atom/models/kimi_k3_mono.py \
  atom/models/glm52_mono.py \
  tests/test_native_monokernel.py
git diff --check
```

A prior focused snapshot produced **23 passed** with:

```bash
pytest -q tests/test_native_monokernel.py tests/test_kimi_k3_plugin_config.py
```

That pytest result was before later model-adapter and paged-cache additions. The
current focused tests, full suite, and final GPU tests have not run and remain
required.

## Historical blocker (resolved)

After the SPUR controller restart, scheduler authentication is invalid for both
`gyu` and `xiaobizh` on the login nodes. The exact administrator-facing error is:

```text
The request does not have valid authentication credentials
authentication required: pass a token (see `spur token user`)
```

This blocks `squeue`, `srun`, and `sbatch`. Job `179589` on node274 had service
health still reachable, so the allocation may survive, but this is not shell
access. It was created as a plain `sleep` allocation and no user `sshd` was
started on port 22200; system port 22 does not admit these users.

## Historical continuation order (superseded)

1. Ask the cluster administrator to restore `spurauthd` or reissue SPUR
   credentials for both users. Verify `squeue`, `srun`, and `sbatch`, not only
   login-node SSH.
2. If job `179589` still survives, immediately start its user SSH service from
   the owning login account:

   ```bash
   srun --jobid=179589 --overlap -n1 bash ~/bin/node-sshd.sh
   ```

3. If it does not survive, declare the desired MI355 hold in the `HOLDS` table
   and allocate through the keeper:

   ```bash
   bash ~/bin/keep-all.sh
   bash ~/bin/node-status.sh
   ```

   Do not submit a manual `sbatch` sleep allocation. Confirm eight clean GPUs,
   Docker access, NFS visibility, and `sshd:22200` before testing.
4. Use the image already verified to contain FlyDSL `0.3.4.1`
   (`rocm/atom-dev:nightly_202609271453`) and confirm the package version inside
   the container before compiling.
5. Run the focused tests above on the current worktree, then the relevant full
   suite. Fix only failures caused by this integration.
6. Run `--load_dummy=xavier --enforce-eager` functional gates for both models,
   with the flag `off`, then `auto`; prove from logs/counters that S4 and S8 hit
   the intended backend and unsupported cases really fall back. Cover Kimi
   BF16 and FP8 KV deployment variants. Cover GLM TP8 BF16 paged cache, full and
   shared IndexShare layers, and padded graph rows.
7. Repeat the supported dummy gates under graph capture after eager startup and
   kernel warmup. A successful server start alone is not evidence of routing.
8. Load the real checkpoints from the paths above and run accuracy before
   performance. Compare with the flag `off` and `auto` using the same prompts,
   seeds, concurrency, and evaluation recipe.
9. Run Kimi's final performance A/B on the same uncontended MI355 node, with
   real weights and graph mode. Use the exact published `recipes/Kimi-K3.md`
   launch and workload unchanged; between paired runs change only:

   ```bash
   export ATOM_NATIVE_DECODE_MONOKERNEL=off   # baseline
   export ATOM_NATIVE_DECODE_MONOKERNEL=auto  # candidate
   ```

10. Make no final GLM performance claim for this first-stage path. Its TP8 BF16,
    no-MTP/no-DPA/no-DCP custom invocation is a functional gate only, not final
    performance evidence. Before a GLM performance claim, extend the
    implementation to support one exact published ATOM GLM recipe, including
    that recipe's FP8 KV, MTP, DPA/DCP, and TP constraints as applicable. Then
    run that published launch and workload unchanged for the same-node,
    real-weight, graph-mode flag-only `off`/`auto` A/B.

## Historical claim boundary (superseded)

- Latest adapter code has not completed focused pytest or GPU compilation.
- Kimi BF16/FP8 deployment routing, actual checkpoint weight mapping, mixed
  KDA/MLA layer transitions, graph replay, and TP8 IPC/all-reduce lifetime are
  not finally verified.
- GLM's final paged adapter, external full/shared IndexShare reuse, physical-row
  addressing, padded-row no-write behavior, and graph replay are not GPU
  verified.
- All-rank refusal/fallback must be shown not to deadlock or partially launch.
- Real-weight accuracy, memory stability, startup/teardown, and final same-node
  graph performance A/B are pending.
- Dummy and single-layer timings are gates only. Kimi final numbers require
  its exact published recipe, the same node, graph mode, real weights, and only
  the flag changed. The custom GLM TP8/BF16 invocation remains functional-only;
  GLM has no final number until an exact published recipe is supported and run
  unchanged under the same conditions.
- There is no PR-ready claim yet. Do not open or describe a PR as validated
  until every gate above has reproducible logs and measured numbers.