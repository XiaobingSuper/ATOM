# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL launches with the AQL barrier bit clear (``hipExtAnyOrderLaunch``).

HIP sets the barrier bit on every kernel packet of a stream (it forces
``GPU_NUM_MEM_DEPENDENCY=0``), so a packet starts only after the one before it
completes: on gfx1250 ~1.5 us from a kernel's last CTA to the next one's first.
A packet without the bit may start as soon as the one before it has
dispatched. The one switch HIP has is ``hipExtAnyOrderLaunch`` in the flags of
``hipExtModuleLaunchKernel``; on a capturing stream the graph's kernel node
keeps it (CLR ``GraphKernelNode::launchFlags_``, ROCm builds from 2026-07-29
on; earlier ones drop it at capture) and the replayed packet keeps the clear
bit. FlyDSL launches through ``mgpuLaunchKernel`` and, for a kernel built
with cluster dims, ``mgpuLaunchClusterKernel`` (``libfly_jit_runtime``:
``hipModuleLaunchKernel`` with no flags, ``hipDrvLaunchKernelEx`` with the
cluster attribute). ``install()`` registers replacements for both in every
FlyDSL ``ExecutionEngine`` built after it: inside ``any_order()`` they launch
with ``hipExtAnyOrderLaunch`` (whose ``ihipLaunchKernelCommand`` takes the
clusters from the code object's ``.cluster_dims``, which FlyDSL's
``rocdl.cluster_dims`` sets), elsewhere through FlyDSL's own entry points. Every launch of such an engine
goes through a Python callback, microseconds paid at graph capture and not at
replay, so ``install()`` belongs behind a switch.

An any-order kernel enforces its dependencies on the kernel before it itself:
it waits for the producer's epoch-tagged device-scope flags, and loads the
producer's output at device scope, which the producer stored at device scope
(the consumer's packet acquire runs before the producer's release). It must
not overwrite anything the kernel before it still reads, the kernel after it
launches in order (its barrier then waits for both), and HIP's system-scope
fence mode (``AMD_OPT_FLUSH=0``) puts the bit back.
"""

import contextlib
import ctypes
import sys
import threading

ANY_ORDER_LAUNCH = 0x1  # hipExtAnyOrderLaunch, hip_runtime_api.h

_LAUNCH = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,  # hipFunction_t
    *(ctypes.c_int64,) * 6,  # grid x, y, z (blocks); block x, y, z
    ctypes.c_int32,  # dynamic LDS bytes
    ctypes.c_void_p,  # stream
    ctypes.c_void_p,  # kernel params
    ctypes.c_void_p,  # extra
    ctypes.c_size_t,  # params count (unused, as in libfly_jit_runtime)
)

_CLUSTER_LAUNCH = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,  # hipFunction_t
    *(ctypes.c_int64,) * 9,  # cluster x, y, z; grid x, y, z (blocks); block x, y, z
    ctypes.c_int32,  # dynamic LDS bytes
    ctypes.c_void_p,  # stream
    ctypes.c_void_p,  # kernel params
    ctypes.c_void_p,  # extra
    ctypes.c_size_t,  # params count
)

_local = threading.local()
_lock = threading.Lock()
_hip = None
_callbacks = None
_fly = None
_flags_buffers: dict = {}
failures: list[int] = []


def _hip_library():
    """The HIP runtime this process already runs (torch's), by its path."""
    with open("/proc/self/maps") as maps:
        for line in maps:
            if "libamdhip64.so" in line:
                return ctypes.CDLL(line.split()[-1])
    return ctypes.CDLL("libamdhip64.so")


def _fly_library():
    """FlyDSL's JIT runtime, whose launch entry points the engines bind."""
    import os

    import flydsl

    return ctypes.CDLL(
        os.path.join(
            os.path.dirname(flydsl.__file__),
            "_mlir",
            "_mlir_libs",
            "libfly_jit_runtime.so",
        )
    )


def _any_order(f, gx, gy, gz, bx, by, bz, lds, stream, params, extra, flags):
    # hipExtModuleLaunchKernel takes the global size in threads
    err = _hip.hipExtModuleLaunchKernel(
        f, gx * bx, gy * by, gz * bz, bx, by, bz, lds, stream, params, extra,
        None, None, flags,
    )  # fmt: skip
    if err:
        failures.append(err)
        print(f"any_order_launch: HIP error {err} (flags {flags})", file=sys.stderr)


def _launch(f, gx, gy, gz, bx, by, bz, lds, stream, params, extra, count):
    flags = getattr(_local, "flags", 0)
    if flags and f:
        _any_order(f, gx, gy, gz, bx, by, bz, lds, stream, params, extra, flags)
    else:
        _fly.mgpuLaunchKernel(
            f, gx, gy, gz, bx, by, bz, lds, stream, params, extra, count
        )


def _launch_cluster(
    f, cx, cy, cz, gx, gy, gz, bx, by, bz, lds, stream, params, extra, count
):
    flags = getattr(_local, "flags", 0)
    if flags and f:
        _any_order(f, gx, gy, gz, bx, by, bz, lds, stream, params, extra, flags)
    else:
        _fly.mgpuLaunchClusterKernel(
            f, cx, cy, cz, gx, gy, gz, bx, by, bz, lds, stream, params, extra, count
        )


def install() -> None:
    """Route the launches of every FlyDSL launcher compiled from now on
    through ``_launch`` (launchers already run keep their binding)."""
    global _hip, _fly, _callbacks
    with _lock:
        if _callbacks is not None:
            return
        from flydsl.compiler import jit_executor

        hip = _hip_library()
        u32, ptr = ctypes.c_uint32, ctypes.c_void_p
        hip.hipExtModuleLaunchKernel.argtypes = [
            ptr, *(u32,) * 6, ctypes.c_size_t, ptr, ptr, ptr, ptr, ptr, u32,
        ]  # fmt: skip
        hip.hipExtModuleLaunchKernel.restype = ctypes.c_int
        fly = _fly_library()
        i64, i32, size = ctypes.c_int64, ctypes.c_int32, ctypes.c_size_t
        fly.mgpuLaunchKernel.argtypes = [ptr, *(i64,) * 6, i32, ptr, ptr, ptr, size]
        fly.mgpuLaunchKernel.restype = None
        fly.mgpuLaunchClusterKernel.argtypes = [
            ptr,
            *(i64,) * 9,
            i32,
            ptr,
            ptr,
            ptr,
            size,
        ]
        fly.mgpuLaunchClusterKernel.restype = None
        _hip, _fly = hip, fly
        _callbacks = (_LAUNCH(_launch), _CLUSTER_LAUNCH(_launch_cluster))
        addresses = {
            name: ctypes.cast(callback, ctypes.c_void_p)
            for name, callback in zip(
                ("mgpuLaunchKernel", "mgpuLaunchClusterKernel"), _callbacks
            )
        }
        base = jit_executor.ExecutionEngine

        class AnyOrderEngine(base):
            def initialize(self):
                # before initialize() materializes the module, so its calls
                # to the external launch functions bind to these definitions
                for name, address in addresses.items():
                    self.raw_register_runtime(name, address)
                return super().initialize()

        jit_executor.ExecutionEngine = AnyOrderEngine


def installed() -> bool:
    return _callbacks is not None


def mode(site: str) -> str | None:
    """``site``'s entry in ``ATOM_KIMI_MONO_ANY_ORDER``: ``site`` itself,
    ``site-serial`` or ``site-late`` (see envs), or None."""
    from atom.utils import envs

    for entry in (site, f"{site}-serial", f"{site}-late"):
        if entry in envs.ATOM_KIMI_MONO_ANY_ORDER:
            return entry
    return None


def enabled(site: str) -> bool:
    """Whether ``site``'s producer publishes and its consumer waits."""
    return mode(site) is not None


def ready_flags(boundary: str):
    """The ready-flag buffer (``ready_flags_gfx1250``) of ``boundary`` on the
    current device, zeroed once and kept for the process: every launch of the
    boundary's producer and consumer, whatever the width, shares it."""
    import torch

    from atom.models.kimi_k3.mono.kernels.stages import ready_flags_gfx1250

    key = (boundary, torch.cuda.current_device())
    if key not in _flags_buffers:
        _flags_buffers[key] = torch.zeros(
            ready_flags_gfx1250.WORDS,
            dtype=torch.int32,
            device=torch.device("cuda", key[1]),
        )
    return _flags_buffers[key]


@contextlib.contextmanager
def any_order(enabled: bool = True):
    """FlyDSL launches on this thread inside the block go out with the barrier
    bit clear (a launcher compiled before ``install()`` ignores it)."""
    if enabled and _callbacks is None:
        raise RuntimeError("any_order_launch.install() first")
    previous = getattr(_local, "flags", 0)
    _local.flags = ANY_ORDER_LAUNCH if enabled else 0
    try:
        yield
    finally:
        _local.flags = previous
