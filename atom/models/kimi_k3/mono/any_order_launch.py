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
bit. FlyDSL launches through ``mgpuLaunchKernel``
(``libfly_jit_runtime``), a ``hipModuleLaunchKernel`` with no flags.
``install()`` registers a replacement ``mgpuLaunchKernel`` in every FlyDSL
``ExecutionEngine`` built after it: inside ``any_order()`` it launches with
``hipExtAnyOrderLaunch``, elsewhere as before. Every launch of such an engine
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

_local = threading.local()
_lock = threading.Lock()
_hip = None
_callback = None
failures: list[int] = []


def _hip_library():
    """The HIP runtime this process already runs (torch's), by its path."""
    with open("/proc/self/maps") as maps:
        for line in maps:
            if "libamdhip64.so" in line:
                return ctypes.CDLL(line.split()[-1])
    return ctypes.CDLL("libamdhip64.so")


def _launch(f, gx, gy, gz, bx, by, bz, lds, stream, params, extra, _count):
    if not f:
        return
    flags = getattr(_local, "flags", 0)
    if flags:
        # hipExtModuleLaunchKernel takes the global size in threads
        err = _hip.hipExtModuleLaunchKernel(
            f, gx * bx, gy * by, gz * bz, bx, by, bz, lds, stream, params, extra,
            None, None, flags,
        )  # fmt: skip
    else:
        err = _hip.hipModuleLaunchKernel(
            f, gx, gy, gz, bx, by, bz, lds, stream, params, extra
        )
    if err:
        failures.append(err)
        print(f"any_order_launch: HIP error {err} (flags {flags})", file=sys.stderr)


def install() -> None:
    """Route the launches of every FlyDSL launcher compiled from now on
    through ``_launch`` (launchers already run keep their binding)."""
    global _hip, _callback
    with _lock:
        if _callback is not None:
            return
        from flydsl.compiler import jit_executor

        hip = _hip_library()
        u32, ptr = ctypes.c_uint32, ctypes.c_void_p
        hip.hipModuleLaunchKernel.argtypes = [ptr, *(u32,) * 7, ptr, ptr, ptr]
        hip.hipModuleLaunchKernel.restype = ctypes.c_int
        hip.hipExtModuleLaunchKernel.argtypes = [
            ptr, *(u32,) * 6, ctypes.c_size_t, ptr, ptr, ptr, ptr, ptr, u32,
        ]  # fmt: skip
        hip.hipExtModuleLaunchKernel.restype = ctypes.c_int
        _hip = hip
        _callback = _LAUNCH(_launch)
        address = ctypes.cast(_callback, ctypes.c_void_p)
        base = jit_executor.ExecutionEngine

        class AnyOrderEngine(base):
            def initialize(self):
                # before initialize() materializes the module, so its call
                # to the external mgpuLaunchKernel binds to this definition
                self.raw_register_runtime("mgpuLaunchKernel", address)
                return super().initialize()

        jit_executor.ExecutionEngine = AnyOrderEngine


def installed() -> bool:
    return _callback is not None


@contextlib.contextmanager
def any_order(enabled: bool = True):
    """FlyDSL launches on this thread inside the block go out with the barrier
    bit clear (a launcher compiled before ``install()`` ignores it)."""
    if enabled and _callback is None:
        raise RuntimeError("any_order_launch.install() first")
    previous = getattr(_local, "flags", 0)
    _local.flags = ANY_ORDER_LAUNCH if enabled else 0
    try:
        yield
    finally:
        _local.flags = previous
