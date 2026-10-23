"""The same bounded process cache and CuPy disk storage for both compilers.

Only LTO IR bytes are retained, never plans, operands or device allocations.
This experiment adds disk reuse to the documented Numba compilation flow;
compile_epilog itself does not provide that cache.
"""

from __future__ import annotations

from collections import OrderedDict
from functools import cache
import hashlib
from importlib import metadata
import json
from pathlib import Path
import threading
from time import perf_counter

import numpy as np


_CODE = OrderedDict()
_LOCK = threading.RLock()
_LIMIT = 128
_LAST_LOOKUP = None


@cache
def _versions(backend):
    packages = ["cupy", "nvmath-python", "numpy"]
    if backend == "numba":
        packages += [
            "numba",
            "numba-cuda",
            "llvmlite",
            "numba-cuda-mlir",
            "nvidia-nvvm",
            "nvidia-cuda-nvcc-cu12",
        ]
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _source_and_versions(backend, source_files):
    # Include external CUDA source too. A callback edit must invalidate both
    # the process cache and disk artifacts. This runs only on FFT plan misses.
    digest = hashlib.sha256()
    for path in (*source_files, __file__):
        contents = Path(path).read_bytes()
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest(), _versions(backend)


def get_code(backend, dtype_name, compile_uncached, *source_files):
    """Cache a runtime-divisor callback for the current device.

    The callable receives (output dtype, arch, cuFFT version). Divisors are
    plan data and never enter this code-cache key. NVRTC still uses CuPy's
    ordinary compiler cache underneath this common storage layer.
    """
    global _LAST_LOOKUP
    from cupy.cuda import Device, compiler, runtime
    from nvmath.bindings import cufft

    started = perf_counter()
    dtype_name = np.dtype(dtype_name).name
    if dtype_name not in ("float32", "float64", "complex64", "complex128"):
        raise ValueError(f"Unsupported callback dtype: {dtype_name}")
    arch = Device().compute_capability
    cufft_version = cufft.get_version()
    source_hash, versions = _source_and_versions(backend, source_files)
    identity = {
        "backend": backend,
        "dtype": dtype_name,
        "arch": arch,
        "cufft": cufft_version,
        "cuda_runtime": runtime.runtimeGetVersion(),
        "source": source_hash,
        "versions": versions,
    }
    if backend == "nvrtc":
        identity["nvrtc"] = compiler._get_nvrtc_version()
    name = (
        "fft_epilog_"
        + backend
        + "_"
        + hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        + ".ltoir"
    )
    with _LOCK:
        code = _CODE.get(name)
        level = "memory"
        if code is None:
            code = compiler._kernel_cache_backend.load(name)
            level = "disk"
            if code is None:
                code = compile_uncached(dtype_name, arch, cufft_version)
                if not isinstance(code, bytes) or not code:
                    raise TypeError(
                        "The compiler must return nonempty LTO IR bytes"
                    )
                compiler._kernel_cache_backend.save(name, code, source_hash)
                level = "compile"
            _CODE[name] = code
            if len(_CODE) > _LIMIT:
                _CODE.popitem(last=False)
        _CODE.move_to_end(name)
        _LAST_LOOKUP = {
            "level": level,
            "milliseconds": (perf_counter() - started) * 1e3,
            "identity": identity,
            "ltoir_bytes": len(code),
            "ltoir_sha256": hashlib.sha256(code).hexdigest(),
        }
    return code


def clear_memory_cache():
    """Leave persistent artifacts intact, as if this process had restarted."""
    global _LAST_LOOKUP
    with _LOCK:
        _CODE.clear()
        _LAST_LOOKUP = None


def cache_info():
    with _LOCK:
        return {
            "entries": len(_CODE),
            "limit": _LIMIT,
            "last_lookup": _LAST_LOOKUP,
        }
