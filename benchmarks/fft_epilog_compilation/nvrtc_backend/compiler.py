"""Compile the CUDA source through CuPy's existing cached LTO path."""

from __future__ import annotations

from pathlib import Path

import compilation_cache


BACKEND = "nvrtc"
SOURCE_FILE = Path(__file__).with_name("scaling.cu")
_TYPE_IDS = {"float32": 0, "float64": 1, "complex64": 2, "complex128": 3}


def source_for_dtype(dtype_name, cufft_version):
    """Read CUDA source and select one callback signature via options."""
    options = (
        "--std=c++17",
        "--relocatable-device-code=true",
        f"-DCUPY_FFT_CALLBACK_TYPE={_TYPE_IDS[dtype_name]}",
        f"-DCUPY_FFT_CALLBACK_OFFSET_UINT64={int(cufft_version >= 11300)}",
    )
    return SOURCE_FILE.read_text(), options


def _compile(dtype_name, arch, cufft_version):
    """Use the same cached NVRTC LTO path as CuPy's native JIT callbacks."""
    from cupy.cuda import compiler

    source, options = source_for_dtype(dtype_name, cufft_version)
    return compiler._compile_module_with_cache(
        source, options, arch=arch, to_ltoir=True
    )


def compile_code(dtype_name):
    """Include CUDA source edits in the shared persistent cache identity."""
    return compilation_cache.get_code(
        BACKEND, dtype_name, _compile, __file__, SOURCE_FILE
    )
