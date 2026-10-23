"""NVRTC store callbacks for nvmath FFT normalization."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import threading

import numpy
import nvmath.fft as nvmath_fft
from nvmath.bindings import cufft

import cupy


_OUTPUT_TYPES = {
    'float32': (0, numpy.float32),
    'float64': (1, numpy.float64),
    'complex64': (2, numpy.float32),
    'complex128': (3, numpy.float64),
}
_compile_lock = threading.Lock()


@lru_cache(maxsize=128)
def _compile_code(dtype_name, arch, offset_uint64):
    """Cache IR by output type, GPU target and cuFFT offset ABI."""
    from cupy.cuda import compiler

    source = Path(__file__).with_suffix('.cu').read_text()
    options = (
        '--std=c++17',
        f'-DCUPY_FFT_CALLBACK_TYPE={_OUTPUT_TYPES[dtype_name][0]}',
        f'-DCUPY_FFT_CALLBACK_OFFSET_UINT64={int(offset_uint64)}',
    )
    return compiler._compile_module_with_cache(
        source, options, arch=arch, to_ltoir=True)


def get_scaling_epilog(dtype_name, divisor):
    """Return a callback owning an immutable scalar in output precision."""
    real_type = _OUTPUT_TYPES[dtype_name][1]
    divisor = real_type(divisor)
    if not numpy.isfinite(divisor) or divisor <= 0:
        raise ValueError(
            'The normalization divisor must be finite and positive')
    if divisor == 1:
        return None

    from cupy.cuda import compiler

    arch = compiler._get_arch()
    offset_uint64 = cufft.get_version() >= 11300
    with _compile_lock:
        ltoir = _compile_code(dtype_name, arch, offset_uint64)
    # Complete the upload before the temporary host buffer is released.
    scalar = cupy.asarray(numpy.asarray([divisor]), blocking=True)
    epilog = nvmath_fft.DeviceCallable(ltoir=ltoir, data=scalar.data.ptr)
    epilog._cupy_scalar = scalar
    return epilog
