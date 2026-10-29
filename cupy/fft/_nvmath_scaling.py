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


@lru_cache(maxsize=1)
def supports_lto_callbacks():
    """Standard LTO callbacks require cuFFT 11.3 (CUDA 12.6U2)."""
    return cufft.get_version() >= 11300


@lru_cache(maxsize=None)
def get_scaling_ltoir(dtype_name, device_id):
    """Return the store callback's LTO-IR for an output dtype and device."""
    from cupy.cuda import compiler

    with cupy.cuda.Device(device_id):
        arch = compiler._get_arch()
    source = Path(__file__).with_suffix('.cu').read_text()
    options = (
        '--std=c++17',
        f'-DCUPY_FFT_CALLBACK_TYPE={_OUTPUT_TYPES[dtype_name][0]}',
    )
    with _compile_lock:
        return compiler._compile_module_with_cache(
            source, options, arch=arch, to_ltoir=True)


def get_scaling_epilog(dtype_name, divisor, device_id):
    """Return a callback owning an immutable scalar in output precision.

    The scalar is allocated on ``device_id``, which need not be the current
    device.
    """
    real_type = _OUTPUT_TYPES[dtype_name][1]
    divisor = real_type(divisor)
    if not numpy.isfinite(divisor) or divisor <= 0:
        raise ValueError(
            'The normalization divisor must be finite and positive')
    if divisor == 1:
        return None

    ltoir = get_scaling_ltoir(dtype_name, device_id)
    with cupy.cuda.Device(device_id):
        scalar = cupy.full(1, divisor, dtype=real_type)
    epilog = nvmath_fft.DeviceCallable(ltoir=ltoir, data=scalar.data.ptr)
    epilog._cupy_scalar = scalar
    return epilog
