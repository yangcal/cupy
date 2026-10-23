"""Compile reusable Python scaling through nvmath's documented API."""

from __future__ import annotations

import nvmath.fft as nvmath_fft
import numpy

import compilation_cache


BACKEND = "numba"


def rescale(data_out, offset, data, user_info, unused):
    """Scale by real-precision user data supplied by the owning FFT plan."""
    data_out[offset] = data / user_info[0]


def _compile(dtype_name, arch, cufft_version):
    """Let nvmath resolve the cuFFT callback symbol and compile Python."""
    real_dtype = numpy.finfo(dtype_name).dtype
    return nvmath_fft.compile_epilog(
        rescale, dtype_name, real_dtype.name, compute_capability=arch
    )


def compile_code(dtype_name):
    """Compile or reuse divisor-independent code through the common cache."""
    return compilation_cache.get_code(BACKEND, dtype_name, _compile, __file__)
