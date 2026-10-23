"""Reusable NVRTC store callbacks for nvmath FFT normalization."""

from functools import lru_cache
from pathlib import Path
import threading
from weakref import WeakValueDictionary

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
_bindings = threading.local()
_compile_lock = threading.Lock()


@lru_cache(maxsize=128)
def _compile_code(dtype_name, arch, offset_uint64):
    """Retain IR in memory and use CuPy's persistent compilation cache."""
    from cupy.cuda import compiler

    source = Path(__file__).with_suffix('.cu').read_text()
    options = (
        '--std=c++17',
        '--relocatable-device-code=true',
        f'-DCUPY_FFT_CALLBACK_TYPE={_OUTPUT_TYPES[dtype_name][0]}',
        f'-DCUPY_FFT_CALLBACK_OFFSET_UINT64={int(offset_uint64)}',
    )
    return compiler._compile_module_with_cache(
        source, options, arch=arch, to_ltoir=True)


def get_scaling_epilog(dtype_name, divisor, stream_ptr):
    """Return an immutable callback binding, shared by matching cached plans.

    Called after a semantic plan-cache miss. Weak interning lets matching plans
    share an immutable scalar without retaining GPU allocations after those
    plans are released. Bindings follow PlanCache's thread, device and stream
    separation. Compiled code is reusable across divisors.

    The caller must retain the returned descriptor for the plan's lifetime,
    including completion of executions using its scalar.
    """
    try:
        real_type = _OUTPUT_TYPES[dtype_name][1]
    except KeyError:
        raise ValueError(
            f'Unsupported callback dtype: {dtype_name!r}') from None
    typed_divisor = real_type(divisor)
    if not numpy.isfinite(typed_divisor) or typed_divisor <= 0:
        raise ValueError(
            'The normalization divisor must be finite and positive')
    if typed_divisor == 1:
        return None

    device_id = cupy.cuda.runtime.getDevice()
    key = device_id, stream_ptr, dtype_name, typed_divisor
    bindings = getattr(_bindings, 'cache', None)
    if bindings is None:
        bindings = _bindings.cache = WeakValueDictionary()
    epilog = bindings.get(key)
    if epilog is not None:
        return epilog

    arch = cupy.cuda.Device(device_id).compute_capability
    offset_uint64 = cufft.get_version() >= 11300
    # Serialize cold lookups so simultaneous threads do not compile the same
    # signature twice. Plan hits never enter this helper.
    with _compile_lock:
        ltoir = _compile_code(dtype_name, arch, offset_uint64)
    host_data = numpy.asarray([typed_divisor], dtype=real_type)
    # Finish the one-time upload before releasing the temporary host buffer.
    scalar = cupy.asarray(host_data, blocking=True)
    epilog = nvmath_fft.DeviceCallable(ltoir=ltoir, data=scalar.data.ptr)
    epilog._cupy_scalar = scalar
    epilog._cupy_divisor = typed_divisor
    bindings[key] = epilog
    return epilog
