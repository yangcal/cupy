from __future__ import annotations

from collections.abc import Sequence
import enum
from functools import cache, lru_cache
import math
from pathlib import Path
import threading
import warnings

import numpy
from nvmath.bindings import cufft as nvmath_cufft
import nvmath.fft as nvmath_fft

import cupy
from cupy.cuda import cufft
from cupy.fft import _fft, config
from cupy.fft._cache import get_plan_cache


_VALID_NORMS = ('backward', 'ortho', 'forward')
_DIRECTIONS = {
    'forward': nvmath_fft.FFTDirection.FORWARD,
    'inverse': nvmath_fft.FFTDirection.INVERSE,
}
_CALLBACK_TYPES = {
    numpy.dtype(numpy.float32): 0,
    numpy.dtype(numpy.float64): 1,
    numpy.dtype(numpy.complex64): 2,
    numpy.dtype(numpy.complex128): 3,
}
_compile_lock = threading.Lock()


@cache
def _get_fft_options(fft_type, inplace):
    # nvmath never mutates these options, so it's safe to reuse the same
    # options across different FFT plans.
    return nvmath_fft.FFTOptions(
        fft_type=fft_type,
        inplace=inplace,
        last_axis_parity='even',
        result_layout='optimized',
    )


class _Scaling(enum.IntEnum):
    NONE = 0
    N = 1
    SQRT_N = 2

    @classmethod
    def from_norm(cls, norm, fft_direction):
        if norm == 'ortho':
            return cls.SQRT_N
        # 'forward' scales forward transforms, 'backward' inverse ones.
        if (norm == 'forward') == (fft_direction == 'forward'):
            return cls.N
        return cls.NONE

    def get_normalization_divisor(self, fft_type, shape, axes):
        """Normalize over logical transform axes, excluding batches."""
        if self is _Scaling.NONE:
            return 1
        size = 1
        for axis in axes:
            size *= shape[axis]
        if fft_type == 'C2R':
            # With no requested shape and even parity, m packed values
            # encode a real axis of length 2 * (m - 1).
            packed = shape[axes[-1]]
            size = size // packed * (2 * (packed - 1))
        return math.sqrt(size) if self is _Scaling.SQRT_N else size


@cache
def _supports_lto_callbacks():
    """Standard LTO callbacks require cuFFT 11.3 (CUDA 12.6U2)."""
    return nvmath_cufft.get_version() >= 11300


@cache
def _get_scaling_ltoir(dtype, arch):
    """Return the store callback's LTO-IR for an output dtype and arch."""
    from cupy.cuda import compiler

    source = Path(__file__).with_name('_nvmath_scaling.cu').read_text()
    options = (
        '--std=c++17',
        f'-DCUPY_FFT_CALLBACK_TYPE={_CALLBACK_TYPES[dtype]}',
    )
    with _compile_lock:
        return compiler._compile_module_with_cache(
            source, options, arch=arch, to_ltoir=True)


def _get_scaling_epilog(dtype, divisor):
    """Return a callback and the immutable scalar in output precision it
    divides by, for the current device.

    The callback only holds a raw pointer to the scalar, so the caller must
    keep the scalar alive for as long as the callback is in use.

    ``divisor`` is not validated, since all callers guarantee a valid value.
    """
    from cupy.cuda import compiler

    ltoir = _get_scaling_ltoir(dtype, compiler._get_arch())
    scalar = cupy.full(1, divisor, dtype=numpy.finfo(dtype).dtype)
    epilog = nvmath_fft.DeviceCallable(ltoir=ltoir, data=scalar.data.ptr)
    return epilog, scalar


class CachedFFT(nvmath_fft.FFT):
    # How PlanCache.show_info() labels these entries, alongside the Plan1d
    # and PlanNd labels it uses for CuPy's own plans.
    _cupy_fft_cache_name = 'nvmath FFT'

    def _cupy_fft_cache_cleanup(self):
        """Release resources when PlanCache relinquishes ownership."""
        self.free()

    @property
    def gpus(self):
        """Identify this as a single-device plan to PlanCache."""
        return None

    @property
    def work_area(self):
        """Exclude released nvmath workspaces from memory accounting."""
        return None


def _normalize_axes(ndim, axes):
    if isinstance(axes, Sequence):
        axes = tuple(axes)
    return _normalize_axes_cached(ndim, axes)


@lru_cache(maxsize=128)
def _normalize_axes_cached(ndim, axes):
    if axes is None:
        return tuple(range(ndim)) if ndim <= 3 else None
    if isinstance(axes, int):
        return (axes % ndim,) if -ndim <= axes < ndim else None
    if len(axes) > 3 or any(not -ndim <= axis < ndim for axis in axes):
        return None
    axes = tuple(axis % ndim for axis in axes)
    return axes if len(set(axes)) == len(axes) else None


def _warn_fallback(exc):
    warnings.warn(
        f'nvmath-python rejected this FFT, so it will run on CuPy native '
        f'cuFFT plans instead: {exc!r}',
        RuntimeWarning,
        # 1 is this frame, 2 is _try_use_nvmath, 3 is the public API function.
        stacklevel=4,
    )


@cache
def _cuda_execution(device_id):
    return nvmath_fft.ExecutionCUDA(device_id=device_id)


def _create_key_from_operand(operand, axes, fft_type, device_id):
    """Build the key for the operand nvmath executes on ``device_id``.

    Returns ``(key, dtype, permutation, axes, options)``. ``dtype`` is
    promoted as on CuPy's native path. An operand that needs promotion or is
    not on ``device_id`` is replaced by a private C-contiguous copy there. The
    caller's layout, or that copy, is used as is when nvmath accepts it, with
    ``permutation`` None. Otherwise the key describes a private C-contiguous
    copy of ``transpose(permutation)`` in the promoted dtype, using the
    permutation nvmath reports. C2C transforms private copies in place.
    """
    shape = operand.shape
    strides = operand.strides
    execution = _cuda_execution(device_id)
    input_dtype = numpy.dtype(_fft._output_dtype(operand.dtype, fft_type))
    copied = (input_dtype != operand.dtype
              or operand.data.device_id != device_id)
    if copied:
        # Omitted strides describe a C-contiguous layout.
        strides = None
    options = _get_fft_options(
        fft_type, inplace=copied and fft_type == 'C2C')
    try:
        key = nvmath_fft.FFT.create_key_from_metadata(
            shape, input_dtype, 'cuda', strides=strides,
            strides_in_bytes=True, axes=axes, options=options,
            execution=execution)
        return key, input_dtype, None, axes, options
    except nvmath_fft.UnsupportedLayoutError as e:
        permutation = tuple(e.permutation)
        shape = tuple(shape[axis] for axis in permutation)
        axes = tuple(e.axes)
    # The rejected layout is always copied into a private operand, so C2C,
    # the only type nvmath runs in place, can overwrite it.
    options = _get_fft_options(fft_type, inplace=fft_type == 'C2C')
    # Omitted strides describe a C-contiguous layout.
    key = nvmath_fft.FFT.create_key_from_metadata(
        shape, input_dtype, 'cuda', axes=axes, options=options,
        execution=execution)
    return key, input_dtype, permutation, axes, options


def _invert_permutation(permutation):
    inverse = [None] * len(permutation)
    for new_axis, old_axis in enumerate(permutation):
        inverse[old_axis] = new_axis
    return tuple(inverse)


def _try_use_nvmath(
        operand, requested_shape, axes, norm, *, fft_type, fft_direction):
    """Execute with nvmath, or return None to use CuPy's native path.

    Shape changes, multi-GPU execution, callbacks and explicit plans fall back
    silently. Anything nvmath rejects while building or planning also falls
    back, but warns, since which operands, dtypes and layouts nvmath accepts is
    nvmath's to decide and changes between versions. Failures once the plan is
    live propagate instead: see the comments on those blocks.

    Fused plans are keyed by their divisor, so plans are shared only when they
    scale identically. Callback preparation happens only on a plan miss;
    compiled code is reusable across divisors. User callback contexts remain
    native.
    """
    if not config.use_nvmath:
        return None
    if not isinstance(operand, cupy.ndarray):
        return None
    if requested_shape is not None:
        # Keep CuPy's existing padding and truncation behavior for now.
        return None
    if config.devices is not None:
        # CuPy's configured multi-GPU execution remains on the native path.
        return None
    if config.get_current_callback_manager() is not None:
        # cuFFT callbacks are supported only by CuPy's native plans.
        return None

    if cufft.get_current_plan() is not None:
        return None

    original_axes = axes
    axes = _normalize_axes(operand.ndim, axes)
    if axes is None:
        return None
    if not axes:
        return operand if fft_type == 'C2C' else None
    if operand.size == 0:
        return None

    if fft_type == 'C2R':
        caster = _fft._compat_caster(operand, original_axes)

    if norm is None:
        norm = 'backward'
    elif norm not in _VALID_NORMS:
        raise ValueError(
            f'Invalid norm value {norm}, should be "backward", "ortho", '
            'or "forward".')

    try:
        direction = _DIRECTIONS[fft_direction]
    except KeyError:
        raise ValueError(
            f'Unsupported FFT direction: {fft_direction!r}') from None

    scaling = _Scaling.from_norm(norm, fft_direction)
    divisor = scaling.get_normalization_divisor(fft_type, operand.shape, axes)

    # Execute on the current device, as CuPy's other functions do.
    device_id = cupy.cuda.runtime.getDevice()
    try:
        nvmath_key, dtype, permutation, axes, options = (
            _create_key_from_operand(operand, axes, fft_type, device_id))
    except Exception as e:
        _warn_fallback(e)
        return None
    # Without LTO callbacks, scale the output after execution instead.
    fused = divisor != 1 and _supports_lto_callbacks()

    stream = cupy.cuda.get_current_stream()
    # NOTE: Keep stateful plans stream-local to avoid concurrent plan/workspace
    # reuse, which is unsafe in CuPy's stream-agnostic cache (cupy/cupy#8079).
    # Metadata and the divisor identify this built-in callback without
    # resolving code or allocating its scalar on warm hits. Unfused scaling
    # shares the unscaled plan.
    key = ('nvmath', stream.ptr, nvmath_key)
    if fused:
        key += (divisor,)
    cache = get_plan_cache()
    plan = cache.get(key)
    cache_miss = plan is None

    if permutation is not None:
        # Materialize the layout nvmath recommended, matching the C-contiguous
        # metadata the key was built from. This has to happen before the
        # operand is bound to a plan on either the hit or the miss path.
        operand = operand.transpose(permutation).astype(
            dtype, order='C', copy=True)
    elif operand.dtype != dtype or operand.data.device_id != device_id:
        operand = operand.astype(dtype, order='C')

    if cache_miss:
        try:
            epilog = scalar = None
            if fused:
                result_dtype = operand.dtype
                if fft_type == 'R2C':
                    result_dtype = numpy.dtype(
                        f'c{result_dtype.itemsize * 2}')
                elif fft_type == 'C2R':
                    result_dtype = numpy.dtype(
                        f'f{result_dtype.itemsize // 2}')
                epilog, scalar = _get_scaling_epilog(result_dtype, divisor)
            plan = CachedFFT(
                operand,
                axes=axes,
                options=options,
                execution='cuda',
                stream=stream,
            )
            # DeviceCallable.data is a raw pointer to the scalar.
            plan._cupy_normalization_scalar = scalar
            plan.plan(stream=stream, epilog=epilog)
        except Exception as e:
            if plan is not None:
                plan.free()
            _warn_fallback(e)
            return None
    else:
        plan.reset_operand_unchecked(operand, stream=stream)

    try:
        try:
            out = plan.execute(
                direction=direction,
                stream=stream,
                release_workspace=True,
            )
        finally:
            plan.release_operand()
        if not fused and divisor != 1:
            out /= divisor
    except Exception:
        if cache_miss:
            plan.free()
        raise

    if cache_miss:
        if cache.is_enabled:
            try:
                cache[key] = plan
            except Exception:
                plan.free()
                raise
        else:
            # A disabled cache drops the insertion silently, so the plan has
            # no owner and has to be released here.
            plan.free()

    if permutation is not None:
        out = out.transpose(_invert_permutation(permutation))
    if fft_type == 'C2R':
        out = caster(out)
    return out
