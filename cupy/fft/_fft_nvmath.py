from __future__ import annotations

import enum
from functools import cache, lru_cache
import math
import operator
from pathlib import Path
import sys
import threading
import warnings

import numpy
from nvmath.bindings import cufft as nvmath_cufft
import nvmath.fft as nvmath_fft

import cupy
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
    """Return axes as non-negative ints, or None to fall back to native.

    Only None, integers, and tuples or lists of integers are supported.
    """
    if axes is None:
        return tuple(range(ndim))
    if type(axes) is not tuple and type(axes) is not list:
        axes = (axes,)
    try:
        return _normalize_axes_cached(ndim, *axes)
    except TypeError:  # Unhashable, such as arrays.
        return None


# Unpacked so typed=True keys on each axis's type (1.0 must not hit 1).
@lru_cache(maxsize=128, typed=True)
def _normalize_axes_cached(ndim, *axes):
    try:
        axes = tuple(map(operator.index, axes))
    except TypeError:
        return None
    if any(not -ndim <= axis < ndim for axis in axes):
        return None
    axes = tuple(axis % ndim for axis in axes)
    return axes if len(set(axes)) == len(axes) else None


@cache
def _cuda_execution(device_id):
    return nvmath_fft.ExecutionCUDA(device_id=device_id)


@lru_cache(maxsize=128)
def _create_key_from_metadata(
        operand_shape, operand_strides, operand_dtype, operand_device_id,
        axes, fft_type, execution_device_id, overwrite):
    """Return ``(key, dtype, permutation, axes, options)`` for an operand.

    Promoted or cross-device operands, and layouts nvmath rejects, are copied
    C-contiguous as ``transpose(permutation)`` to the execution device, with
    the FFT axes innermost; ``permutation`` is None when no transpose is
    needed. C2C runs in place on copies, and on operands that ``overwrite``
    allows to be overwritten.

    Returns None, after warning, if nvmath does not support the operand. Key
    creation depends only on these arguments, so caching the None warns once
    per operand kind. A warning filter that raises skips the cache instead.
    """
    try:
        return _create_key_from_metadata_uncached(
            operand_shape, operand_strides, operand_dtype, operand_device_id,
            axes, fft_type, execution_device_id, overwrite)
    except Exception as e:
        _warn_fallback(
            f'nvmath-python rejected this FFT, so it will run on CuPy native '
            f'cuFFT plans instead: {e!r}. Later calls with the same shape, '
            f'strides and dtype will use the native path without warning.')
        return None


_FFT_DIR = Path(__file__).parent


def _warn_fallback(message):
    # Chunked execution nests calls, so no fixed stacklevel reaches the
    # caller; point at the first frame outside cupy.fft instead.
    frame = sys._getframe(1)
    stacklevel = 2
    while (frame is not None
           and Path(frame.f_code.co_filename).parent == _FFT_DIR):
        frame = frame.f_back
        stacklevel += 1
    warnings.warn(message, RuntimeWarning, stacklevel=stacklevel)


def _create_key_from_metadata_uncached(
        operand_shape, operand_strides, operand_dtype, operand_device_id,
        axes, fft_type, execution_device_id, overwrite):
    execution = _cuda_execution(execution_device_id)
    input_dtype = numpy.dtype(_fft._output_dtype(operand_dtype, fft_type))
    if (input_dtype == operand_dtype
            and operand_device_id == execution_device_id):
        options = _get_fft_options(
            fft_type, inplace=overwrite and fft_type == 'C2C')
        try:
            key = nvmath_fft.FFT.create_key_from_metadata(
                'cupy', operand_shape, input_dtype, execution_device_id,
                strides=operand_strides,
                strides_in_bytes=True, axes=axes, options=options,
                execution=execution)
            return key, input_dtype, None, axes, options
        except nvmath_fft.UnsupportedLayoutError as e:
            permutation = tuple(e.permutation)
            axes = tuple(e.axes)
    else:
        # The operand is copied anyway, so lay the copy out like nvmath's
        # suggested permutation: batch axes, then the FFT axes. The batch
        # axes keep their memory order so the copy stays cheap.
        ndim = len(operand_shape)
        batch = sorted((axis for axis in range(ndim) if axis not in axes),
                       key=lambda axis: -abs(operand_strides[axis]))
        permutation = (*batch, *axes)
        axes = tuple(range(len(batch), ndim))
        if permutation == tuple(range(ndim)):
            permutation = None
    shape = (operand_shape if permutation is None
             else tuple(operand_shape[axis] for axis in permutation))
    # The copy is private, so C2C, the only type nvmath runs in place, can
    # overwrite it.
    options = _get_fft_options(fft_type, inplace=fft_type == 'C2C')
    # Omitted strides describe a C-contiguous layout.
    key = nvmath_fft.FFT.create_key_from_metadata(
        'cupy', shape, input_dtype, execution_device_id,
        axes=axes, options=options,
        execution=execution)
    return key, input_dtype, permutation, axes, options


def _invert_permutation(permutation):
    inverse = [None] * len(permutation)
    for new_axis, old_axis in enumerate(permutation):
        inverse[old_axis] = new_axis
    return tuple(inverse)


def _execute_in_chunks(operand, axes, norm, fft_direction):
    """Run a C2C FFT over more than 3 axes as batched FFTs of up to 3 axes.

    The norm applies per chunk, since chunk divisors multiply to the total.
    Axes are taken innermost first. The first chunk writes a private result,
    which later chunks transform in place.
    """
    axes = sorted(axes, key=lambda axis: abs(operand.strides[axis]))
    out = operand
    for start in range(0, len(axes), 3):
        result = _try_use_nvmath(
            out, None, tuple(sorted(axes[start:start + 3])), norm,
            fft_type='C2C', fft_direction=fft_direction,
            overwrite_operand=out is not operand)
        if result is None:
            if out is operand:
                return None
            # Finish natively, which also applies the norm per axis.
            cufft = cupy.cuda.cufft  # lazy import
            direction = (cufft.CUFFT_FORWARD if fft_direction == 'forward'
                         else cufft.CUFFT_INVERSE)
            return _fft._fft(out, None, tuple(axes[start:]), norm, direction)
        out = result
    return out


def _try_use_nvmath(
        operand, requested_shape, axes, norm, *, fft_type, fft_direction,
        overwrite_operand=False):
    """Execute with nvmath, or return None to use CuPy's native path.

    ``overwrite_operand`` lets C2C run in place on ``operand``; only private
    intermediates may set it.

    Shape changes, misaligned real inputs, multi-GPU execution, callbacks and
    explicit plans fall back silently. Anything nvmath rejects while building
    or planning also falls back, but warns, since which operands, dtypes and
    layouts nvmath accepts is nvmath's to decide and changes between versions.
    Failures once the plan is live propagate instead: see the comments on those
    blocks.

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
    # cuFFT R2C requires alignment to the complex element size (2 * itemsize):
    # 8 bytes for float32 input, 16 bytes for float64 input.
    if (fft_type == 'R2C'
            and operand.dtype == _fft._output_dtype(operand.dtype, fft_type)
            and operand.data.ptr % (2 * operand.itemsize)):
        # Native CuPy copies misaligned real inputs before execution;
        # nvmath does not.
        return None
    if config.devices is not None:
        # CuPy's configured multi-GPU execution remains on the native path.
        return None
    if config.get_current_callback_manager() is not None:
        # cuFFT callbacks are supported only by CuPy's native plans.
        return None

    if cupy.cuda.cufft.get_current_plan() is not None:  # lazy import
        return None

    original_axes = axes
    axes = _normalize_axes(operand.ndim, axes)
    if axes is None:
        return None
    if not axes:
        return operand if fft_type == 'C2C' else None
    if operand.size == 0:
        return None
    if len(axes) > 3:
        # cuFFT plans have rank at most 3, so C2C runs as batched chunks.
        if fft_type != 'C2C':
            return None
        return _execute_in_chunks(operand, axes, norm, fft_direction)

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
    execution_device_id = cupy.cuda.runtime.getDevice()
    result = _create_key_from_metadata(
        operand.shape, operand.strides, operand.dtype,
        operand.data.device_id, axes, fft_type, execution_device_id,
        overwrite_operand)
    if result is None:
        return None
    nvmath_key, dtype, permutation, axes, options = result
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
        # Materialize the copy's layout in one pass with any promotion,
        # matching the C-contiguous metadata the key was built from. This
        # has to happen before the operand is bound to a plan on either the
        # hit or the miss path.
        operand = operand.transpose(permutation).astype(
            dtype, order='C', copy=True)
    elif (operand.dtype != dtype
          or operand.data.device_id != execution_device_id):
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
            _warn_fallback(
                f'nvmath-python rejected this FFT, so it will run on CuPy '
                f'native cuFFT plans instead: {e!r}')
            return None
    else:
        plan.reset_operand_unchecked(operand, axes=axes, stream=stream)

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
