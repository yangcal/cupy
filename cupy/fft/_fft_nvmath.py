from __future__ import annotations

import math
import operator

import nvmath.fft as nvmath_fft
import numpy

import cupy
from cupy.fft import config
from cupy.fft._cache import get_plan_cache


_VALID_NORMS = ('backward', 'ortho', 'forward')
_SUPPORTED_DTYPES = {
    'C2C': (numpy.dtype('complex64'), numpy.dtype('complex128')),
    'R2C': (numpy.dtype('float32'), numpy.dtype('float64')),
    'C2R': (numpy.dtype('complex64'), numpy.dtype('complex128')),
}
_DIRECTIONS = {
    'forward': nvmath_fft.FFTDirection.FORWARD,
    'inverse': nvmath_fft.FFTDirection.INVERSE,
}


class CachedFFT(nvmath_fft.FFT):
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
    if axes is None:
        axes = tuple(range(ndim))
    else:
        try:
            axis = operator.index(axes)
        except TypeError:
            axes = tuple(axes)
        else:
            axes = (axis,)

    if not axes:
        return axes
    if len(axes) > 3:
        return None
    if any(axis >= ndim or axis < -ndim for axis in axes):
        return None

    axes = tuple(axis % ndim for axis in axes)
    return axes if len(set(axes)) == len(axes) else None


def _is_supported_dtype(dtype, fft_type):
    return dtype in _SUPPORTED_DTYPES[fft_type]


def _scale_result(out, operand, axes, norm, fft_type, fft_direction):
    shape = operand.shape if fft_type == 'R2C' else out.shape
    size = math.prod(shape[axis] for axis in axes)
    if norm == 'backward' and fft_direction == 'inverse':
        out /= size
    elif norm == 'ortho':
        out /= math.sqrt(size)
    elif norm == 'forward' and fft_direction == 'forward':
        out /= size


def _try_use_nvmath(
        operand, requested_shape, axes, norm, *, fft_type, fft_direction):
    """Execute with nvmath, or return None to use CuPy's native path.

    Fallbacks cover shape changes, multi-GPU execution, callbacks or explicit
    plans, and unsupported operands, axes, dtypes, or layouts.
    """
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

    from cupy.cuda import cufft
    if cufft.get_current_plan() is not None:
        return None

    axes = _normalize_axes(operand.ndim, axes)
    if axes is None:
        return None
    if not axes:
        return operand if fft_type == 'C2C' else None
    if operand.size == 0 or not _is_supported_dtype(operand.dtype, fft_type):
        return None

    if norm is None:
        norm = 'backward'
    elif norm not in _VALID_NORMS:
        raise ValueError(
            f'Invalid norm value {norm}, should be "backward", "ortho", '
            'or "forward".')

    options = nvmath_fft.FFTOptions(
        fft_type=fft_type,
        inplace=False,
        last_axis_parity='even',
        result_layout='natural',
    )
    element_strides = tuple(
        stride // operand.itemsize for stride in operand.strides)

    try:
        nvmath_key = nvmath_fft.FFT.create_key_from_metadata(
            tuple(operand.shape),
            operand.dtype.name,
            'cuda',
            strides=element_strides,
            axes=axes,
            options=options,
            execution='cuda',
        )
    # TODO: Support more cases with layout changes.
    except (nvmath_fft.UnsupportedLayoutError, ValueError):
        return None

    stream = cupy.cuda.get_current_stream()
    # NOTE: Keep stateful plans stream-local to avoid concurrent plan/workspace
    # reuse, which is unsafe in CuPy's stream-agnostic cache (cupy/cupy#8079).
    key = ('nvmath', stream.ptr, nvmath_key)
    cache = get_plan_cache()
    plan = cache.get(key)
    cache_miss = plan is None

    if cache_miss:
        try:
            plan = CachedFFT(
                operand,
                axes=axes,
                options=options,
                execution='cuda',
                stream=stream,
            )
            plan.plan(stream=stream)
        except Exception:
            if plan is not None:
                plan.free()
            raise
    else:
        plan.reset_operand_unchecked(operand, stream=stream)

    try:
        direction = _DIRECTIONS[fft_direction]
    except KeyError:
        raise ValueError(
            f'Unsupported FFT direction: {fft_direction!r}') from None

    try:
        try:
            out = plan.execute(
                direction=direction,
                stream=stream,
                release_workspace=True,
            )
            _scale_result(
                out, operand, axes, norm, fft_type, fft_direction)
        finally:
            plan.release_operand()
    except Exception:
        if cache_miss:
            plan.free()
        raise

    if cache_miss:
        # Either zero limit disables PlanCache, even for zero-weight plans.
        cache_enabled = (
            cache.get_size() != 0 and cache.get_memsize() != 0)
        if cache_enabled:
            try:
                cache[key] = plan
            except Exception:
                plan.free()
                raise
        else:
            plan.free()

    return out
