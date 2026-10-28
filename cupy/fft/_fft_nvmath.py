from __future__ import annotations

from functools import lru_cache
import math
import operator
import warnings

import nvmath.fft as nvmath_fft

import cupy
from cupy.fft import config
from cupy.fft import _nvmath_scaling
from cupy.fft._cache import get_plan_cache


_VALID_NORMS = ('backward', 'ortho', 'forward')
_DIRECTIONS = {
    'forward': nvmath_fft.FFTDirection.FORWARD,
    'inverse': nvmath_fft.FFTDirection.INVERSE,
}
# The options only vary with the FFT type, and nvmath never mutates these
# options, so it's safe to reuse the same options across different FFT plans.
_FFT_OPTIONS = {
    fft_type: nvmath_fft.FFTOptions(
        fft_type=fft_type,
        inplace=False,
        last_axis_parity='even',
        result_layout='optimized',
    )
    for fft_type in ('C2C', 'R2C', 'C2R')
}


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
    # Lists/arrays are valid axes, and tuple cache keys do not distinguish
    # integer elements from hash-equal floats. Cache only plain integer inputs.
    if (axes is None or type(axes) is int
            or type(axes) is tuple
            and all(type(axis) is int for axis in axes)):
        return _normalize_axes_cached(ndim, axes)
    return _normalize_axes_cached.__wrapped__(ndim, axes)


@lru_cache(maxsize=128)
def _normalize_axes_cached(ndim, axes):
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


def _warn_fallback(exc):
    warnings.warn(
        f'nvmath-python rejected this FFT, so it will run on CuPy native '
        f'cuFFT plans instead: {exc!r}',
        RuntimeWarning,
        # 1 is this frame, 2 is _try_use_nvmath, 3 is the public API function.
        stacklevel=4,
    )


def _normalization_divisor(operand, axes, fft_type, policy):
    """Compute normalization over logical transform axes, excluding batches."""
    if policy is None:
        return 1
    size = math.prod(operand.shape[axis] for axis in axes)
    if fft_type == 'C2R':
        # With no requested shape and even parity, m packed values encode
        # a real axis of length 2 * (m - 1).
        packed = operand.shape[axes[-1]]
        size = size // packed * (2 * (packed - 1))
    return math.sqrt(size) if policy == 'sqrtN' else size


def _normalization_epilog(operand, axes, fft_type, policy):
    """Prepare a plan-owned scaling callback only on a plan-cache miss."""
    divisor = _normalization_divisor(operand, axes, fft_type, policy)
    if divisor == 1:
        return None
    dtype = operand.dtype.name
    if fft_type == 'R2C':
        dtype = {'float32': 'complex64', 'float64': 'complex128'}[dtype]
    elif fft_type == 'C2R':
        dtype = {'complex64': 'float32', 'complex128': 'float64'}[dtype]
    return _nvmath_scaling.get_scaling_epilog(dtype, divisor)


def _c_contiguous_element_strides(shape):
    strides = [1] * len(shape)
    for axis in range(len(shape) - 1, 0, -1):
        strides[axis - 1] = strides[axis] * shape[axis]
    return tuple(strides)


def _create_key_from_operand(operand, axes, options, permutation=None):
    """Build the nvmath key from operand metadata.

    When ``permutation`` is given, the key describes the not-yet-materialized
    ``operand.transpose(permutation).copy()`` instead of ``operand`` itself, so
    that the recommended layout only has to be materialized once the key is
    known to be usable.
    """
    if permutation is None:
        shape = tuple(operand.shape)
        element_strides = tuple(
            stride // operand.itemsize for stride in operand.strides)
    else:
        shape = tuple(operand.shape[axis] for axis in permutation)
        element_strides = _c_contiguous_element_strides(shape)
    return nvmath_fft.FFT.create_key_from_metadata(
        shape,
        operand.dtype.name,
        'cuda',
        strides=element_strides,
        axes=axes,
        options=options,
        execution='cuda',
    )


def _invert_permutation(permutation):
    inverse = [None] * len(permutation)
    for new_axis, old_axis in enumerate(permutation):
        inverse[old_axis] = new_axis
    return tuple(inverse)


def _try_use_nvmath(
        operand, requested_shape, axes, norm, *, fft_type, fft_direction):
    """Execute with an NVRTC normalization epilog, or use CuPy's native path.

    Shape changes, multi-GPU execution, callbacks and explicit plans fall back
    silently. Anything nvmath rejects while building or planning also falls
    back, but warns, since which operands, dtypes and layouts nvmath accepts is
    nvmath's to decide and changes between versions. Failures once the plan is
    live propagate instead: see the comments on those blocks.

    The plan key describes the fixed built-in scaling policy. Divisor
    resolution and callback preparation happen only on a plan miss; compiled
    code is reusable across divisors. User callback contexts remain native.
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
        # CuPy user callback contexts remain on native plans.
        return None

    from cupy.cuda import cufft
    if cufft.get_current_plan() is not None:
        return None

    operand_device = operand.data.device_id
    axes = _normalize_axes(operand.ndim, axes)
    if axes is None:
        return None
    if not axes:
        return operand if fft_type == 'C2C' else None
    if operand.size == 0:
        return None

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

    options = _FFT_OPTIONS[fft_type]
    if norm == 'ortho':
        scaling_policy = 'sqrtN'
    elif (norm == 'backward' and fft_direction == 'inverse'
            or norm == 'forward' and fft_direction == 'forward'):
        scaling_policy = 'N'
    else:
        scaling_policy = None
    post_scale_policy = None
    permutation = None
    result_permutation = None

    stream = cupy.cuda.get_current_stream(operand_device)
    try:
        try:
            nvmath_key = _create_key_from_operand(operand, axes, options)
        except nvmath_fft.UnsupportedLayoutError as e:
            permutation = tuple(e.permutation)
            axes = tuple(e.axes)
            result_permutation = _invert_permutation(permutation)
            # Stay out-of-place: the copy's key may also describe a caller's
            # contiguous input, which must never be overwritten.
            nvmath_key = _create_key_from_operand(
                operand, axes, options, permutation)
        if (scaling_policy is not None
                and not _nvmath_scaling.supports_lto_callbacks()):
            post_scale_policy, scaling_policy = scaling_policy, None
    except Exception as e:
        _warn_fallback(e)
        return None

    # NOTE: Keep stateful plans stream-local to avoid concurrent plan/workspace
    # reuse, which is unsafe in CuPy's stream-agnostic cache (cupy/cupy#8079).
    # Metadata and scaling policy identify this built-in callback without
    # resolving code or allocating its scalar on warm hits.
    key = ('nvmath', stream.ptr, nvmath_key)
    if scaling_policy is not None:
        key += (('normalization', scaling_policy),)
    cache = get_plan_cache()
    plan = cache.get(key)
    cache_miss = plan is None

    if permutation is not None:
        # Materialize the layout nvmath recommended, matching the C-contiguous
        # metadata the key was built from. This has to happen before the
        # operand is bound to a plan on either the hit or the miss path.
        operand = operand.transpose(permutation).copy()

    if cache_miss:
        try:
            epilog = (_normalization_epilog(
                operand, axes, fft_type, scaling_policy)
                if scaling_policy is not None else None)
            plan = CachedFFT(
                operand,
                axes=axes,
                options=options,
                execution='cuda',
                stream=stream,
            )
            # DeviceCallable.data is a raw pointer; retain its scalar owner.
            plan._normalization_epilog = epilog
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
        if post_scale_policy is not None:
            divisor = _normalization_divisor(
                operand, axes, fft_type, post_scale_policy)
            if divisor != 1:
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

    if result_permutation is not None:
        out = out.transpose(result_permutation)
    return out
