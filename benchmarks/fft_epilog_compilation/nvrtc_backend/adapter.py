"""Baseline request routing and FFT lifecycle for the NVRTC row."""

from __future__ import annotations

import nvmath.fft as nvmath_fft

import cupy
from cupy.fft import config
from cupy.fft._cache import get_plan_cache
from cupy.fft._fft_nvmath import (
    CachedFFT,
    _DIRECTIONS,
    _FFT_OPTIONS,
    _VALID_NORMS,
    _create_key_from_operand,
    _invert_permutation,
    _normalize_axes,
    _warn_fallback,
)

from .epilog import compile_normalization_epilog


BACKEND = "nvrtc"


def try_use_nvmath(
    operand, requested_shape, axes, norm, *, fft_type, fft_direction
):
    """Use the baseline lifecycle with an immutable per-plan scaling divisor.

    Generic axes, dtype/stride metadata, operand resets and workspace release
    remain those of the production adapter. Only the epilog compiler and its
    cache namespace differ between the two benchmark siblings.
    """
    if not config.use_nvmath:
        return None
    if not isinstance(operand, cupy.ndarray):
        return None
    if requested_shape is not None:
        return None
    if config.devices is not None:
        return None
    if config.get_current_callback_manager() is not None:
        return None

    from cupy.cuda import cufft

    if cufft.get_current_plan() is not None:
        return None

    axes = _normalize_axes(operand.ndim, axes)
    if axes is None:
        return None
    if not axes:
        return operand if fft_type == "C2C" else None
    if operand.size == 0:
        return None

    if norm is None:
        norm = "backward"
    elif norm not in _VALID_NORMS:
        raise ValueError(
            f'Invalid norm value {norm}, should be "backward", "ortho", '
            'or "forward".'
        )
    elif type(norm) is not str:
        # Cache the validated meaning, including unhashable string subclasses.
        norm = _VALID_NORMS[_VALID_NORMS.index(norm)]

    try:
        direction = _DIRECTIONS[fft_direction]
    except KeyError:
        raise ValueError(
            f"Unsupported FFT direction: {fft_direction!r}"
        ) from None

    options = _FFT_OPTIONS[fft_type]
    permutation = None
    result_permutation = None

    try:
        nvmath_key = _create_key_from_operand(operand, axes, options)
    except nvmath_fft.UnsupportedLayoutError as e:
        permutation = tuple(e.permutation)
        axes = tuple(e.axes)
        result_permutation = _invert_permutation(permutation)
        # Preserve the baseline's out-of-place layout-recovery policy.
        nvmath_key = _create_key_from_operand(
            operand, axes, options, permutation
        )
    except Exception as e:
        _warn_fallback(e)
        return None

    stream = cupy.cuda.get_current_stream()
    # Plans stay stream-local, as in the production adapter. The per-plan
    # divisor fixes normalization, so scaled norms/directions stay apart.
    key = ("nvmath_epilog", BACKEND, stream.ptr, nvmath_key)
    if (
        norm == "ortho"
        or norm == "backward"
        and fft_direction == "inverse"
        or norm == "forward"
        and fft_direction == "forward"
    ):
        key += (norm, fft_direction)
    cache = get_plan_cache()
    plan = cache.get(key)
    cache_miss = plan is None

    if permutation is not None:
        operand = operand.transpose(permutation).copy()

    if cache_miss:
        try:
            plan = CachedFFT(
                operand,
                axes=axes,
                options=options,
                execution="cuda",
                stream=stream,
            )
            epilog = compile_normalization_epilog(
                plan, operand, axes, norm, fft_direction
            )
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
            plan.free()

    if result_permutation is not None:
        out = out.transpose(result_permutation)
    return out
