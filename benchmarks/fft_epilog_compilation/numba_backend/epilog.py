"""Prepare one plan's immutable normalization data and reusable epilog."""

from __future__ import annotations

import math

import cupy
import numpy

from .compiler import compile_code


def compile_normalization_epilog(plan, operand, axes, norm, fft_direction):
    """Resolve normalization on a plan miss and bind immutable device data."""
    scaled = (
        norm == "ortho"
        or norm == "backward"
        and fft_direction == "inverse"
        or norm == "forward"
        and fft_direction == "forward"
    )
    if scaled:
        shape = (
            plan.get_output_layout()[0]
            if plan.fft_abstract_type == "C2R"
            else operand.shape
        )
        divisor = math.prod(shape[axis] for axis in axes)
        if norm == "ortho":
            divisor = math.sqrt(divisor)
    else:
        divisor = 1

    plan._cupy_fft_epilog_divisor = divisor
    plan._cupy_fft_epilog_ltoir = None
    plan._cupy_fft_epilog_data = None
    if divisor == 1:
        return None

    real_dtype = numpy.finfo(plan.result_data_type).dtype
    typed_divisor = real_dtype.type(divisor)
    if not numpy.isfinite(typed_divisor) or typed_divisor <= 0:
        raise ValueError(
            "The normalization divisor must be finite and positive"
        )
    host_data = numpy.asarray([typed_divisor], dtype=real_dtype)
    with cupy.cuda.Device(plan.device_id):
        ltoir = compile_code(plan.result_data_type)
        # Cold, one-time upload. Blocking completes the transfer before the
        # temporary host scalar is released; warm calls never update it.
        data = cupy.asarray(host_data, blocking=True)
    plan._cupy_fft_epilog_ltoir = ltoir
    # DeviceCallable.data is only an address. Keep its owner on the plan;
    # the benchmark synchronizes outstanding execution before freeing plans.
    plan._cupy_fft_epilog_data = data
    return {"ltoir": ltoir, "data": data.data.ptr}
