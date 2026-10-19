"""The saved minimal cached lifecycle, with its actual metadata providers.

The function below is byte-identical to the saved reference. The benchmark
calls it directly; axes and options are prepared before each timed block.
"""

from __future__ import annotations

# Keep the saved function's source verbatim, including its long lines.
# ruff: noqa: E501

import cupy as cp
import nvmath.fft as nf
from nvmath.internal._strides_helpers import element_strides
from nvmath.internal.tensor_ifc import _numpy_dtype_name


def execute_cached(operand, axes, options, cache, *, release_workspace=False):
    """Derive the key, look up a stream-local plan, bind, execute, then release."""
    # Fetch metadata on every hit, as a caller must do for a fresh operand.
    key = nf.FFT.create_key_from_metadata(
        tuple(operand.shape), _numpy_dtype_name(operand.dtype), "cuda",
        strides=element_strides(operand.strides, operand.itemsize),
        axes=axes, options=options, execution="cuda",
    )
    stream = cp.cuda.get_current_stream()
    plan = cache[(stream.ptr, key)]  # A miss raises KeyError; planning is outside timing.
    plan.reset_operand_unchecked(operand, stream=stream)
    try:
        return plan.execute(direction=nf.FFTDirection.FORWARD, stream=stream,
                            release_workspace=release_workspace)
    finally:
        plan.release_operand()
