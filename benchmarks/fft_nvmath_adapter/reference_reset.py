"""Benchmark-only reset snapshot from nvmath.

Source revision: eafc59f5f7f335957c34f8aa5a1a0d449694efd0.

The executable body is FFT.reset_operand_unchecked before prepared C2R copying.
Keep the old reset here so Original and Fused-only controls remain meaningful
when nvmath integrates that optimization into its existing reset API.
"""

# ruff: noqa: E501 -- preserve the reference method body verbatim.
from __future__ import annotations

from nvmath.internal import tensor_wrapper, utils


def reset_operand_unchecked(self, operand, *, stream=None):
    """Reference reset; operand compatibility is guaranteed by the caller."""
    with self._device_scope:
        # Case 1: Same execution and memory space, non-C2R transform
        # The user guarantees that the operand is in the same memory space as
        # the original operand so we can directly update the operand
        # reference without copying data.
        # The TensorHolder wrapper is always alive (release_operand only clears
        # .tensor), so we can unconditionally swap the inner tensor.
        if self.execution_space == self.memory_space and self.fft_abstract_type != "C2R":
            self.operand.tensor = operand
            self.operand_backup = None
            self._operand_released = False
            self._update_layout_and_plan_traits_if_needed()
            return

        # Cases 2 and 3 require data copying, so we need the stream
        stream_holder = self._stream_resolver(stream)
        # and also require the wrapped operand
        operand_wrapped = tensor_wrapper.wrap_operand(operand)

        # Case 2: C2R transform with same execution and memory space
        # Data must be copied to prevent cuFFT from overwriting the user's input buffer.
        # This is a corner case that stems from cuFFT behavior.
        if self.execution_space == self.memory_space:
            needs_reallocation = self._operand_released or self.operand.shape != operand_wrapped.shape
            if needs_reallocation:
                self.operand = utils.create_empty_tensor(
                    operand_wrapped.__class__,
                    operand_wrapped.shape,
                    operand_wrapped.dtype,
                    self.device_id,
                    stream_holder,
                    verify_strides=True,
                    strides=operand_wrapped.strides,
                )
                if self.execution_space == "cuda":
                    self.c2r_buffer_stream = stream_holder.obj  # type: ignore[union-attr]

            self.operand.copy_(operand_wrapped, stream_holder=stream_holder)
            self.operand_backup = None
            self._operand_released = False
            self._c2r_operand_stale = False
            self._update_layout_and_plan_traits_if_needed()
            return

        # Case 3: Cross-space scenario (execution_space != memory_space)
        # Example: CPU operand with CUDA execution, or CUDA operand with CPU execution.
        # Data must be copied between memory spaces.
        needs_reallocation = self._operand_released or self.operand.shape != operand_wrapped.shape
        if needs_reallocation:
            if self.execution_space == "cuda":
                to_device = self.device_id
            else:
                to_device = "cpu"
            self.operand = operand_wrapped.to(to_device, stream_holder)
        else:
            self.operand.copy_(operand_wrapped, stream_holder=stream_holder)
        self.operand_backup.tensor = operand_wrapped.tensor
        self._operand_released = False
        self._c2r_operand_stale = False
        self._update_layout_and_plan_traits_if_needed()
