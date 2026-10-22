# Warm FFT comparison

tldr:
Fusing normalization into the FFT store removes substantial repeated adapter work and delivers consistent warm-call improvements across the tested transform families. Compilation latency, caching policy and generated-code efficiency remain implementation questions before production adoption.

Compare public CuPy FFT calls with cached and continuously bound nvmath execution.
Ajay adds common-axis shortcuts, fused normalization and prepared C2R input copying.

## C2R improvement

CPU time in µs from the focused four-row comparison: 24 balanced blocks,
500 samples per row. Fused-only already has the axis and scaling optimizations;
Ajay also prepares the input-copy path. Operands below are complex input shapes.

| Request | Operand | Fused-only | Ajay | Native |
| --- | --- | ---: | ---: | ---: |
| `irfft(a)` | `(17,)` complex64 | 39.71 | 22.32 | 28.30 |
| `irfft(a)` | `(524289,)` complex64 | 45.18 | 28.15 | 31.86 |
| `irfft2(a)` | `(8,9)` complex64 | 44.50 | 26.60 | 43.97 |
| `irfftn(a)` | `(4,8,9)` complex64 | 47.46 | 28.30 | 46.14 |
| `irfftn(a, norm='ortho')` | `(64,64,33)` complex128 | 48.08 | 28.25 | 46.34 |

Across nine C2R requests, prepared copying saves 16.9–19.9 µs over Fused-only.
All nine beat Native on CPU and CUDA-event time. The full sweep below is a
separate 30-block measurement; remaining substantial CPU gaps are on unscaled
requests, especially 1D and batched single-axis transforms.

## Implementations

| Row | What it measures | Definition |
| --- | --- | --- |
| Original | Existing CuPy nvmath adapter, with compiled strides and cached dtype names. | [`_try_use_nvmath`](../../cupy/fft/_fft_nvmath.py#L222) |
| Ajay | Common-axis shortcuts, fused normalization and nvmath's prepared C2R copying. | [`_try_use_nvmath_ajay`](../../cupy/fft/_fft_nvmath.py#L357), nvmath `FFT.reset_operand_unchecked()` |
| Fused-only | Ajay's adapter with the original nvmath reset; focused C2R comparison only. | [`reference_reset.py`](reference_reset.py), selected by `--compare-c2r-reset` in [`main.py`](main.py) |
| Native | Public `cupy.fft` using CuPy's cached cuFFT plans. | [`cupy/fft/_fft.py`](../../cupy/fft/_fft.py) |
| Minimal | Key creation, dictionary lookup, reset, execution and operand release. Forward C2C only. | [`execute_cached`](minimal_lifecycle.py#L18) |
| Stateful | Repeated `FFT.execute()` on a continuously bound operand. Forward C2C only. | `stateful.execute` in [`main.py`](main.py) |

All nvmath rows use the same execute/workspace candidate and stride/dtype
optimizations. Fused normalization uses the documented `compile_epilog` API;
C2R preparation lives inside nvmath's existing `FFT.reset_operand_unchecked()` and
`internal/tensor_ifc_cupy.py` (`CupyTensor._prepare_c2r_copy_in_current_context`).
Each C2R call still allocates fresh scratch and copies the current input.
[`main.py`](main.py) selects the reset method on the existing `CachedFFT` class
before each row, outside timing: Original and Fused-only use
[`reference_reset.py`](reference_reset.py), while Ajay uses nvmath's public reset.
Rows with identical keys share a plan; the CuPy adapter needs no C2R-specific
subclass or routing change.
The results below were measured before this public-API integration; that
refactor still needs a correctness and timing rerun.
Requests are in [`cases.py`](cases.py).

## Replicate

From this CuPy checkout, with a CUDA 13 development toolkit available, install
nvmath's [`ajay/fft-c2r-preparation-inline`](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/tree/ajay/fft-c2r-preparation-inline)
candidate. That branch contains only the C2R preparation changes. The reported
measurements also use compiled strides and cached dtype names from separate MRs;
apply those prerequisites before building to reproduce the comparison.

```sh
NVMATH_SOURCE=/path/to/nvmath-candidate
python -m pip install 'setuptools>=77.0.3' wheel 'Cython>=3.2,<3.3' 'numpy<2.5'
python -m pip install --no-build-isolation -e "$NVMATH_SOURCE[sysctk13,numba]"
CUPY_USE_CUDA_PYTHON=1 PYTHONPATH="$NVMATH_SOURCE" \
  python -m pip install --no-build-isolation -e .

python -P benchmarks/fft_nvmath_adapter/main.py --suite basic \
  --repeats 20 --warmups 5 --json smoke.json
python -P benchmarks/fft_nvmath_adapter/main.py --case irfft \
  --compare-c2r-reset --blocks 24 --json c2r-comparison.json
python -P benchmarks/fft_nvmath_adapter/main.py --json comparison.json
```

The nvmath candidate requires internal repository access. General CuPy build
instructions are in [Installation](../../docs/source/install.rst).
Use `--list-cases` or `--case SUBSTRING` to select requests.
JSON records raw CPU/GPU samples, preparation times, environment and source hashes.

<details>
<summary>Separate metadata prerequisites used in the measurements</summary>

These prerequisites are outside the C2R change. With the earlier experiment
commits available locally, the CuPy metadata patch can be applied before building:

```sh
git -C "$NVMATH_SOURCE" diff eafc59f5f 84d59bc -- \
  nvmath/internal/_strides_helpers.pyx nvmath/internal/_strides_helpers.pyi \
  nvmath/internal/tensor_ifc.py nvmath/internal/tensor_ifc_cupy.py \
  pyproject.toml |
  git -C "$NVMATH_SOURCE" apply
```

</details>

## Full results

2026-10-22 · NVIDIA B200 · Intel Xeon Platinum 8570 · `umbriel-b200-074`.
Python 3.12.3 · NumPy 2.4.6 · pinned CPU 30.

CPU times in µs: medians of block medians from 30 balanced blocks, 500 samples
and 20 warmups per row, measured with `cupyx.profiler.benchmark`.
Planning/validation are outside timing; workspace is retained. All 87 cases
passed correctness checks and Ajay improved CPU time over Original in every block.
CUDA-event intervals include host submission gaps; four unscaled cases show
consistent 0.22–0.74 µs event slowdowns versus Original despite faster CPU calls.

Scaled norms/directions use separate callback plans. Callback compilation remains
a cold-path cost: first use took about 1.1 s, outside these warm measurements.

![Warm CPU times and paired Ajay-minus-Native gaps, grouped by operand family](warm-cpu-comparison.png)

Case IDs match [`cases.py`](cases.py); table and chart use the same group order.

### C2R input preservation and normalization

| ID | Request | Operand | Native | Original | Ajay | Minimal | Stateful |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 62 | `irfft(a)` | `(17,)` complex64 C | 26.900 | 53.249 | 21.999 | — | — |
| 63 | `irfft(a)` | `(524289,)` complex64 C | 31.521 | 59.508 | 27.814 | — | — |
| 66 | `irfft2(a)` | `(8, 9)` complex64 C | 42.252 | 60.245 | 25.877 | — | — |
| 74 | `irfft2(a, norm='forward')` | `(8, 9)` complex64 C | 32.962 | 46.350 | 26.093 | — | — |
| 75 | `irfft2(a, norm='ortho')` | `(512, 513)` complex64 C | 43.936 | 61.554 | 26.663 | — | — |
| 67 | `irfft2(a)` | `(1024, 513)` complex64 C | 43.026 | 60.890 | 26.487 | — | — |
| 70 | `irfftn(a)` | `(4, 8, 9)` complex64 C | 44.293 | 62.918 | 27.237 | — | — |
| 71 | `irfftn(a)` | `(128, 64, 65)` complex64 C | 44.456 | 63.604 | 27.031 | — | — |
| 79 | `irfftn(a, norm='ortho')` | `(64, 64, 33)` complex128 C | 44.611 | 63.600 | 27.142 | — | — |

### C2C normalization — contiguous operands

| ID | Request | Operand | Native | Original | Ajay | Minimal | Stateful |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 18 | `fft(a, axis=-1, norm='forward')` | `(1024,)` complex64 C | 13.737 | 27.559 | 14.275 | — | — |
| 19 | `fft(a, axis=-1, norm='ortho')` | `(1024,)` complex64 C | 13.688 | 27.295 | 13.856 | — | — |
| 20 | `ifft(a, axis=-1, norm=None)` | `(1024,)` complex64 C | 13.702 | 27.149 | 13.739 | — | — |
| 22 | `ifft(a, axis=-1, norm='ortho')` | `(1024,)` complex64 C | 13.663 | 27.167 | 13.799 | — | — |
| 4 | `fft(a, axis=-1)` | `(1024,)` complex64 C | 8.069 | 16.737 | 13.813 | 11.022 | 7.406 |
| 17 | `fft(a, axis=-1, norm='backward')` | `(1024,)` complex64 C | 8.081 | 16.948 | 14.067 | — | — |
| 21 | `ifft(a, axis=-1, norm='forward')` | `(1024,)` complex64 C | 8.060 | 16.674 | 13.810 | — | — |
| 23 | `ifft(a, axis=-1)` | `(1048576,)` complex64 C | 15.455 | 28.909 | 16.003 | — | — |
| 6 | `fft(a, axis=-1)` | `(1048576,)` complex64 C | 9.835 | 18.727 | 15.904 | 12.922 | 9.153 |
| 25 | `fftn(a, axes=(0, 1), norm='forward')` | `(32, 32)` complex64 C | 26.233 | 32.444 | 17.696 | — | — |
| 26 | `fftn(a, axes=(0, 1), norm='ortho')` | `(32, 32)` complex64 C | 26.508 | 32.559 | 17.340 | — | — |
| 27 | `ifftn(a, axes=(0, 1), norm=None)` | `(32, 32)` complex64 C | 26.589 | 32.452 | 17.440 | — | — |
| 29 | `ifftn(a, axes=(0, 1), norm='ortho')` | `(32, 32)` complex64 C | 26.426 | 32.786 | 17.483 | — | — |
| 9 | `fftn(a, axes=(0, 1))` | `(32, 32)` complex64 C | 15.596 | 20.131 | 17.454 | 13.384 | 9.195 |
| 24 | `fftn(a, axes=(0, 1), norm='backward')` | `(32, 32)` complex64 C | 15.662 | 19.952 | 17.234 | — | — |
| 28 | `ifftn(a, axes=(0, 1), norm='forward')` | `(32, 32)` complex64 C | 15.628 | 20.131 | 17.674 | — | — |
| 30 | `ifftn(a, axes=(0, 1))` | `(1024, 1024)` complex64 C | 26.440 | 32.744 | 17.782 | — | — |
| 11 | `fftn(a, axes=(0, 1))` | `(1024, 1024)` complex64 C | 15.828 | 20.328 | 17.718 | 13.385 | 9.093 |
| 43 | `fftn(a, axes=(1, 0))` | `(1024, 1024)` complex64 C | 15.651 | 20.448 | 19.795 | — | — |
| 51 | `fftn(a, axes=(0, -1))` | `(1024, 1024)` complex64 C | 15.773 | 20.487 | 19.834 | — | — |
| 32 | `fftn(a, axes=(0, 1, 2), norm='forward')` | `(16, 16, 16)` complex64 C | 29.298 | 34.858 | 20.004 | — | — |
| 33 | `fftn(a, axes=(0, 1, 2), norm='ortho')` | `(16, 16, 16)` complex64 C | 29.315 | 35.509 | 20.091 | — | — |
| 34 | `ifftn(a, axes=(0, 1, 2), norm=None)` | `(16, 16, 16)` complex64 C | 29.385 | 34.828 | 19.820 | — | — |
| 36 | `ifftn(a, axes=(0, 1, 2), norm='ortho')` | `(16, 16, 16)` complex64 C | 29.117 | 34.991 | 19.675 | — | — |
| 14 | `fftn(a, axes=(0, 1, 2))` | `(16, 16, 16)` complex64 C | 18.081 | 22.324 | 19.538 | 15.302 | 11.063 |
| 31 | `fftn(a, axes=(0, 1, 2), norm='backward')` | `(16, 16, 16)` complex64 C | 17.993 | 22.457 | 19.761 | — | — |
| 35 | `ifftn(a, axes=(0, 1, 2), norm='forward')` | `(16, 16, 16)` complex64 C | 18.326 | 22.508 | 19.801 | — | — |
| 37 | `ifftn(a, axes=(0, 1, 2))` | `(128, 64, 128)` complex64 C | 29.018 | 34.664 | 19.647 | — | — |
| 16 | `fftn(a, axes=(0, 1, 2))` | `(128, 64, 128)` complex64 C | 17.985 | 22.381 | 19.558 | 15.242 | 10.975 |
| 77 | `ifft(a, norm=None)` | `(65536,)` complex128 C | 15.591 | 29.149 | 15.931 | — | — |

### Batched C2C normalization

| ID | Request | Operand | Native | Original | Ajay | Minimal | Stateful |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 54 | `ifft2(a, norm=None)` | `(2, 8, 16)` complex64 C | 27.426 | 32.708 | 19.884 | — | — |
| 56 | `ifft2(a, norm='ortho')` | `(2, 8, 16)` complex64 C | 27.548 | 32.905 | 19.906 | — | — |
| 40 | `fft2(a)` | `(2, 8, 16)` complex64 C | 16.341 | 20.414 | 19.780 | — | — |
| 55 | `ifft2(a, norm='forward')` | `(2, 8, 16)` complex64 C | 16.396 | 20.605 | 20.046 | — | — |
| 57 | `ifft2(a, norm=None)` | `(4, 256, 512)` complex64 C | 27.809 | 33.110 | 20.124 | — | — |
| 59 | `ifft2(a, norm='ortho')` | `(4, 256, 512)` complex64 C | 28.247 | 33.492 | 20.163 | — | — |
| 41 | `fft2(a)` | `(4, 256, 512)` complex64 C | 16.506 | 20.953 | 20.209 | — | — |
| 58 | `ifft2(a, norm='forward')` | `(4, 256, 512)` complex64 C | 16.530 | 20.797 | 20.198 | — | — |

### C2C normalization — Fortran and strided operands

| ID | Request | Operand | Native | Original | Ajay | Minimal | Stateful |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 84 | `ifft(a)` | `(8, 16)` complex64 F | 22.584 | 28.975 | 13.940 | — | — |
| 80 | `fft(a)` | `(8, 16)` complex64 F | 14.814 | 16.630 | 13.747 | — | — |
| 85 | `ifft(a)` | `(16, 65536)` complex64 F | 24.293 | 31.071 | 16.075 | — | — |
| 81 | `fft(a)` | `(16, 65536)` complex64 F | 16.448 | 18.686 | 16.114 | — | — |
| 86 | `ifft(a)` | `(64,)` complex64 strided | 19.709 | 27.111 | 13.742 | — | — |
| 82 | `fft(a)` | `(64,)` complex64 strided | 13.963 | 16.463 | 13.400 | — | — |
| 87 | `ifft(a)` | `(1048576,)` complex64 strided | 21.480 | 28.969 | 15.775 | — | — |
| 83 | `fft(a)` | `(1048576,)` complex64 strided | 15.761 | 18.469 | 15.714 | — | — |

### R2C normalization

| ID | Request | Operand | Native | Original | Ajay | Minimal | Stateful |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 72 | `rfft(a, norm='forward')` | `(33,)` float32 C | 13.763 | 27.157 | 13.996 | — | — |
| 73 | `rfft(a, norm='ortho')` | `(1048575,)` float32 C | 23.982 | 37.766 | 24.849 | — | — |
| 78 | `rfftn(a, norm='forward')` | `(64, 64, 64)` float64 C | 29.531 | 34.272 | 18.863 | — | — |

### Unscaled C2C — full transforms

| ID | Request | Operand | Native | Original | Ajay | Minimal | Stateful |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `fft(a, axis=-1)` | `(2,)` complex64 C | 8.001 | 16.423 | 13.550 | 10.949 | 7.401 |
| 2 | `fft(a, axis=-1)` | `(8,)` complex64 C | 8.006 | 16.667 | 13.587 | 10.983 | 7.371 |
| 3 | `fft(a, axis=-1)` | `(64,)` complex64 C | 8.046 | 16.706 | 13.802 | 11.051 | 7.427 |
| 5 | `fft(a, axis=-1)` | `(65536,)` complex64 C | 9.877 | 18.669 | 15.818 | 12.886 | 9.182 |
| 7 | `fftn(a, axes=(0, 1))` | `(2, 2)` complex64 C | 15.651 | 19.999 | 17.356 | 13.232 | 9.168 |
| 8 | `fftn(a, axes=(0, 1))` | `(8, 8)` complex64 C | 15.690 | 19.927 | 17.312 | 13.313 | 9.244 |
| 42 | `fftn(a, axes=(1, 0))` | `(8, 16)` complex64 C | 15.743 | 20.293 | 19.694 | — | — |
| 50 | `fftn(a, axes=(0, -1))` | `(8, 16)` complex64 C | 15.680 | 20.163 | 19.554 | — | — |
| 10 | `fftn(a, axes=(0, 1))` | `(128, 128)` complex64 C | 15.808 | 20.374 | 17.643 | 13.485 | 9.320 |
| 12 | `fftn(a, axes=(0, 1, 2))` | `(2, 2, 2)` complex64 C | 17.970 | 22.136 | 19.371 | 15.249 | 10.977 |
| 13 | `fftn(a, axes=(0, 1, 2))` | `(4, 4, 4)` complex64 C | 17.886 | 22.069 | 19.313 | 15.131 | 10.989 |
| 15 | `fftn(a, axes=(0, 1, 2))` | `(64, 64, 64)` complex64 C | 18.161 | 22.570 | 19.769 | 15.333 | 11.070 |
| 76 | `fft(a, norm=None)` | `(1048576,)` complex128 C | 10.019 | 18.533 | 15.698 | — | — |

### Unscaled C2C — batched and nonadjacent axes

| ID | Request | Operand | Native | Original | Ajay | Minimal | Stateful |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 38 | `fftn(a, axes=(1,))` | `(3, 16)` complex64 C | 9.346 | 17.138 | 16.536 | — | — |
| 39 | `fftn(a, axes=(1,))` | `(16, 65536)` complex64 C | 11.170 | 19.621 | 18.836 | — | — |
| 52 | `fftn(a, axes=(2, 3))` | `(2, 3, 8, 16)` complex64 C | 16.557 | 20.922 | 20.300 | — | — |
| 53 | `fftn(a, axes=(2, 3))` | `(2, 4, 256, 512)` complex64 C | 16.468 | 20.983 | 20.227 | — | — |
| 44 | `fftn(a, axes=(0, 2))` | `(4, 3, 8)` complex64 C | 34.990 | 20.951 | 20.229 | — | — |
| 46 | `fftn(a, axes=[0, 2])` | `(4, 3, 8)` complex64 C | 34.573 | 20.895 | 20.186 | — | — |
| 48 | `fftn(a, axes=(np.int64(0), np.int64(2)))` | `(4, 3, 8)` complex64 C | 36.439 | 21.721 | 20.459 | — | — |
| 45 | `fftn(a, axes=(0, 2))` | `(128, 4, 1024)` complex64 C | 36.732 | 21.398 | 20.558 | — | — |
| 47 | `fftn(a, axes=[0, 2])` | `(128, 4, 1024)` complex64 C | 35.924 | 21.347 | 20.640 | — | — |
| 49 | `fftn(a, axes=(np.int64(0), np.int64(2)))` | `(128, 4, 1024)` complex64 C | 38.083 | 22.253 | 20.934 | — | — |

### Unscaled R2C transforms

| ID | Request | Operand | Native | Original | Ajay | Minimal | Stateful |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 60 | `rfft(a)` | `(32,)` float32 C | 8.017 | 16.549 | 13.598 | — | — |
| 61 | `rfft(a)` | `(1048576,)` float32 C | 11.869 | 20.912 | 18.222 | — | — |
| 64 | `rfft2(a)` | `(8, 16)` float32 C | 16.471 | 20.599 | 17.842 | — | — |
| 65 | `rfft2(a)` | `(1024, 1024)` float32 C | 16.603 | 20.609 | 17.981 | — | — |
| 68 | `rfftn(a)` | `(4, 8, 16)` float32 C | 17.285 | 22.056 | 18.514 | — | — |
| 69 | `rfftn(a)` | `(128, 64, 128)` float32 C | 17.416 | 22.281 | 18.801 | — | — |

