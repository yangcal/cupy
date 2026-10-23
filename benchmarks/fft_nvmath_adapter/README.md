# Warm FFT adapter experiments

Compare three adapter changes separately and together: common-axis/no-op-scaling
shortcuts, fused normalization, and prepared C2R input copying. This branch
records experiments; production changes will be submitted separately.

## Implementations

| Row | Implementation | Definition |
| --- | --- | --- |
| Original | Baseline adapter, with compiled strides and cached dtype names. | [`_try_use_nvmath`](../../cupy/fft/_fft_nvmath.py) |
| Axes | Common-axis/no-op-scaling shortcuts and shared request gates; original C2R reset. | [`_try_use_nvmath_ajay_unfused`](../../cupy/fft/_fft_nvmath.py), [`reference_reset.py`](reference_reset.py) |
| Fusion | Axes + NVRTC fused normalization; original C2R reset. | [`_try_use_nvmath_ajay`](../../cupy/fft/_fft_nvmath.py), [`reference_reset.py`](reference_reset.py) |
| C2R | Axes + prepared C2R copying; separate CuPy scaling. | [`_try_use_nvmath_ajay_unfused`](../../cupy/fft/_fft_nvmath.py), nvmath `FFT.reset_operand_unchecked()` |
| Ajay | Axes + fusion + prepared C2R copying. | [`_try_use_nvmath_ajay`](../../cupy/fft/_fft_nvmath.py) |
| Native | Public CuPy FFT with cached native cuFFT plans. | [`cupy/fft/_fft.py`](../../cupy/fft/_fft.py) |
| Minimal | Metadata key, dictionary lookup, reset, execute and operand release. | [`minimal_lifecycle.py`](minimal_lifecycle.py) |
| Stateful | Repeated execution on a continuously bound nvmath FFT. | [`main.py`](main.py) |

All nvmath rows use the same execute/workspace candidate and metadata helpers.
Minimal and Stateful cover the 16 marked default-forward C2C cases. Row selection
and reset-method changes happen outside timing.

Fusion uses CuPy's NVRTC compiler/cache with an immutable runtime divisor and a
semantic plan key. Callback preparation happens only on plan misses; no Numba
dependency is required. Code and scalar ownership live in
[`_nvmath_scaling.py`](../../cupy/fft/_nvmath_scaling.py); the store callback is
in [`_nvmath_scaling.cu`](../../cupy/fft/_nvmath_scaling.cu). Compilation and
cuFFT linking remain cold costs.

C2R preparation lives in nvmath's existing `reset_operand_unchecked()` API. Each
call still allocates scratch and copies the current input; no additional reset
API or reset-specific subclass is required in CuPy.

## Replicate

Use a CUDA 13 development toolkit and the nvmath
[`ajay/fft-c2r-preparation-inline`](https://gitlab-master.nvidia.com/cuda-hpc-libraries/python/nvmath-python/-/tree/ajay/fft-c2r-preparation-inline)
candidate with the compiled-stride and cached-dtype prerequisites from their
separate MRs. From this CuPy checkout:

```sh
NVMATH_SOURCE=/path/to/nvmath-candidate
python -m pip install 'setuptools>=77' wheel 'Cython>=3.2,<3.3' 'numpy>=2.0,<2.7'
python -m pip install --no-build-isolation -e "$NVMATH_SOURCE[sysctk13]"
CUPY_USE_CUDA_PYTHON=1 python -m pip install --no-build-isolation -e .

python -P benchmarks/fft_nvmath_adapter/main.py --suite basic \
  --repeats 20 --warmups 5 --json smoke.json
python -P benchmarks/fft_nvmath_adapter/main.py --json comparison.json
```

The full run sweeps 87 requests and writes raw CPU/GPU samples and a grouped
`comparison.md`. Use `--list-cases` or `--case SUBSTRING` to select requests.
`--compare-fusion` and `--case irfft --compare-c2r-reset` retain focused comparisons.

<details>
<summary>Metadata prerequisites used in this recorded run</summary>

These changes are separate from the C2R MR. With the experiment commits available
in the nvmath clone, apply before building:

```sh
git -C "$NVMATH_SOURCE" diff eafc59f5f 84d59bc -- \
  nvmath/internal/_strides_helpers.pyx nvmath/internal/_strides_helpers.pyi \
  nvmath/internal/tensor_ifc.py nvmath/internal/tensor_ifc_cupy.py \
  pyproject.toml | git -C "$NVMATH_SOURCE" apply
```

</details>

## Results

2026-10-23 · NVIDIA B200 · Intel Xeon Platinum 8570 · `umbriel-b200-018`.
Python 3.12.3 · NumPy 2.4.6 · CUDA 13.4 · pinned CPU 28.

CPU times in µs: medians of block medians from 24 balanced blocks, 500 samples
and 20 warmups per row. Plans and compilation are warmed; workspace is retained.
All 87 cases passed correctness, input-preservation and backend-route checks.

Ajay beats Original in all 87 CPU cases and Native in 37/87; 53/87 are faster
than Native or no more than 2 µs slower. All nine C2R cases beat Native.
These are unweighted coverage counts. Axes has four small batched-inverse
regressions (0.38–0.54 µs); fusion adds about 0.22 µs on unscaled controls.

### Small forward transforms

| Operand | Stateful | Minimal | Original | Axes | Fusion | C2R | Ajay | Native |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `(2,)` | 7.227 | 10.859 | 16.387 | 13.589 | 13.915 | 13.576 | 13.813 | 7.753 |
| `(2, 2)` | 9.154 | 13.157 | 20.035 | 17.568 | 17.845 | 17.613 | 17.847 | 15.384 |
| `(2, 2, 2)` | 10.829 | 15.142 | 22.236 | 19.670 | 19.798 | 19.672 | 19.784 | 18.070 |

### Representative C2R transforms

| Request / complex input | Original | Axes | Fusion | C2R | Ajay | Native |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `irfft (17,)` | 53.762 | 52.406 | 39.890 | 34.584 | 22.401 | 27.534 |
| `irfftn (4, 8, 9)` | 64.718 | 62.790 | 47.490 | 42.802 | 27.893 | 45.335 |
| `irfftn (64, 64, 33) complex128, ortho` | 65.008 | 63.062 | 47.519 | 42.930 | 28.101 | 45.779 |

### Full comparison

![CPU times and paired Ajay-minus-Native gaps, grouped by operand](warm-cpu-comparison.png)

Table and chart share the operand ordering below; IDs remain those in
[`cases.py`](cases.py). Positive gaps in the chart mean Ajay is slower.

### C2R input preservation and normalization

| ID | Request | Operand | Stateful | Minimal | Original | Axes | Fusion | C2R | Ajay | Native |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 62 | `irfft(a)` | `(17,) complex64 C` | — | — | 53.762 | 52.406 | 39.890 | 34.584 | 22.401 | 27.534 |
| 66 | `irfft2(a)` | `(8, 9) complex64 C` | — | — | 61.633 | 59.824 | 44.351 | 41.299 | 26.723 | 43.288 |
| 74 | `irfft2(a, norm='forward')` | `(8, 9) complex64 C` | — | — | 47.187 | 44.563 | 44.508 | 26.593 | 26.649 | 32.995 |
| 70 | `irfftn(a)` | `(4, 8, 9) complex64 C` | — | — | 64.718 | 62.790 | 47.490 | 42.802 | 27.893 | 45.335 |
| 79 | `irfftn(a, norm='ortho')` | `(64, 64, 33) complex128 C` | — | — | 65.008 | 63.062 | 47.519 | 42.930 | 28.101 | 45.779 |
| 75 | `irfft2(a, norm='ortho')` | `(512, 513) complex64 C` | — | — | 63.080 | 61.204 | 45.486 | 42.262 | 27.425 | 44.367 |
| 63 | `irfft(a)` | `(524289,) complex64 C` | — | — | 59.922 | 58.809 | 45.929 | 40.813 | 28.409 | 31.631 |
| 67 | `irfft2(a)` | `(1024, 513) complex64 C` | — | — | 62.357 | 60.694 | 45.078 | 42.124 | 27.298 | 43.863 |
| 71 | `irfftn(a)` | `(128, 64, 65) complex64 C` | — | — | 65.253 | 63.287 | 47.805 | 43.011 | 28.332 | 45.655 |

### C2C normalization — contiguous operands

| ID | Request | Operand | Stateful | Minimal | Original | Axes | Fusion | C2R | Ajay | Native |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | `fftn(a, axes=(0, 1), norm='forward')` | `(32, 32) complex64 C` | — | — | 32.929 | 31.626 | 18.014 | 31.636 | 18.047 | 27.174 |
| 26 | `fftn(a, axes=(0, 1), norm='ortho')` | `(32, 32) complex64 C` | — | — | 33.060 | 31.614 | 17.845 | 31.541 | 17.840 | 27.078 |
| 27 | `ifftn(a, axes=(0, 1), norm=None)` | `(32, 32) complex64 C` | — | — | 33.310 | 31.856 | 17.956 | 31.890 | 18.027 | 27.540 |
| 29 | `ifftn(a, axes=(0, 1), norm='ortho')` | `(32, 32) complex64 C` | — | — | 33.201 | 31.793 | 17.765 | 31.711 | 17.821 | 27.368 |
| 9 | `fftn(a, axes=(0, 1))` | `(32, 32) complex64 C` | 9.057 | 13.207 | 20.176 | 17.625 | 17.856 | 17.628 | 17.932 | 15.454 |
| 24 | `fftn(a, axes=(0, 1), norm='backward')` | `(32, 32) complex64 C` | — | — | 19.960 | 17.478 | 17.771 | 17.529 | 17.766 | 15.465 |
| 28 | `ifftn(a, axes=(0, 1), norm='forward')` | `(32, 32) complex64 C` | — | — | 20.050 | 17.829 | 17.829 | 17.839 | 17.772 | 15.485 |
| 18 | `fft(a, axis=-1, norm='forward')` | `(1024,) complex64 C` | — | — | 27.202 | 26.103 | 14.508 | 26.127 | 14.596 | 13.498 |
| 19 | `fft(a, axis=-1, norm='ortho')` | `(1024,) complex64 C` | — | — | 27.568 | 26.324 | 14.532 | 26.313 | 14.611 | 13.592 |
| 20 | `ifft(a, axis=-1, norm=None)` | `(1024,) complex64 C` | — | — | 27.184 | 25.921 | 14.444 | 26.111 | 14.533 | 13.502 |
| 22 | `ifft(a, axis=-1, norm='ortho')` | `(1024,) complex64 C` | — | — | 27.574 | 26.522 | 14.567 | 26.440 | 14.642 | 13.769 |
| 4 | `fft(a, axis=-1)` | `(1024,) complex64 C` | 7.285 | 11.020 | 16.715 | 14.015 | 14.185 | 14.073 | 14.324 | 7.851 |
| 17 | `fft(a, axis=-1, norm='backward')` | `(1024,) complex64 C` | — | — | 16.853 | 14.081 | 14.392 | 14.169 | 14.353 | 7.942 |
| 21 | `ifft(a, axis=-1, norm='forward')` | `(1024,) complex64 C` | — | — | 16.840 | 14.377 | 14.378 | 14.383 | 14.324 | 7.872 |
| 32 | `fftn(a, axes=(0, 1, 2), norm='forward')` | `(16, 16, 16) complex64 C` | — | — | 35.871 | 34.376 | 20.451 | 34.311 | 20.488 | 30.782 |
| 33 | `fftn(a, axes=(0, 1, 2), norm='ortho')` | `(16, 16, 16) complex64 C` | — | — | 36.007 | 34.232 | 20.376 | 34.280 | 20.302 | 30.590 |
| 34 | `ifftn(a, axes=(0, 1, 2), norm=None)` | `(16, 16, 16) complex64 C` | — | — | 35.953 | 34.219 | 20.340 | 34.211 | 20.344 | 30.666 |
| 36 | `ifftn(a, axes=(0, 1, 2), norm='ortho')` | `(16, 16, 16) complex64 C` | — | — | 35.717 | 34.144 | 20.197 | 34.085 | 20.180 | 30.300 |
| 14 | `fftn(a, axes=(0, 1, 2))` | `(16, 16, 16) complex64 C` | 10.890 | 15.449 | 22.523 | 20.092 | 20.119 | 20.083 | 20.188 | 18.095 |
| 31 | `fftn(a, axes=(0, 1, 2), norm='backward')` | `(16, 16, 16) complex64 C` | — | — | 22.366 | 19.805 | 20.006 | 19.792 | 19.991 | 18.011 |
| 35 | `ifftn(a, axes=(0, 1, 2), norm='forward')` | `(16, 16, 16) complex64 C` | — | — | 22.450 | 20.019 | 19.976 | 20.021 | 19.980 | 17.957 |
| 77 | `ifft(a, norm=None)` | `(65536,) complex128 C` | — | — | 29.205 | 28.046 | 16.376 | 28.088 | 16.311 | 15.540 |
| 37 | `ifftn(a, axes=(0, 1, 2))` | `(128, 64, 128) complex64 C` | — | — | 35.433 | 33.770 | 20.188 | 33.715 | 20.182 | 30.035 |
| 16 | `fftn(a, axes=(0, 1, 2))` | `(128, 64, 128) complex64 C` | 10.849 | 15.169 | 22.281 | 19.704 | 19.920 | 19.617 | 19.837 | 17.926 |
| 30 | `ifftn(a, axes=(0, 1))` | `(1024, 1024) complex64 C` | — | — | 33.431 | 32.100 | 18.448 | 32.019 | 18.418 | 27.507 |
| 11 | `fftn(a, axes=(0, 1))` | `(1024, 1024) complex64 C` | 8.966 | 13.389 | 20.373 | 17.860 | 18.047 | 17.911 | 18.126 | 15.589 |
| 23 | `ifft(a, axis=-1)` | `(1048576,) complex64 C` | — | — | 29.464 | 28.265 | 16.474 | 28.268 | 16.458 | 15.582 |
| 6 | `fft(a, axis=-1)` | `(1048576,) complex64 C` | 8.974 | 12.782 | 18.440 | 15.849 | 16.123 | 15.843 | 16.097 | 9.623 |
| 1 | `fft(a, axis=-1)` | `(2,) complex64 C` | 7.227 | 10.859 | 16.387 | 13.589 | 13.915 | 13.576 | 13.813 | 7.753 |
| 7 | `fftn(a, axes=(0, 1))` | `(2, 2) complex64 C` | 9.154 | 13.157 | 20.035 | 17.568 | 17.845 | 17.613 | 17.847 | 15.384 |
| 12 | `fftn(a, axes=(0, 1, 2))` | `(2, 2, 2) complex64 C` | 10.829 | 15.142 | 22.236 | 19.670 | 19.798 | 19.672 | 19.784 | 18.070 |
| 2 | `fft(a, axis=-1)` | `(8,) complex64 C` | 7.267 | 10.885 | 16.456 | 13.739 | 14.087 | 13.744 | 13.955 | 7.802 |
| 13 | `fftn(a, axes=(0, 1, 2))` | `(4, 4, 4) complex64 C` | 10.853 | 15.052 | 22.146 | 19.495 | 19.702 | 19.474 | 19.714 | 17.903 |
| 8 | `fftn(a, axes=(0, 1))` | `(8, 8) complex64 C` | 9.065 | 13.199 | 19.951 | 17.568 | 17.776 | 17.549 | 17.804 | 15.429 |
| 3 | `fft(a, axis=-1)` | `(64,) complex64 C` | 7.248 | 10.949 | 16.504 | 13.827 | 13.952 | 13.745 | 14.018 | 7.824 |
| 10 | `fftn(a, axes=(0, 1))` | `(128, 128) complex64 C` | 9.167 | 13.496 | 20.512 | 18.014 | 18.223 | 17.964 | 18.154 | 15.594 |
| 5 | `fft(a, axis=-1)` | `(65536,) complex64 C` | 9.028 | 12.958 | 18.791 | 16.062 | 16.294 | 16.053 | 16.310 | 9.655 |
| 15 | `fftn(a, axes=(0, 1, 2))` | `(64, 64, 64) complex64 C` | 10.993 | 15.468 | 22.704 | 20.115 | 20.379 | 20.139 | 20.376 | 18.129 |
| 76 | `fft(a, norm=None)` | `(1048576,) complex128 C` | — | — | 18.489 | 15.956 | 16.161 | 15.924 | 16.109 | 9.709 |

### R2C normalization

| ID | Request | Operand | Stateful | Minimal | Original | Axes | Fusion | C2R | Ajay | Native |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 72 | `rfft(a, norm='forward')` | `(33,) float32 C` | — | — | 27.397 | 26.267 | 14.490 | 26.318 | 14.504 | 13.556 |
| 78 | `rfftn(a, norm='forward')` | `(64, 64, 64) float64 C` | — | — | 35.254 | 33.510 | 19.565 | 33.540 | 19.614 | 30.668 |
| 73 | `rfft(a, norm='ortho')` | `(1048575,) float32 C` | — | — | 38.042 | 36.937 | 25.303 | 37.005 | 25.380 | 24.041 |
| 60 | `rfft(a)` | `(32,) float32 C` | — | — | 16.587 | 13.688 | 14.078 | 13.778 | 14.017 | 7.728 |
| 64 | `rfft2(a)` | `(8, 16) float32 C` | — | — | 20.137 | 17.800 | 18.018 | 17.826 | 18.006 | 16.046 |
| 68 | `rfftn(a)` | `(4, 8, 16) float32 C` | — | — | 21.962 | 18.719 | 18.835 | 18.688 | 18.922 | 17.221 |
| 69 | `rfftn(a)` | `(128, 64, 128) float32 C` | — | — | 22.241 | 18.947 | 19.161 | 18.904 | 19.139 | 17.293 |
| 65 | `rfft2(a)` | `(1024, 1024) float32 C` | — | — | 20.629 | 18.157 | 18.376 | 18.186 | 18.408 | 16.270 |
| 61 | `rfft(a)` | `(1048576,) float32 C` | — | — | 20.936 | 18.348 | 18.569 | 18.355 | 18.632 | 11.587 |

### Batches and axes variants

| ID | Request | Operand | Stateful | Minimal | Original | Axes | Fusion | C2R | Ajay | Native |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 54 | `ifft2(a, norm=None)` | `(2, 8, 16) complex64 C` | — | — | 33.454 | 33.881 | 20.176 | 33.826 | 20.315 | 28.686 |
| 56 | `ifft2(a, norm='ortho')` | `(2, 8, 16) complex64 C` | — | — | 33.666 | 34.111 | 20.386 | 33.998 | 20.421 | 28.613 |
| 40 | `fft2(a)` | `(2, 8, 16) complex64 C` | — | — | 20.385 | 19.918 | 20.154 | 19.909 | 20.049 | 16.171 |
| 55 | `ifft2(a, norm='forward')` | `(2, 8, 16) complex64 C` | — | — | 20.490 | 20.330 | 20.312 | 20.292 | 20.334 | 16.244 |
| 57 | `ifft2(a, norm=None)` | `(4, 256, 512) complex64 C` | — | — | 33.769 | 34.232 | 20.529 | 34.251 | 20.548 | 29.045 |
| 59 | `ifft2(a, norm='ortho')` | `(4, 256, 512) complex64 C` | — | — | 33.952 | 34.340 | 20.597 | 34.382 | 20.628 | 29.047 |
| 41 | `fft2(a)` | `(4, 256, 512) complex64 C` | — | — | 20.709 | 20.055 | 20.229 | 20.114 | 20.348 | 16.359 |
| 58 | `ifft2(a, norm='forward')` | `(4, 256, 512) complex64 C` | — | — | 20.827 | 20.564 | 20.497 | 20.583 | 20.555 | 16.507 |
| 38 | `fftn(a, axes=(1,))` | `(3, 16) complex64 C` | — | — | 17.281 | 16.807 | 16.921 | 16.902 | 17.016 | 9.052 |
| 44 | `fftn(a, axes=(0, 2))` | `(4, 3, 8) complex64 C` | — | — | 20.953 | 20.352 | 20.608 | 20.387 | 20.685 | 36.096 |
| 46 | `fftn(a, axes=[0, 2])` | `(4, 3, 8) complex64 C` | — | — | 20.885 | 20.339 | 20.579 | 20.372 | 20.630 | 36.031 |
| 48 | `fftn(a, axes=(np.int64(0), np.int64(2)))` | `(4, 3, 8) complex64 C` | — | — | 21.765 | 20.719 | 20.935 | 20.750 | 20.840 | 38.520 |
| 42 | `fftn(a, axes=(1, 0))` | `(8, 16) complex64 C` | — | — | 20.180 | 19.700 | 19.984 | 19.722 | 19.864 | 15.612 |
| 50 | `fftn(a, axes=(0, -1))` | `(8, 16) complex64 C` | — | — | 20.131 | 19.682 | 19.885 | 19.647 | 19.893 | 15.528 |
| 52 | `fftn(a, axes=(2, 3))` | `(2, 3, 8, 16) complex64 C` | — | — | 20.676 | 20.252 | 20.393 | 20.174 | 20.421 | 16.323 |
| 45 | `fftn(a, axes=(0, 2))` | `(128, 4, 1024) complex64 C` | — | — | 21.432 | 20.673 | 20.996 | 20.699 | 21.013 | 37.520 |
| 47 | `fftn(a, axes=[0, 2])` | `(128, 4, 1024) complex64 C` | — | — | 21.364 | 20.730 | 20.930 | 20.734 | 20.952 | 37.941 |
| 49 | `fftn(a, axes=(np.int64(0), np.int64(2)))` | `(128, 4, 1024) complex64 C` | — | — | 22.334 | 21.195 | 21.444 | 21.174 | 21.384 | 39.267 |
| 53 | `fftn(a, axes=(2, 3))` | `(2, 4, 256, 512) complex64 C` | — | — | 20.851 | 20.289 | 20.479 | 20.343 | 20.539 | 16.441 |
| 39 | `fftn(a, axes=(1,))` | `(16, 65536) complex64 C` | — | — | 19.621 | 19.005 | 19.259 | 19.082 | 19.110 | 10.924 |
| 43 | `fftn(a, axes=(1, 0))` | `(1024, 1024) complex64 C` | — | — | 20.342 | 19.754 | 19.964 | 19.768 | 19.978 | 15.682 |
| 51 | `fftn(a, axes=(0, -1))` | `(1024, 1024) complex64 C` | — | — | 20.395 | 19.861 | 20.099 | 19.817 | 19.991 | 15.583 |

### C2C normalization — Fortran and strided operands

| ID | Request | Operand | Stateful | Minimal | Original | Axes | Fusion | C2R | Ajay | Native |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 86 | `ifft(a)` | `(64,) complex64 strided` | — | — | 27.067 | 25.827 | 14.050 | 25.829 | 13.982 | 19.883 |
| 82 | `fft(a)` | `(64,) complex64 strided` | — | — | 16.416 | 13.581 | 13.759 | 13.449 | 13.822 | 13.872 |
| 84 | `ifft(a)` | `(8, 16) complex64 F` | — | — | 29.405 | 28.406 | 14.788 | 28.399 | 14.781 | 23.846 |
| 80 | `fft(a)` | `(8, 16) complex64 F` | — | — | 16.572 | 13.935 | 14.302 | 14.065 | 14.307 | 14.833 |
| 85 | `ifft(a)` | `(16, 65536) complex64 F` | — | — | 31.847 | 30.606 | 16.515 | 30.556 | 16.465 | 25.709 |
| 81 | `fft(a)` | `(16, 65536) complex64 F` | — | — | 18.663 | 16.075 | 16.239 | 16.112 | 16.316 | 16.442 |
| 87 | `ifft(a)` | `(1048576,) complex64 strided` | — | — | 29.238 | 27.939 | 16.181 | 27.891 | 16.318 | 22.411 |
| 83 | `fft(a)` | `(1048576,) complex64 strided` | — | — | 18.558 | 15.701 | 16.037 | 15.731 | 16.010 | 15.657 |

## Recorded data

[`results.json`](results.json) contains case summaries, paired CPU/GPU medians,
validation and source/environment provenance. GPU-event intervals include host
submission gaps. Raw per-call samples are retained in the recorded result folder.

Regenerate the tables and chart from the committed record:

```sh
python benchmarks/fft_nvmath_adapter/report.py \
  benchmarks/fft_nvmath_adapter/results.json --output comparison.md
# Matplotlib is needed only to render the chart.
python benchmarks/fft_nvmath_adapter/plot.py \
  benchmarks/fft_nvmath_adapter/results.json \
  --output benchmarks/fft_nvmath_adapter/warm-cpu-comparison.png
```

Earlier records: [initial Numba fusion](https://github.com/yangcal/cupy/blob/fe8971486d9f1f27f11749f30f1854fa30d689c3/benchmarks/fft_nvmath_adapter/README.md)
and [previous C2R comparison](https://github.com/yangcal/cupy/blob/f4b9e357c/benchmarks/fft_nvmath_adapter/README.md).
