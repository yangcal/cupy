# Warm FFT comparison

Compare the public CuPy FFT paths with a minimal cached nvmath lifecycle.

| Row | What it measures | Definition |
| --- | --- | --- |
| Original | CuPy's nvmath adapter, including compiled stride conversion and cached dtype names. | [`_try_use_nvmath`](../../cupy/fft/_fft_nvmath.py#L202) |
| Ajay | Original plus common-axis shortcuts and skipping no-op scaling setup. | [`_try_use_nvmath_ajay`](../../cupy/fft/_fft_nvmath.py#L337), with adjacent `_normalize_axes_ajay` and `_scale_result_ajay` helpers |
| Native | Public `cupy.fft` with nvmath disabled, using CuPy's cached cuFFT plans. | [`cupy/fft/_fft.py`](../../cupy/fft/_fft.py); selected in [`main.py`](main.py) |
| Stateful | Repeated direct `FFT.execute()` on a separately planned, continuously bound operand; no key, cache lookup or rebinding. Forward C2C only. | `stateful.execute` in [`main.py`](main.py) |
| Minimal | Fresh metadata key, dictionary lookup, operand binding, forward execution and operand release. Forward C2C only. | [`execute_cached`](minimal_lifecycle.py#L18) |

Original, Ajay and Minimal use the same execute candidate and share one
cached plan. Stateful uses that candidate on its own bound object. Native
uses its own plans. Operands and requests are in
[`cases.py`](cases.py).

## Replicate

With a working CUDA 13 build environment, clone and install both sources:

```sh
git clone --recurse-submodules --branch ajay/fft-adapter-request-benchmark \
  https://github.com/yangcal/cupy.git cupy-fft-request
git clone --branch ajay/fft-cache-lifecycle \
  ssh://git@gitlab-master.nvidia.com:12051/cuda-hpc-libraries/python/nvmath-python.git \
  nvmath-fft-request
git -C nvmath-fft-request checkout --detach \
  84d59bc0fa8128142669ec9586537f5bba169254

python -m pip install 'setuptools>=77.0.3' wheel 'Cython==3.2.9' 'numpy==2.5.4'
python -m pip install --no-build-isolation -e './nvmath-fft-request[sysctk13]'
CUPY_USE_CUDA_PYTHON=1 PYTHONPATH="$PWD/nvmath-fft-request" \
  python -m pip install --no-build-isolation -e ./cupy-fft-request
cd cupy-fft-request
```

The nvmath clone requires internal repository access. For general source-build
setup, see [CuPy installation](../../docs/source/install.rst).

Run a short three-operand check, then the full 87-case sweep:

```sh
python -P benchmarks/fft_nvmath_adapter/main.py --suite basic \
  --repeats 20 --warmups 5 --json smoke.json
python -P benchmarks/fft_nvmath_adapter/main.py --json results.json
```

Use `--list-cases` to inspect the requests or `--case SUBSTRING` to select
cases. Defaults are 30 balanced blocks, 500 samples and 20 warmups per row.
Planning and validation are outside timing; nvmath workspace is retained.
Raw samples, environment and source hashes are saved in JSON. The five-row
cases use a balanced ten-order design repeated three times; other cases use
all six permutations five times.

## Results

NVIDIA B200 warm CPU call times in microseconds, measured with
`cupyx.profiler.benchmark` on 2026-10-20. Values are medians of block
medians from 30 balanced blocks, with 500 samples per row per block.
Planning and validation are outside timing. These are host call times,
not isolated GPU kernel times.

All columns come from the same run. Operand descriptions give input
shape, dtype and layout; `strided` means a positive stride-two view.
Minimal and Stateful are included only for the matched forward C2C cases.

Ajay was faster in 83 cases. Four batched `ifft2` cases were slightly
slower (0.12–0.16 µs). All 87 cases passed numerical and
backend-routing checks. These are curated cases, without production
workload weighting.

Tiny forward C2C transforms (complex64, C-contiguous), CPU time in µs:

| Operand | Stateful | Minimal | Original | Ajay | Native |
| --- | ---: | ---: | ---: | ---: | ---: |
| `(2,)` | 7.44 | 11.02 | 16.33 | 13.36 | 7.78 |
| `(2,2)` | 9.23 | 13.42 | 20.05 | 17.31 | 15.48 |
| `(2,2,2)` | 11.03 | 15.18 | 22.05 | 19.05 | 17.74 |

### Full results

| Request | Operand | Stateful | Minimal | Original | Ajay | Native |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `fft(a, axis=-1)` | `(2,)` complex64 C | 7.444 | 11.017 | 16.328 | 13.360 | 7.780 |
| `fft(a, axis=-1)` | `(8,)` complex64 C | 7.442 | 10.980 | 16.382 | 13.150 | 7.770 |
| `fft(a, axis=-1)` | `(64,)` complex64 C | 7.531 | 11.035 | 16.491 | 13.319 | 7.798 |
| `fft(a, axis=-1)` | `(1024,)` complex64 C | 7.507 | 11.085 | 16.641 | 13.440 | 7.894 |
| `fft(a, axis=-1)` | `(65536,)` complex64 C | 9.194 | 12.864 | 18.362 | 15.356 | 9.612 |
| `fft(a, axis=-1)` | `(1048576,)` complex64 C | 9.228 | 12.928 | 18.499 | 15.393 | 9.660 |
| `fftn(a, axes=(0, 1))` | `(2, 2)` complex64 C | 9.235 | 13.424 | 20.049 | 17.306 | 15.476 |
| `fftn(a, axes=(0, 1))` | `(8, 8)` complex64 C | 9.295 | 13.444 | 19.998 | 17.213 | 15.485 |
| `fftn(a, axes=(0, 1))` | `(32, 32)` complex64 C | 9.312 | 13.402 | 20.056 | 17.142 | 15.573 |
| `fftn(a, axes=(0, 1))` | `(128, 128)` complex64 C | 9.360 | 13.611 | 20.165 | 17.324 | 15.595 |
| `fftn(a, axes=(0, 1))` | `(1024, 1024)` complex64 C | 9.211 | 13.582 | 20.317 | 17.518 | 15.621 |
| `fftn(a, axes=(0, 1, 2))` | `(2, 2, 2)` complex64 C | 11.032 | 15.179 | 22.051 | 19.047 | 17.737 |
| `fftn(a, axes=(0, 1, 2))` | `(4, 4, 4)` complex64 C | 11.034 | 15.288 | 22.116 | 19.140 | 17.881 |
| `fftn(a, axes=(0, 1, 2))` | `(16, 16, 16)` complex64 C | 11.128 | 15.545 | 22.468 | 19.511 | 18.035 |
| `fftn(a, axes=(0, 1, 2))` | `(64, 64, 64)` complex64 C | 11.150 | 15.546 | 22.519 | 19.534 | 18.054 |
| `fftn(a, axes=(0, 1, 2))` | `(128, 64, 128)` complex64 C | 11.063 | 15.525 | 22.498 | 19.490 | 18.119 |
| `fft(a, axis=-1, norm='backward')` | `(1024,)` complex64 C | — | — | 16.982 | 13.694 | 7.908 |
| `fft(a, axis=-1, norm='forward')` | `(1024,)` complex64 C | — | — | 27.201 | 25.773 | 13.676 |
| `fft(a, axis=-1, norm='ortho')` | `(1024,)` complex64 C | — | — | 27.317 | 25.886 | 13.637 |
| `ifft(a, axis=-1, norm=None)` | `(1024,)` complex64 C | — | — | 27.036 | 25.693 | 13.536 |
| `ifft(a, axis=-1, norm='forward')` | `(1024,)` complex64 C | — | — | 16.971 | 14.298 | 7.948 |
| `ifft(a, axis=-1, norm='ortho')` | `(1024,)` complex64 C | — | — | 27.389 | 26.006 | 13.647 |
| `ifft(a, axis=-1)` | `(1048576,)` complex64 C | — | — | 28.866 | 27.539 | 15.265 |
| `fftn(a, axes=(0, 1), norm='backward')` | `(32, 32)` complex64 C | — | — | 20.258 | 17.354 | 15.549 |
| `fftn(a, axes=(0, 1), norm='forward')` | `(32, 32)` complex64 C | — | — | 32.625 | 30.831 | 25.962 |
| `fftn(a, axes=(0, 1), norm='ortho')` | `(32, 32)` complex64 C | — | — | 32.863 | 30.947 | 26.431 |
| `ifftn(a, axes=(0, 1), norm=None)` | `(32, 32)` complex64 C | — | — | 32.855 | 31.025 | 26.505 |
| `ifftn(a, axes=(0, 1), norm='forward')` | `(32, 32)` complex64 C | — | — | 20.069 | 17.468 | 15.446 |
| `ifftn(a, axes=(0, 1), norm='ortho')` | `(32, 32)` complex64 C | — | — | 32.857 | 30.946 | 26.139 |
| `ifftn(a, axes=(0, 1))` | `(1024, 1024)` complex64 C | — | — | 33.549 | 31.688 | 26.934 |
| `fftn(a, axes=(0, 1, 2), norm='backward')` | `(16, 16, 16)` complex64 C | — | — | 22.361 | 19.328 | 17.877 |
| `fftn(a, axes=(0, 1, 2), norm='forward')` | `(16, 16, 16)` complex64 C | — | — | 35.337 | 33.372 | 29.363 |
| `fftn(a, axes=(0, 1, 2), norm='ortho')` | `(16, 16, 16)` complex64 C | — | — | 35.232 | 33.239 | 29.223 |
| `ifftn(a, axes=(0, 1, 2), norm=None)` | `(16, 16, 16)` complex64 C | — | — | 35.429 | 33.501 | 29.740 |
| `ifftn(a, axes=(0, 1, 2), norm='forward')` | `(16, 16, 16)` complex64 C | — | — | 22.615 | 20.104 | 18.296 |
| `ifftn(a, axes=(0, 1, 2), norm='ortho')` | `(16, 16, 16)` complex64 C | — | — | 35.623 | 33.572 | 29.559 |
| `ifftn(a, axes=(0, 1, 2))` | `(128, 64, 128)` complex64 C | — | — | 35.067 | 33.083 | 29.012 |
| `fftn(a, axes=(1,))` | `(3, 16)` complex64 C | — | — | 17.155 | 16.368 | 9.076 |
| `fftn(a, axes=(1,))` | `(16, 65536)` complex64 C | — | — | 19.271 | 18.540 | 10.855 |
| `fft2(a)` | `(2, 8, 16)` complex64 C | — | — | 20.223 | 19.473 | 16.274 |
| `fft2(a)` | `(4, 256, 512)` complex64 C | — | — | 20.621 | 19.843 | 16.330 |
| `fftn(a, axes=(1, 0))` | `(8, 16)` complex64 C | — | — | 20.133 | 19.409 | 15.479 |
| `fftn(a, axes=(1, 0))` | `(1024, 1024)` complex64 C | — | — | 20.534 | 19.611 | 15.548 |
| `fftn(a, axes=(0, 2))` | `(4, 3, 8)` complex64 C | — | — | 20.711 | 19.810 | 34.611 |
| `fftn(a, axes=(0, 2))` | `(128, 4, 1024)` complex64 C | — | — | 21.214 | 20.362 | 36.168 |
| `fftn(a, axes=[0, 2])` | `(4, 3, 8)` complex64 C | — | — | 20.751 | 19.934 | 35.291 |
| `fftn(a, axes=[0, 2])` | `(128, 4, 1024)` complex64 C | — | — | 21.265 | 20.454 | 36.374 |
| `fftn(a, axes=(np.int64(0), np.int64(2)))` | `(4, 3, 8)` complex64 C | — | — | 21.418 | 20.185 | 36.629 |
| `fftn(a, axes=(np.int64(0), np.int64(2)))` | `(128, 4, 1024)` complex64 C | — | — | 22.127 | 20.733 | 38.018 |
| `fftn(a, axes=(0, -1))` | `(8, 16)` complex64 C | — | — | 20.248 | 19.449 | 15.630 |
| `fftn(a, axes=(0, -1))` | `(1024, 1024)` complex64 C | — | — | 20.437 | 19.595 | 15.742 |
| `fftn(a, axes=(2, 3))` | `(2, 3, 8, 16)` complex64 C | — | — | 20.623 | 19.908 | 16.237 |
| `fftn(a, axes=(2, 3))` | `(2, 4, 256, 512)` complex64 C | — | — | 20.849 | 19.987 | 16.220 |
| `ifft2(a, norm=None)` | `(2, 8, 16)` complex64 C | — | — | 33.046 | 33.234 | 27.922 |
| `ifft2(a, norm='forward')` | `(2, 8, 16)` complex64 C | — | — | 20.555 | 20.047 | 16.144 |
| `ifft2(a, norm='ortho')` | `(2, 8, 16)` complex64 C | — | — | 33.268 | 33.400 | 27.624 |
| `ifft2(a, norm=None)` | `(4, 256, 512)` complex64 C | — | — | 33.351 | 33.477 | 27.945 |
| `ifft2(a, norm='forward')` | `(4, 256, 512)` complex64 C | — | — | 20.796 | 20.318 | 16.412 |
| `ifft2(a, norm='ortho')` | `(4, 256, 512)` complex64 C | — | — | 33.669 | 33.806 | 27.821 |
| `rfft(a)` | `(32,)` float32 C | — | — | 16.520 | 13.334 | 7.791 |
| `rfft(a)` | `(1048576,)` float32 C | — | — | 20.862 | 17.912 | 11.588 |
| `irfft(a)` | `(17,)` complex64 C | — | — | 53.300 | 52.174 | 26.869 |
| `irfft(a)` | `(524289,)` complex64 C | — | — | 59.335 | 58.259 | 31.228 |
| `rfft2(a)` | `(8, 16)` float32 C | — | — | 20.055 | 17.262 | 15.951 |
| `rfft2(a)` | `(1024, 1024)` float32 C | — | — | 20.549 | 17.754 | 16.223 |
| `irfft2(a)` | `(8, 9)` complex64 C | — | — | 60.802 | 58.813 | 42.711 |
| `irfft2(a)` | `(1024, 513)` complex64 C | — | — | 61.806 | 59.589 | 43.203 |
| `rfftn(a)` | `(4, 8, 16)` float32 C | — | — | 22.000 | 18.302 | 17.160 |
| `rfftn(a)` | `(128, 64, 128)` float32 C | — | — | 22.218 | 18.721 | 17.340 |
| `irfftn(a)` | `(4, 8, 9)` complex64 C | — | — | 63.548 | 61.324 | 44.462 |
| `irfftn(a)` | `(128, 64, 65)` complex64 C | — | — | 63.917 | 61.732 | 44.569 |
| `rfft(a, norm='forward')` | `(33,)` float32 C | — | — | 26.988 | 25.481 | 13.503 |
| `rfft(a, norm='ortho')` | `(1048575,)` float32 C | — | — | 38.064 | 36.517 | 23.866 |
| `irfft2(a, norm='forward')` | `(8, 9)` complex64 C | — | — | 46.338 | 43.600 | 32.792 |
| `irfft2(a, norm='ortho')` | `(512, 513)` complex64 C | — | — | 61.983 | 60.012 | 43.474 |
| `fft(a, norm=None)` | `(1048576,)` complex128 C | — | — | 18.533 | 15.444 | 9.692 |
| `ifft(a, norm=None)` | `(65536,)` complex128 C | — | — | 29.057 | 27.533 | 15.413 |
| `rfftn(a, norm='forward')` | `(64, 64, 64)` float64 C | — | — | 34.680 | 32.470 | 29.555 |
| `irfftn(a, norm='ortho')` | `(64, 64, 33)` complex128 C | — | — | 64.223 | 62.048 | 45.110 |
| `fft(a)` | `(8, 16)` complex64 F | — | — | 16.531 | 13.630 | 14.729 |
| `fft(a)` | `(16, 65536)` complex64 F | — | — | 18.540 | 15.438 | 16.504 |
| `fft(a)` | `(64,)` complex64 strided | — | — | 16.419 | 13.485 | 13.944 |
| `fft(a)` | `(1048576,)` complex64 strided | — | — | 18.475 | 15.433 | 15.777 |
| `ifft(a)` | `(8, 16)` complex64 F | — | — | 29.005 | 27.617 | 23.119 |
| `ifft(a)` | `(16, 65536)` complex64 F | — | — | 31.030 | 29.722 | 24.526 |
| `ifft(a)` | `(64,)` complex64 strided | — | — | 26.711 | 25.271 | 19.669 |
| `ifft(a)` | `(1048576,)` complex64 strided | — | — | 29.118 | 27.694 | 21.816 |
