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

| ID | Request | Operand | Native | Original | Ajay | Minimal | Stateful |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `fft(a, axis=-1)` | `(2,)` complex64 C | 7.780 | 16.328 | 13.360 | 11.017 | 7.444 |
| 2 | `fft(a, axis=-1)` | `(8,)` complex64 C | 7.770 | 16.382 | 13.150 | 10.980 | 7.442 |
| 3 | `fft(a, axis=-1)` | `(64,)` complex64 C | 7.798 | 16.491 | 13.319 | 11.035 | 7.531 |
| 4 | `fft(a, axis=-1)` | `(1024,)` complex64 C | 7.894 | 16.641 | 13.440 | 11.085 | 7.507 |
| 5 | `fft(a, axis=-1)` | `(65536,)` complex64 C | 9.612 | 18.362 | 15.356 | 12.864 | 9.194 |
| 6 | `fft(a, axis=-1)` | `(1048576,)` complex64 C | 9.660 | 18.499 | 15.393 | 12.928 | 9.228 |
| 7 | `fftn(a, axes=(0, 1))` | `(2, 2)` complex64 C | 15.476 | 20.049 | 17.306 | 13.424 | 9.235 |
| 8 | `fftn(a, axes=(0, 1))` | `(8, 8)` complex64 C | 15.485 | 19.998 | 17.213 | 13.444 | 9.295 |
| 9 | `fftn(a, axes=(0, 1))` | `(32, 32)` complex64 C | 15.573 | 20.056 | 17.142 | 13.402 | 9.312 |
| 10 | `fftn(a, axes=(0, 1))` | `(128, 128)` complex64 C | 15.595 | 20.165 | 17.324 | 13.611 | 9.360 |
| 11 | `fftn(a, axes=(0, 1))` | `(1024, 1024)` complex64 C | 15.621 | 20.317 | 17.518 | 13.582 | 9.211 |
| 12 | `fftn(a, axes=(0, 1, 2))` | `(2, 2, 2)` complex64 C | 17.737 | 22.051 | 19.047 | 15.179 | 11.032 |
| 13 | `fftn(a, axes=(0, 1, 2))` | `(4, 4, 4)` complex64 C | 17.881 | 22.116 | 19.140 | 15.288 | 11.034 |
| 14 | `fftn(a, axes=(0, 1, 2))` | `(16, 16, 16)` complex64 C | 18.035 | 22.468 | 19.511 | 15.545 | 11.128 |
| 15 | `fftn(a, axes=(0, 1, 2))` | `(64, 64, 64)` complex64 C | 18.054 | 22.519 | 19.534 | 15.546 | 11.150 |
| 16 | `fftn(a, axes=(0, 1, 2))` | `(128, 64, 128)` complex64 C | 18.119 | 22.498 | 19.490 | 15.525 | 11.063 |
| 17 | `fft(a, axis=-1, norm='backward')` | `(1024,)` complex64 C | 7.908 | 16.982 | 13.694 | — | — |
| 18 | `fft(a, axis=-1, norm='forward')` | `(1024,)` complex64 C | 13.676 | 27.201 | 25.773 | — | — |
| 19 | `fft(a, axis=-1, norm='ortho')` | `(1024,)` complex64 C | 13.637 | 27.317 | 25.886 | — | — |
| 20 | `ifft(a, axis=-1, norm=None)` | `(1024,)` complex64 C | 13.536 | 27.036 | 25.693 | — | — |
| 21 | `ifft(a, axis=-1, norm='forward')` | `(1024,)` complex64 C | 7.948 | 16.971 | 14.298 | — | — |
| 22 | `ifft(a, axis=-1, norm='ortho')` | `(1024,)` complex64 C | 13.647 | 27.389 | 26.006 | — | — |
| 23 | `ifft(a, axis=-1)` | `(1048576,)` complex64 C | 15.265 | 28.866 | 27.539 | — | — |
| 24 | `fftn(a, axes=(0, 1), norm='backward')` | `(32, 32)` complex64 C | 15.549 | 20.258 | 17.354 | — | — |
| 25 | `fftn(a, axes=(0, 1), norm='forward')` | `(32, 32)` complex64 C | 25.962 | 32.625 | 30.831 | — | — |
| 26 | `fftn(a, axes=(0, 1), norm='ortho')` | `(32, 32)` complex64 C | 26.431 | 32.863 | 30.947 | — | — |
| 27 | `ifftn(a, axes=(0, 1), norm=None)` | `(32, 32)` complex64 C | 26.505 | 32.855 | 31.025 | — | — |
| 28 | `ifftn(a, axes=(0, 1), norm='forward')` | `(32, 32)` complex64 C | 15.446 | 20.069 | 17.468 | — | — |
| 29 | `ifftn(a, axes=(0, 1), norm='ortho')` | `(32, 32)` complex64 C | 26.139 | 32.857 | 30.946 | — | — |
| 30 | `ifftn(a, axes=(0, 1))` | `(1024, 1024)` complex64 C | 26.934 | 33.549 | 31.688 | — | — |
| 31 | `fftn(a, axes=(0, 1, 2), norm='backward')` | `(16, 16, 16)` complex64 C | 17.877 | 22.361 | 19.328 | — | — |
| 32 | `fftn(a, axes=(0, 1, 2), norm='forward')` | `(16, 16, 16)` complex64 C | 29.363 | 35.337 | 33.372 | — | — |
| 33 | `fftn(a, axes=(0, 1, 2), norm='ortho')` | `(16, 16, 16)` complex64 C | 29.223 | 35.232 | 33.239 | — | — |
| 34 | `ifftn(a, axes=(0, 1, 2), norm=None)` | `(16, 16, 16)` complex64 C | 29.740 | 35.429 | 33.501 | — | — |
| 35 | `ifftn(a, axes=(0, 1, 2), norm='forward')` | `(16, 16, 16)` complex64 C | 18.296 | 22.615 | 20.104 | — | — |
| 36 | `ifftn(a, axes=(0, 1, 2), norm='ortho')` | `(16, 16, 16)` complex64 C | 29.559 | 35.623 | 33.572 | — | — |
| 37 | `ifftn(a, axes=(0, 1, 2))` | `(128, 64, 128)` complex64 C | 29.012 | 35.067 | 33.083 | — | — |
| 38 | `fftn(a, axes=(1,))` | `(3, 16)` complex64 C | 9.076 | 17.155 | 16.368 | — | — |
| 39 | `fftn(a, axes=(1,))` | `(16, 65536)` complex64 C | 10.855 | 19.271 | 18.540 | — | — |
| 40 | `fft2(a)` | `(2, 8, 16)` complex64 C | 16.274 | 20.223 | 19.473 | — | — |
| 41 | `fft2(a)` | `(4, 256, 512)` complex64 C | 16.330 | 20.621 | 19.843 | — | — |
| 42 | `fftn(a, axes=(1, 0))` | `(8, 16)` complex64 C | 15.479 | 20.133 | 19.409 | — | — |
| 43 | `fftn(a, axes=(1, 0))` | `(1024, 1024)` complex64 C | 15.548 | 20.534 | 19.611 | — | — |
| 44 | `fftn(a, axes=(0, 2))` | `(4, 3, 8)` complex64 C | 34.611 | 20.711 | 19.810 | — | — |
| 45 | `fftn(a, axes=(0, 2))` | `(128, 4, 1024)` complex64 C | 36.168 | 21.214 | 20.362 | — | — |
| 46 | `fftn(a, axes=[0, 2])` | `(4, 3, 8)` complex64 C | 35.291 | 20.751 | 19.934 | — | — |
| 47 | `fftn(a, axes=[0, 2])` | `(128, 4, 1024)` complex64 C | 36.374 | 21.265 | 20.454 | — | — |
| 48 | `fftn(a, axes=(np.int64(0), np.int64(2)))` | `(4, 3, 8)` complex64 C | 36.629 | 21.418 | 20.185 | — | — |
| 49 | `fftn(a, axes=(np.int64(0), np.int64(2)))` | `(128, 4, 1024)` complex64 C | 38.018 | 22.127 | 20.733 | — | — |
| 50 | `fftn(a, axes=(0, -1))` | `(8, 16)` complex64 C | 15.630 | 20.248 | 19.449 | — | — |
| 51 | `fftn(a, axes=(0, -1))` | `(1024, 1024)` complex64 C | 15.742 | 20.437 | 19.595 | — | — |
| 52 | `fftn(a, axes=(2, 3))` | `(2, 3, 8, 16)` complex64 C | 16.237 | 20.623 | 19.908 | — | — |
| 53 | `fftn(a, axes=(2, 3))` | `(2, 4, 256, 512)` complex64 C | 16.220 | 20.849 | 19.987 | — | — |
| 54 | `ifft2(a, norm=None)` | `(2, 8, 16)` complex64 C | 27.922 | 33.046 | 33.234 | — | — |
| 55 | `ifft2(a, norm='forward')` | `(2, 8, 16)` complex64 C | 16.144 | 20.555 | 20.047 | — | — |
| 56 | `ifft2(a, norm='ortho')` | `(2, 8, 16)` complex64 C | 27.624 | 33.268 | 33.400 | — | — |
| 57 | `ifft2(a, norm=None)` | `(4, 256, 512)` complex64 C | 27.945 | 33.351 | 33.477 | — | — |
| 58 | `ifft2(a, norm='forward')` | `(4, 256, 512)` complex64 C | 16.412 | 20.796 | 20.318 | — | — |
| 59 | `ifft2(a, norm='ortho')` | `(4, 256, 512)` complex64 C | 27.821 | 33.669 | 33.806 | — | — |
| 60 | `rfft(a)` | `(32,)` float32 C | 7.791 | 16.520 | 13.334 | — | — |
| 61 | `rfft(a)` | `(1048576,)` float32 C | 11.588 | 20.862 | 17.912 | — | — |
| 62 | `irfft(a)` | `(17,)` complex64 C | 26.869 | 53.300 | 52.174 | — | — |
| 63 | `irfft(a)` | `(524289,)` complex64 C | 31.228 | 59.335 | 58.259 | — | — |
| 64 | `rfft2(a)` | `(8, 16)` float32 C | 15.951 | 20.055 | 17.262 | — | — |
| 65 | `rfft2(a)` | `(1024, 1024)` float32 C | 16.223 | 20.549 | 17.754 | — | — |
| 66 | `irfft2(a)` | `(8, 9)` complex64 C | 42.711 | 60.802 | 58.813 | — | — |
| 67 | `irfft2(a)` | `(1024, 513)` complex64 C | 43.203 | 61.806 | 59.589 | — | — |
| 68 | `rfftn(a)` | `(4, 8, 16)` float32 C | 17.160 | 22.000 | 18.302 | — | — |
| 69 | `rfftn(a)` | `(128, 64, 128)` float32 C | 17.340 | 22.218 | 18.721 | — | — |
| 70 | `irfftn(a)` | `(4, 8, 9)` complex64 C | 44.462 | 63.548 | 61.324 | — | — |
| 71 | `irfftn(a)` | `(128, 64, 65)` complex64 C | 44.569 | 63.917 | 61.732 | — | — |
| 72 | `rfft(a, norm='forward')` | `(33,)` float32 C | 13.503 | 26.988 | 25.481 | — | — |
| 73 | `rfft(a, norm='ortho')` | `(1048575,)` float32 C | 23.866 | 38.064 | 36.517 | — | — |
| 74 | `irfft2(a, norm='forward')` | `(8, 9)` complex64 C | 32.792 | 46.338 | 43.600 | — | — |
| 75 | `irfft2(a, norm='ortho')` | `(512, 513)` complex64 C | 43.474 | 61.983 | 60.012 | — | — |
| 76 | `fft(a, norm=None)` | `(1048576,)` complex128 C | 9.692 | 18.533 | 15.444 | — | — |
| 77 | `ifft(a, norm=None)` | `(65536,)` complex128 C | 15.413 | 29.057 | 27.533 | — | — |
| 78 | `rfftn(a, norm='forward')` | `(64, 64, 64)` float64 C | 29.555 | 34.680 | 32.470 | — | — |
| 79 | `irfftn(a, norm='ortho')` | `(64, 64, 33)` complex128 C | 45.110 | 64.223 | 62.048 | — | — |
| 80 | `fft(a)` | `(8, 16)` complex64 F | 14.729 | 16.531 | 13.630 | — | — |
| 81 | `fft(a)` | `(16, 65536)` complex64 F | 16.504 | 18.540 | 15.438 | — | — |
| 82 | `fft(a)` | `(64,)` complex64 strided | 13.944 | 16.419 | 13.485 | — | — |
| 83 | `fft(a)` | `(1048576,)` complex64 strided | 15.777 | 18.475 | 15.433 | — | — |
| 84 | `ifft(a)` | `(8, 16)` complex64 F | 23.119 | 29.005 | 27.617 | — | — |
| 85 | `ifft(a)` | `(16, 65536)` complex64 F | 24.526 | 31.030 | 29.722 | — | — |
| 86 | `ifft(a)` | `(64,)` complex64 strided | 19.669 | 26.711 | 25.271 | — | — |
| 87 | `ifft(a)` | `(1048576,)` complex64 strided | 21.816 | 29.118 | 27.694 | — | — |

<img width="3600" height="1300" alt="image" src="https://github.com/user-attachments/assets/d7d71d43-6a13-436c-82cf-13ba02a4b468" />


