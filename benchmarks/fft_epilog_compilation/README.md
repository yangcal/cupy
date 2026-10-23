# FFT scaling epilog compilation

Compare two ways of compiling the same runtime-divisor scaling epilog through
the public `cupy.fft` API. Both use the adapter from `nvmath/fft-cache` and a
published nvmath-python package. No nvmath source changes or other adapter
optimizations are included.

| Row | Adapter implementation | Callback compiler |
| --- | --- | --- |
| Numba | [numba_backend/adapter.py](numba_backend/adapter.py) | Documented `nvmath.fft.compile_epilog()` |
| NVRTC | [nvrtc_backend/adapter.py](nvrtc_backend/adapter.py) | CuPy's existing CUDA source-to-LTO-IR compilation path |

Each folder separates request handling (`adapter.py`), normalization and
scalar ownership (`epilog.py`), and compilation (`compiler.py`). The device
callbacks live in [numba_backend/compiler.py](numba_backend/compiler.py) and
[nvrtc_backend/scaling.cu](nvrtc_backend/scaling.cu). Both use the shared
[compilation cache](compilation_cache.py) and [driver](main.py).

Both compile reusable code for each output dtype, target and callback ABI,
using the same bounded process cache and persistent CuPy cache backend. The
cache keys include compiler and source identity; they do not include the
divisor. Numba CUDA adds a compiler dependency for CuPy; NVRTC is already
used by CuPy's CUDA JIT.

Each scaled plan owns an immutable device scalar, uploaded once on its miss.
Both pass `{'ltoir': code, 'data': scalar.data.ptr}` to ordinary
`plan(epilog=...)`; the callback divides by `user_info[0]`. The plan retains
the scalar until cleanup. Norm and direction still partition the plan cache,
because different factors require different immutable scalars: sharing
compiled code does not imply sharing FFT plans.

The NVRTC route follows the convention already used by CuPy's
[FFT callback implementation](../../cupy/fft/_callback.pyx).
The Numba route uses nvmath's
[documented epilog API](https://docs.nvidia.com/cuda/nvmath-python/latest/host-apis/fft/generated/nvmath.fft.compile_epilog.html).

## Run

From this checkout, use a fresh virtual environment and a compatible CUDA
toolkit. These commands select CUDA 13; use the corresponding `cu12` extra
for CUDA 12.

```bash
git submodule update --init --recursive
python -m pip install -U pip
python -m pip install 'nvmath-python[cu13,numba]' cuda-bindings \
  'setuptools>=77' wheel 'Cython>=3.2,<3.3' 'numpy>=2.0,<2.5'
CUPY_USE_CUDA_PYTHON=1 python -m pip install --no-build-isolation -v .

python benchmarks/fft_epilog_compilation/main.py \
  --smoke --output results/epilog-smoke
python benchmarks/fft_epilog_compilation/main.py \
  --output results/epilog-comparison
```

`--smoke` selects a smaller correctness and timing run. Omit it for the full
operand sweep. Add `--warm-only` to skip cold measurements.

## Measurements

[main.py](main.py) checks numerical results and measures warm public calls
through each sibling adapter. `comparison.json` records CPU time and GPU-event
time in microseconds; public-call GPU intervals include host submission gaps.
The full sweep has 67 C-contiguous cases: 1D/2D/3D and batched transforms,
C2C/R2C/C2R, both precisions, and scaling/no-op controls.

`cold.json` records compilation/cache lookup, planning and the whole first
public call. Fresh processes use empty and then populated disk caches; another
measurement creates a new plan with the callback code already in process.
These separate compilation-cache hits from plan-cache hits, without claiming
to control every internal cuFFT cache.

Runtime divisors avoid compiling new code for each factor, at the cost of
scalar ownership and a load during execution. A constant divisor could enable
more compiler folding, but needs more compiled variants. This comparison
isolates the compiler choice with runtime divisors; production policy is open.
