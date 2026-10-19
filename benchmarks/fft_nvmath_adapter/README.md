# FFT adapter request overhead

Candidate functions live beside the originals in `cupy/fft/_fft_nvmath.py`:
`_normalize_axes`, `_scale_result` and `_try_use_nvmath` each have an `_ajay`
version. Both handlers use compiled stride conversion and cached dtype names.
The public default still uses the original axes/scaling implementation.

`main.py` selects the real module's original or Ajay hook outside timing and
calls public `cupy.fft`. Both rows use the same actual cache-owned FFT plan.
The comparable forward C2C size cases also time the saved minimal lifecycle
in `minimal_lifecycle.py`, using a dictionary referencing that same plan.
The minimal function fetches current metadata and builds a fresh canonical
key on every call; prepared axes/options are supplied outside timing.
Other cases compare original versus Ajay because the saved minimal function
does not implement their inverse/scaling semantics.

## Required environment

Use Linux with Python 3.11 or newer, an editable CUDA-Python CuPy build of
this checkout, and a full nvmath build at
`84d59bc0fa8128142669ec9586537f5bba169254`. The benchmark checks
that CuPy imports this checkout's adapter, rather than another installed copy.
Both handlers and the minimal lifecycle require compiled
`nvmath.internal._strides_helpers.element_strides` and cached
`nvmath.internal.tensor_ifc._numpy_dtype_name`; there is no provider fallback.
A matching package version alone does not establish the required backend.

The candidate contains the `ajay/inline-stacked-decorators` execute version
(revision `eafc59f5f7f335957c34f8aa5a1a0d449694efd0`). Its FFT source is
identical, including the enforced `FFT.execute` SHA256:
`6c69df10b7762d6e04ad70b24c12857dcd7cbd360fe2ae2f8805a0eae6630779`.
The inline revision alone lacks the metadata providers; build the complete
candidate including its compiled extensions.

This is a research prerequisite for the adapter module itself, including its
original handler: the metadata helpers are imported eagerly. Supporting
published nvmath versions requires a supported dependency contract or
compatibility handling before this patch is promoted to production.

## Build the research comparison

After the CuPy branch is pushed, prepare both checkouts on your workstation
or cluster frontend. The nvmath branch is in NVIDIA's internal GitLab;
reviewers need access to that repository. A public nvmath release or a
stock CuPy wheel is not a replacement for either source build.

```sh
git clone --recurse-submodules --branch ajay/fft-adapter-request-benchmark \
  https://github.com/yangcal/cupy.git cupy-fft-request
git clone --branch ajay/fft-cache-lifecycle \
  ssh://git@gitlab-master.nvidia.com:12051/cuda-hpc-libraries/python/nvmath-python.git \
  nvmath-fft-request
git -C nvmath-fft-request checkout --detach \
  84d59bc0fa8128142669ec9586537f5bba169254
```

The following ordinary source-install recipe assumes Python 3.12, a C++17
compiler, a compatible NVIDIA driver, and a complete system CUDA 13.4
toolkit. The nvmath build uses host CPU count for parallelism, so allow
adequate build memory. Set `CUDA_PATH` to its actual directory, including
NVCC, the profiler API headers and CUDA libraries. It does not install a
system toolkit. See [CuPy installation](../../docs/source/install.rst)
for toolkit prerequisites.

On ComputeLab, keep those prepared checkouts on shared storage. Run the
following in Bash on the GPU node, from their parent directory; compilation
and the fresh environment stay in node-local `/tmp`.

```sh
FFT_CUPY_SOURCE=$(realpath cupy-fft-request)
FFT_NVMATH_SOURCE=$(realpath nvmath-fft-request)
FFT_WORK=$(mktemp -d /tmp/fft-adapter.XXXXXX)
rsync -a --exclude=.git "$FFT_CUPY_SOURCE/" "$FFT_WORK/cupy/"
rsync -a --exclude=.git "$FFT_NVMATH_SOURCE/" "$FFT_WORK/nvmath/"
python3.12 -m venv "$FFT_WORK/venv"
source "$FFT_WORK/venv/bin/activate"

: "${CUDA_PATH:?Set CUDA_PATH to your complete CUDA 13.4 toolkit}"
export CUDA_PATH
test -x "$CUDA_PATH/bin/nvcc"
test -f "$CUDA_PATH/include/cuda_profiler_api.h"
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PIP_CACHE_DIR="$FFT_WORK/pip-cache"
python -m pip install 'setuptools>=77.0.3' wheel 'Cython==3.2.9' \
  'numpy==2.5.4'
python -m pip install --no-build-isolation -e "$FFT_WORK/nvmath[sysctk13]"
CUPY_USE_CUDA_PYTHON=1 CUPY_NUM_BUILD_JOBS=8 PYTHONPATH="$FFT_WORK/nvmath" \
  python -m pip install --no-build-isolation -e "$FFT_WORK/cupy"
python -m pip check
cd "$FFT_WORK/cupy"
```

Then run the smoke check and sweep below. Choose a durable shared-storage
path for `--json` when running from `/tmp`. The benchmark rejects an
incorrect checkout, execute implementation, or metadata provider before
timing. The measured B200 environment used pip-provided CUDA 13.4.3
components rather than a system toolkit; this recipe uses the repositories'
standard installation flow and has not had a separate clean-machine run.

## Run from the repository root

Inspect the case list without importing GPU packages:

```sh
python -P benchmarks/fft_nvmath_adapter/main.py --list-cases
```

Run the size/regression sweep (the default suite):

```sh
python -P benchmarks/fft_nvmath_adapter/main.py --suite sweep --json results.json
```

The sweep covers 1D/2D/3D sizes from tiny transforms to approximately one
million input elements, plus inverse/scaled norms, general axes and batches,
real transforms, double precision, and F/positive-stride layouts. It is a
curated matrix rather than a Cartesian product or weighted production mix.

For the historical tiny 1D/2D/3D comparison and a short smoke check:

```sh
python -P benchmarks/fft_nvmath_adapter/main.py --suite basic --blocks 6 --repeats 20 --warmups 5 --json smoke.json
```

Defaults use 12 balanced blocks, 500 samples and 20 warmups per row/block,
retained workspace and a fixed row-order seed. Planning and numerical checks
occur outside sampling. CPU host times come from uninstrumented
`cupyx.profiler.benchmark`, with GC disabled during sampling. JSON records
raw samples, paired block differences, routing checks and source provenance.
Preflight first requires bitwise agreement between rows, then checks each
against a double-precision NumPy reference using nvmath's 100-epsilon L2
criterion per transformed batch. Norm and maximum absolute errors are saved.
Every negative original-minus-Ajay median is reported, including small or
inconsistent changes; any regression label is a candidate for confirmation.
GPU event intervals include host gaps. This measures warm request overhead;
it does not isolate axes alone or claim a production hit percentage.

## B200 results, 2026-10-19

The completed run `fft-adapter-size-sweep-20261019T230816Z-egFUkp` used the
defaults above on `umb-b200-249`: NVIDIA B200, driver 615.71.09, Python
3.12.3, NumPy 2.5.4, CuPy 14.3.0.dev0, and the required nvmath candidate.
All 87 cases passed numerical and nvmath-routing checks. Original and Ajay
outputs agreed bitwise on both fresh operands; comparable minimal outputs
did too. Every raw median and paired summary was checked against the saved
samples, including all 1,044 checkpoint blocks.

Savings below are medians of paired original-minus-Ajay block medians in
microseconds. Each of the 83 positive cases improved in all 12 blocks.

| Cases | Improved / tested | Savings range (us) |
| --- | ---: | ---: |
| Default-forward C2C sizes | 16 / 16 | 2.694 to 3.007 |
| Norms and inverse transforms | 21 / 21 | 1.243 to 2.911 |
| General axes and batches | 18 / 22 | -0.197 to 1.494 |
| Real transforms and double precision | 20 / 20 | 1.131 to 3.495 |
| F and positive-stride layouts | 8 / 8 | 1.134 to 2.970 |

The four negative cases were batched `ifft2` on the last two axes, with
`norm=None` or `norm="ortho"`. They require scaling and miss the full-axis
tuple lookup. The added guards are a plausible cause of the slowdown;
this experiment does not isolate that attribution.

| Operand shape | Norm | Slowdown (us) | Slowdown (%) |
| --- | --- | ---: | ---: |
| (2, 8, 16) | None | 0.197 | 0.58 |
| (2, 8, 16) | ortho | 0.160 | 0.47 |
| (4, 256, 512) | None | 0.175 | 0.51 |
| (4, 256, 512) | ortho | 0.149 | 0.43 |

These slowdowns occurred in 10 or 11 of 12 blocks and remain negative in
both row-order subsets. They fall below the script's material-regression
flag but should not be treated as evidence of no regression.

The historical tiny cases retain the same execute candidate in every row:

| Shape | Original minus minimal (us) | Ajay minus minimal (us) | Savings (us) |
| --- | ---: | ---: | ---: |
| (2,) | 5.572 | 2.662 | 2.979 |
| (2, 2) | 6.942 | 4.202 | 2.760 |
| (2, 2, 2) | 7.169 | 4.308 | 2.849 |

Each difference is summarized independently from paired blocks, so table
columns need not add exactly. Across the 16 size cases, Ajay-minus-minimal
ranges from 2.662 to 3.098 us for 1D, 4.202 to 4.430 us for 2D, and 4.308
to 4.595 us for 3D. The 2-3 us target is not reached for all ranks.

The saved result directory contains the exact measured source snapshots.
The adapter is unchanged since that run; the only subsequent Python change
adds the CuPy tensor-interface source hash to provenance collection.
Raw results and environment setup remain outside this research patch.
This sweep does not validate cold performance, intentional fallbacks,
callback or explicit-plan contexts, execution errors, or multiple streams.
