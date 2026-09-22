# CuPy FFT native/nvmath integration latency

This research benchmark answers two integration questions against the same
public `cupy.fft` entry points in Yang's nvmath integration branch:

1. **Hot cache:** after both backends have cached their plans, does routing
   through nvmath regress repeated calls?
2. **Cold cache:** are there supported FFT problems whose native planning is
   cheap enough that nvmath initialization becomes visible end to end?

The runner switches only `cupy.fft.config.use_nvmath`. It fails if the
requested backend is not represented by the resulting cache entries, so a
silent CuPy fallback cannot be mistaken for nvmath performance.

## Measurement boundaries

Both experiments run in a warm process after initializing CUDA, CuPy,
nvmath-python, cuFFT, and the allocator.

- The **hot** experiment creates and verifies one cached plan, warms repeated
  hits, and then records complete public-call host-return latency. A second
  pass records synchronized completion and a CUDA-event span. The cached
  plan identities must remain unchanged throughout collection.
- The **cold** experiment synchronizes and clears CuPy's plan cache outside
  the timed interval before every sample. It records host-return,
  synchronized completion, and a CUDA-event span from the same public call.
  This is plan-cache cold, not process-first-touch cold.

For a cold call, the CUDA-event span includes device-idle time while the host
performs blocking planning between the two event insertions. It must not be
interpreted as pure kernel execution time. Host-return and synchronized wall
time are the primary cold metrics.

All raw `perf_counter_ns` samples and adjacent clock-pair samples are retained.
Garbage collection is disabled while sampling. Correctness, backend routing,
and cache-plan identity checks run outside the timed regions.

The full fixture matrix covers small through larger transforms, powers of two,
prime and mixed dimensions, batches, direct 1-D/2-D/3-D transforms, C2C/R2C/
C2R, single and double precision, C/F/strided layouts, inverse transforms, and
normalization. Every fixture is expected to route through nvmath.

Intentional fallback cases such as padding/truncation, repeated axes, more
than three transformed axes, explicit plans, callbacks, and multi-GPU
execution are deliberately excluded. They are coverage work, not valid
nvmath-backend timing cases.

## Collection

Use an environment containing this CuPy worktree built with CUDA Python and a
compatible nvmath-python build. Run from outside the CuPy source root so Python
loads the built installation rather than an unbuilt source package.

### ComputeLab convention

Do not compile CuPy or keep virtual environments on NFS. Build the modified
CuPy branch once in node-local storage and retain only its wheel:

```bash
bash benchmarks/fft_nvmath_integration/build_computelab_wheel.sh \
  /home/scratch.ajdesai_ent/artifacts/cupy/fft-nvmath-integration-wheels
```

The script prints the final wheel path. Each benchmark allocation then creates
a fresh node-local runtime environment, installs the persisted wheel and a
pinned nvmath-python package, and writes only results to NFS:

```bash
bash benchmarks/fft_nvmath_integration/run_computelab.sh \
  /home/scratch.ajdesai_ent/artifacts/cupy/fft-nvmath-integration-wheels/<key>/cupy-*.whl
```

The source checkout and all of its submodules must be prepared on the frontend
before obtaining a compute allocation. Neither script performs Git network
operations. The wheel is keyed by CuPy commit, Python version, CUDA major, and
machine architecture, so later allocations reuse it without recompiling.

Quick validation:

```bash
python /path/to/cupy/benchmarks/fft_nvmath_integration/run.py \
  --mode all \
  --suite smoke \
  --hot-warmup 2 \
  --hot-iters 3 \
  --hot-rounds 2 \
  --hot-completion-iters 2 \
  --hot-completion-rounds 2 \
  --cold-warmup 2 \
  --cold-iters 2 \
  --cold-rounds 2 \
  --json /path/to/artifacts/smoke.json
```

Primary hot-cache acceptance check, using the common practical 1-D C2C case:

```bash
python /path/to/cupy/benchmarks/fft_nvmath_integration/run.py \
  --mode hot \
  --fixture c2c_len1024_c64 \
  --json /path/to/artifacts/hot.json
```

The same hot check can later be broadened without changing methodology:

```bash
python /path/to/cupy/benchmarks/fft_nvmath_integration/run.py \
  --mode hot \
  --suite full \
  --json /path/to/artifacts/hot-full.json
```

Full cold-cache sweep:

```bash
python /path/to/cupy/benchmarks/fft_nvmath_integration/run.py \
  --mode cold \
  --suite full \
  --json /path/to/artifacts/cold.json
```

The mode-specific runs intentionally produce independent raw files. That
keeps the quick hot-path acceptance check separate from the longer cold sweep
and allows either experiment to be repeated without replacing the other.

## Reduction and plots

The plotter requires Matplotlib but not a GPU:

```bash
python /path/to/cupy/benchmarks/fft_nvmath_integration/plot.py \
  --json /path/to/artifacts/hot.json \
  --output-dir /path/to/artifacts \
  | tee /path/to/artifacts/hot-report.txt

python /path/to/cupy/benchmarks/fft_nvmath_integration/plot.py \
  --json /path/to/artifacts/cold.json \
  --output-dir /path/to/artifacts \
  | tee /path/to/artifacts/cold-report.txt
```

The hot and cold plots pair native and nvmath means for host return,
synchronized completion, and the CUDA-event span. The cold-candidate plot
places native cold host latency on the x-axis and the nvmath-minus-native gap
on the y-axis. Its lower-left positive-gap cases are the first candidates for
the existing fine-grained initialization/planning profiler.

This benchmark deliberately does not add phase timestamps to the production
adapter. It discovers integration regressions. The existing pre-plan profiler
is the follow-up attribution tool for any fixture that warrants deeper work.
