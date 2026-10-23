"""Compare two compiler frontends through otherwise identical CuPy adapters."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
import tempfile
from time import perf_counter
import warnings


HERE = Path(__file__).resolve().parent
BACKENDS = ("numba", "nvrtc")


def cases(smoke=False):
    # Shapes describe the full real transform, including for C2R. Its stored
    # input has a shortened final transformed axis, produced by NumPy below.
    shapes = [
        ((32,), (0,)),
        ((1024,), (0,)),
        ((65536,), (0,)),
        ((8, 16), (0, 1)),
        ((128, 256), (0, 1)),
        ((4, 8, 16), (0, 1, 2)),
        ((16, 32, 64), (0, 1, 2)),
        ((8, 32), (1,)),
    ]
    if smoke:
        shapes = shapes[:1]
    result = []
    for shape, axes in shapes:
        for precision in ("single", "double"):
            for family, norm in [
                ("fft", "ortho"),
                ("ifft", "backward"),
                ("rfft", "forward"),
                ("irfft", "ortho"),
            ]:
                rank = len(axes)
                public = family + (
                    "2" if rank == 2 else "n" if rank == 3 else ""
                )
                name = (
                    f"{public}_{'x'.join(map(str, shape))}_{precision}_{norm}"
                )
                result.append(
                    dict(
                        name=name,
                        public=public,
                        shape=shape,
                        axes=axes,
                        precision=precision,
                        norm=norm,
                    )
                )
    # Controls: no scale by definition, and unit normalization for singleton N.
    for norm, shape in [(None, (32,)), ("forward", (32,)), ("ortho", (1,))]:
        public = "ifft" if norm == "forward" else "fft"
        result.append(
            dict(
                name=f"control_{public}_{shape[0]}_{norm}",
                public=public,
                shape=shape,
                axes=(0,),
                precision="single",
                norm=norm,
            )
        )
    return result


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def kwargs_for(case):
    axis = (
        {"axis": case["axes"][0]}
        if len(case["axes"]) == 1
        else {"axes": case["axes"]}
    )
    return dict(axis, norm=case["norm"])


def inputs(case, seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    real = "float32" if case["precision"] == "single" else "float64"
    complex_type = "complex64" if real == "float32" else "complex128"
    host = rng.standard_normal(case["shape"])
    if case["public"].startswith("irfft"):
        host = np.fft.rfftn(host, axes=case["axes"]).astype(complex_type)
    elif case["public"].startswith("rfft"):
        host = host.astype(real)
    else:
        host = (host + 1j * rng.standard_normal(case["shape"])).astype(
            complex_type
        )
    reference_input = host.astype(
        np.complex128 if np.iscomplexobj(host) else np.float64
    )
    wanted = getattr(np.fft, case["public"])(
        reference_input, **kwargs_for(case)
    )
    output_dtype = real if case["public"].startswith("irfft") else complex_type
    return host, wanted, output_dtype


def check_result(result, wanted, output_dtype, axes):
    import numpy as np

    if result.shape != wanted.shape or result.dtype.name != output_dtype:
        raise RuntimeError("Incorrect result shape or dtype")
    if wanted.dtype.name not in ("float64", "complex128"):
        raise RuntimeError("Reference must use double precision")
    actual = result.get()
    if not np.isfinite(actual).all():
        raise RuntimeError("Nonfinite FFT output")
    error = np.sqrt(np.sum(np.abs(actual - wanted) ** 2, axis=axes))
    scale = np.sqrt(np.sum(np.abs(wanted) ** 2, axis=axes))
    relative = np.divide(
        error, scale, out=np.zeros_like(error), where=scale != 0
    )
    bound = 100 * np.finfo(output_dtype).eps
    if np.max(relative) > bound or np.any((scale == 0) & (error > bound)):
        raise RuntimeError(
            f"FFT normalization error: {np.max(relative)} > {bound}"
        )
    return float(np.max(relative))


@contextmanager
def routed(cp):
    """Change the hook outside timing; measure public cupy.fft calls."""
    module = importlib.import_module("cupy.fft._fft")
    if not module._is_cuda_python_build:
        raise RuntimeError("Rebuild this branch with CUPY_USE_CUDA_PYTHON=1")
    installed = Path(importlib.import_module("cupy.fft._fft_nvmath").__file__)
    expected = HERE.parents[1] / "cupy/fft/_fft_nvmath.py"
    if installed.read_bytes() != expected.read_bytes():
        raise RuntimeError(
            "Installed CuPy adapter differs from this checkout; "
            "build and install this branch before benchmarking"
        )
    original = module._try_use_nvmath
    enabled = cp.fft.config.use_nvmath
    cp.fft.config.use_nvmath = True
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="nvmath-python rejected")
            yield lambda backend: setattr(
                module,
                "_try_use_nvmath",
                importlib.import_module(
                    backend + "_backend.adapter"
                ).try_use_nvmath,
            )
    finally:
        module._try_use_nvmath = original
        cp.fft.config.use_nvmath = enabled


def plans(cache):
    return {key: node.plan for key, node in cache}


def provenance(cp):
    from nvmath.bindings import cufft

    versions = {}
    for package in (
        "cupy",
        "nvmath-python",
        "numpy",
        "cuda-bindings",
        "numba",
        "numba-cuda",
        "llvmlite",
    ):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    prop = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = prop["name"]
    files = sorted(
        p for p in HERE.rglob("*") if p.suffix in (".py", ".cu")
    ) + [Path(importlib.import_module("cupy.fft._fft_nvmath").__file__)]
    return dict(
        hostname=platform.node(),
        cpu=cpu,
        architecture=platform.machine(),
        python=sys.version,
        executable=sys.executable,
        gpu=name.decode() if isinstance(name, bytes) else name,
        driver=cp.cuda.runtime.driverGetVersion(),
        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
        cufft=cufft.get_version(),
        packages=versions,
        nvmath_source=metadata.distribution("nvmath-python").read_text(
            "direct_url.json"
        ),
        sources={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
    )


def warm_run(args, selected):
    import cupy as cp
    import numpy as np
    from cupyx.profiler import benchmark

    cache = cp.fft.config.get_plan_cache()
    if not cache.is_enabled or cache.get_size() < len(BACKENDS):
        raise RuntimeError(
            "Enable a CuPy plan cache with at least two entries"
        )
    payload = dict(
        status="running", provenance=provenance(cp), cases=[], code_variants={}
    )
    destination = args.output / "comparison.json"
    with routed(cp) as select:
        try:
            for index, case in enumerate(selected, 1):
                print(
                    f"[case] {index}/{len(selected)} {case['name']}",
                    flush=True,
                )
                cache.clear()
                host, wanted, dtype = inputs(case, 1234)
                second, wanted_second, _ = inputs(case, 5678)
                operand, other = cp.asarray(host), cp.asarray(second)
                public = getattr(cp.fft, case["public"])
                kwargs = kwargs_for(case)
                row = dict(
                    case,
                    checks={},
                    blocks=[],
                    input_layout=dict(
                        shape=operand.shape,
                        strides=operand.strides,
                        dtype=operand.dtype.name,
                    ),
                    output_shape=wanted.shape,
                )
                payload["cases"].append(row)
                for backend in BACKENDS:
                    select(backend)
                    row["checks"][backend] = [
                        check_result(
                            public(x, **kwargs), target, dtype, case["axes"]
                        )
                        for x, target in [
                            (operand, wanted),
                            (other, wanted_second),
                        ]
                    ]
                    np.testing.assert_array_equal(operand.get(), host)
                    np.testing.assert_array_equal(other.get(), second)
                expected_plans = plans(cache)
                if len(expected_plans) != 2:
                    raise RuntimeError(
                        "Expected two distinct nvmath cache entries"
                    )
                # Callback-free controls must stay callback-free. Scaled cases
                # must use fused nvmath plans; a native fallback fails the run.
                for plan in expected_plans.values():
                    divisor = plan._cupy_fft_epilog_divisor
                    if bool(plan._cupy_fft_epilog_ltoir) != (divisor != 1):
                        raise RuntimeError("Incorrect callback eligibility")
                    if (plan._cupy_fft_epilog_data is not None) != (
                        divisor != 1
                    ):
                        raise RuntimeError(
                            "Incorrect divisor buffer ownership"
                        )
                    if plan._cupy_fft_epilog_data is not None:
                        np.testing.assert_array_equal(
                            plan._cupy_fft_epilog_data.get(),
                            np.asarray(
                                [divisor],
                                dtype=np.finfo(plan.result_data_type).dtype,
                            ),
                        )
                row["normalization"] = {
                    key[1]: {
                        "divisor": plan._cupy_fft_epilog_divisor,
                        "ltoir_sha256": (
                            hashlib.sha256(
                                plan._cupy_fft_epilog_ltoir
                            ).hexdigest()
                            if plan._cupy_fft_epilog_ltoir
                            else None
                        ),
                    }
                    for key, plan in expected_plans.items()
                }
                for key, plan in expected_plans.items():
                    if plan._cupy_fft_epilog_ltoir:
                        variant = f"{key[1]}:{plan.result_data_type}"
                        code_hash = row["normalization"][key[1]][
                            "ltoir_sha256"
                        ]
                        previous = payload["code_variants"].setdefault(
                            variant, code_hash
                        )
                        if previous != code_hash:
                            raise RuntimeError(
                                "A divisor changed compiled code"
                            )
                was_enabled = gc.isenabled()
                gc.disable()
                try:
                    for block in range(args.blocks):
                        order = BACKENDS if block % 2 == 0 else BACKENDS[::-1]
                        print(
                            f"[warm] {case['name']} {block + 1}/{args.blocks}",
                            flush=True,
                        )
                        samples = {}
                        for backend in order:
                            select(backend)
                            timing = benchmark(
                                public,
                                args=(operand,),
                                kwargs=kwargs,
                                n_repeat=args.repeats,
                                n_warmup=10,
                            )
                            samples[backend] = {
                                "cpu_us": (timing.cpu_times * 1e6).tolist(),
                                "gpu_event_us": (
                                    timing.gpu_times[0] * 1e6
                                ).tolist(),
                            }
                            if plans(cache) != expected_plans:
                                raise RuntimeError(
                                    "A timed request replaced a cached plan"
                                )
                        row["blocks"].append(
                            dict(order=order, samples=samples)
                        )
                finally:
                    if was_enabled:
                        gc.enable()
                row["median_us"] = {
                    backend: {
                        metric: median(
                            median(b["samples"][backend][metric])
                            for b in row["blocks"]
                        )
                        for metric in ("cpu_us", "gpu_event_us")
                    }
                    for backend in BACKENDS
                }
                row["paired_nvrtc_minus_numba_us"] = {
                    metric: median(
                        median(b["samples"]["nvrtc"][metric])
                        - median(b["samples"]["numba"][metric])
                        for b in row["blocks"]
                    )
                    for metric in ("cpu_us", "gpu_event_us")
                }
                print(json.dumps(row["median_us"]), flush=True)
                cp.cuda.get_current_stream().synchronize()
                np.testing.assert_array_equal(operand.get(), host)
                save(destination, payload)
            payload["status"] = "complete"
        except BaseException as error:
            payload.update(status="failed", error=repr(error))
            raise
        finally:
            cp.cuda.get_current_stream().synchronize()
            cache.clear()
            save(destination, payload)


def cold_worker(args):
    import cupy as cp
    import compilation_cache
    from unittest.mock import patch

    case = next(c for c in cases(smoke=True) if c["name"] == args.case)
    host, wanted, dtype = inputs(case, 1234)
    operand = cp.asarray(host)
    cache = cp.fft.config.get_plan_cache()
    cache.clear()
    cp.cuda.get_current_stream().synchronize()
    with routed(cp) as select:
        select(args.worker)
        public = getattr(cp.fft, case["public"])
        adapter = importlib.import_module(args.worker + "_backend.adapter")
        original_plan = adapter.CachedFFT.plan
        plan_times = []

        def timed_plan(plan, *positional, **keywords):
            started = perf_counter()
            try:
                return original_plan(plan, *positional, **keywords)
            finally:
                plan_times.append((perf_counter() - started) * 1e3)

        # Only the coarse cold measurement has this single timing wrapper.
        # Warm measurements call the library methods without instrumentation.
        with patch.object(adapter.CachedFFT, "plan", timed_plan):
            started = perf_counter()
            out = public(operand, **kwargs_for(case))
            cpu_ms = (perf_counter() - started) * 1e3
        cp.cuda.get_current_stream().synchronize()
        complete_ms = (perf_counter() - started) * 1e3
        first_lookup = compilation_cache.cache_info()["last_lookup"]
        if len(plans(cache)) != 1 or first_lookup is None:
            raise RuntimeError(
                "Cold request did not create a fused nvmath plan"
            )
        accuracy = check_result(out, wanted, dtype, case["axes"])
        # An ordinary plan miss with compiled code already in process memory.
        cache.clear()
        started = perf_counter()
        public(operand, **kwargs_for(case))
        memory_cpu_ms = (perf_counter() - started) * 1e3
        cp.cuda.get_current_stream().synchronize()
        memory_lookup = compilation_cache.cache_info()["last_lookup"]
        if memory_lookup["level"] != "memory":
            raise RuntimeError(
                "Expected a process IR cache hit on the second plan"
            )
        result = dict(
            case=case,
            backend=args.worker,
            provenance=provenance(cp),
            first_call_cpu_ms=cpu_ms,
            first_call_complete_ms=complete_ms,
            plan_ms=plan_times[0],
            first_ir_lookup=first_lookup,
            relative_l2=accuracy,
            memory_ir_plan_miss_cpu_ms=memory_cpu_ms,
            memory_ir_lookup=memory_lookup,
        )
        cache.clear()
        save(args.worker_result, result)


def cold_run(args, temporary):
    # Four output signatures. Each pair has separate fresh processes but shares
    # only its disk caches, so the second observes persistent IR reuse.
    selected = [
        c
        for c in cases(smoke=True)
        if c["public"] in ("fft", "irfft")
        and c["norm"] == "ortho"
        and c["shape"] == (32,)
    ]
    results = []
    for trial in range(args.cold_repeats):
        order = BACKENDS if trial % 2 == 0 else BACKENDS[::-1]
        for case in selected:
            for backend in order:
                cache_dir = temporary / f"{trial}-{backend}-{case['name']}"
                for phase in ("empty", "disk"):
                    print(
                        f"[cold] {trial + 1}/{args.cold_repeats} "
                        f"{case['name']} {backend} {phase}",
                        flush=True,
                    )
                    worker_file = args.output / "cold-worker.json"
                    env = dict(
                        os.environ,
                        CUPY_CACHE_DIR=str(cache_dir / "cupy"),
                        CUDA_CACHE_PATH=str(cache_dir / "cuda"),
                    )
                    (cache_dir / "cuda").mkdir(parents=True, exist_ok=True)
                    completed = subprocess.run(
                        [
                            sys.executable,
                            str(HERE / "main.py"),
                            "--worker",
                            backend,
                            "--case",
                            case["name"],
                            "--worker-result",
                            str(worker_file),
                        ],
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )
                    if completed.returncode:
                        raise RuntimeError(completed.stdout + completed.stderr)
                    row = json.loads(worker_file.read_text())
                    expected = "compile" if phase == "empty" else "disk"
                    if row["first_ir_lookup"]["level"] != expected:
                        raise RuntimeError(f"Expected an {expected} IR lookup")
                    results.append(dict(row, phase=phase, trial=trial))
                    save(args.output / "cold.json", results)
                    worker_file.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("epilog-results"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--warm-only", action="store_true")
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--cold-repeats", type=int, default=4)
    parser.add_argument("--worker", choices=BACKENDS, help=argparse.SUPPRESS)
    parser.add_argument("--case", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.blocks, args.repeats, args.cold_repeats) <= 0:
        parser.error("Repeat counts must be positive")
    if args.worker:
        cold_worker(args)
        return
    if args.smoke:
        args.blocks, args.repeats, args.cold_repeats = 2, 30, 1
    if args.blocks % 2:
        parser.error("--blocks must be even to balance compiler order")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(
        (args.output / name).exists()
        for name in ("comparison.json", "cold.json")
    ):
        parser.error(
            "Choose a fresh output directory to preserve previous results"
        )
    # Do not clear or contaminate a user's existing compiler caches.
    with tempfile.TemporaryDirectory(prefix="fft-epilog-") as temporary:
        temporary = Path(temporary)
        if not args.warm_only:
            cold_run(args, temporary)
        os.environ["CUPY_CACHE_DIR"] = str(temporary / "warm-cupy")
        os.environ["CUDA_CACHE_PATH"] = str(temporary / "warm-cuda")
        (temporary / "warm-cuda").mkdir()
        warm_run(args, cases(args.smoke))
    print(f"Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
