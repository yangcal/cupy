"""Attribute fresh-plan public CuPy FFT calls through the Ajay nvmath adapter.

Each sample starts with an empty, enabled PlanCache. Insertion is timed;
stream drain, eviction and input validation are outside the public-call timer.
Baseline, phase and granular runs all invoke the same public cupy.fft function.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gc
import hashlib
import importlib
import importlib.machinery
import importlib.util
import inspect
import itertools
import json
import logging
import os
from pathlib import Path
import platform
import random
from statistics import median
import sys
from time import perf_counter_ns
import warnings

HERE = Path(__file__).resolve().parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


warm = load(HERE.parent / "fft_nvmath_adapter/main.py", "_cold_warm_helpers")
catalog = load(HERE / "cases.py", "_cold_cases")
profiling = load(HERE / "instrumentation.py", "_cold_instrumentation")
MODES = ("baseline", "phases", "detail")


def phase_totals(records):
    result = defaultdict(int)
    for record in records:
        result[record["phase"]] += record["ns"]
    return dict(result)


def summarize(samples):
    result = {}
    for mode in MODES:
        rows = [s for s in samples if s["mode"] == mode]
        phases = [phase_totals(s["public_records"]) for s in rows]
        names = sorted({name for row in phases for name in row})
        result[mode] = {
            "public_cpu_us": median(s["cpu_seconds"] * 1e6 for s in rows),
            "eviction_cpu_us": median(s["eviction_ns"] / 1000 for s in rows),
            "phases_us": {
                name: median(r.get(name, 0) / 1000 for r in phases)
                for name in names
            },
        }
        if mode != "baseline":
            result[mode]["unattributed_cpu_us"] = median(
                s["cpu_seconds"] * 1e6 - sum(r.values()) / 1000
                for s, r in zip(rows, phases)
            )
    # Absent branches count as zero; also report execution counts.
    details = {}
    for sample in (s for s in samples if s["mode"] == "detail"):
        intervals = defaultdict(int)
        for record in sample["public_records"]:
            for span in record["spans"]:
                intervals[(record["phase"], span["name"], span["line"])] += (
                    span["ns"]
                )
        for record in sample["eviction_records"]:
            for span in record["spans"]:
                intervals[
                    ("eviction." + record["phase"], span["name"], span["line"])
                ] += span["ns"]
        details[sample["block"], sample["repeat"]] = intervals
    keys = sorted(
        {key for row in details.values() for key in row},
        key=lambda k: (k[0], k[2] or 0, k[1]),
    )
    result["boundaries"] = [
        {
            "phase": phase,
            "interval": name,
            "source_line": line,
            "median_us": median(
                r.get((phase, name, line), 0) / 1000 for r in details.values()
            ),
            "executed_samples": sum(
                (phase, name, line) in r for r in details.values()
            ),
        }
        for phase, name, line in keys
    ]
    paired = {}
    lookup = {
        (s["block"], s["repeat"], s["mode"]): s["cpu_seconds"] * 1e6
        for s in samples
    }
    for mode in ("phases", "detail"):
        differences = [
            lookup[b, r, mode] - lookup[b, r, "baseline"]
            for b, r, label in lookup
            if label == "baseline"
        ]
        paired[mode + "_minus_baseline_us"] = median(differences)
    result["observer_effect"] = paired
    return result


def write_tables(path, payload):
    summary_path = path.with_suffix(".summary.csv")
    detail_path = path.with_suffix(".detail.csv")
    with summary_path.open("w") as f:
        fields = [
            "case",
            "public",
            "shape",
            "dtype",
            "layout",
            "baseline_public_us",
            "phase_public_us",
            "init_us",
            "plan_us",
            "execute_us",
            "scaling_us",
            "key_us",
            "axes_us",
            "release_operand_us",
            "unattributed_us",
            "eviction_us",
            "workspace_bytes",
            "phase_observer_us",
            "detail_observer_us",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in payload["cases"]:
            s = c["summary"]
            phases = s["phases"]["phases_us"]
            spec = c["case"]
            writer.writerow(
                {
                    "case": spec["name"],
                    "public": spec["public"],
                    "shape": c["operand_shape"],
                    "dtype": spec["dtype"],
                    "layout": spec["layout"],
                    "baseline_public_us": s["baseline"]["public_cpu_us"],
                    "phase_public_us": s["phases"]["public_cpu_us"],
                    "init_us": phases.get("__init__", 0),
                    "plan_us": phases.get("plan", 0),
                    "execute_us": phases.get("execute", 0),
                    "scaling_us": phases.get("scaling", 0),
                    "key_us": phases.get("key", 0),
                    "axes_us": phases.get("axes", 0),
                    "release_operand_us": phases.get("release_operand", 0),
                    "unattributed_us": s["phases"]["unattributed_cpu_us"],
                    "eviction_us": s["baseline"]["eviction_cpu_us"],
                    "workspace_bytes": c["workspace_bytes"],
                    "phase_observer_us": s["observer_effect"][
                        "phases_minus_baseline_us"
                    ],
                    "detail_observer_us": s["observer_effect"][
                        "detail_minus_baseline_us"
                    ],
                }
            )
    with detail_path.open("w") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "phase",
                "interval",
                "source_line",
                "median_us",
                "executed_samples",
            ],
        )
        writer.writeheader()
        for c in payload["cases"]:
            writer.writerows(
                {"case": c["case"]["name"], **r}
                for r in c["summary"]["boundaries"]
            )


def calibration():
    clock = []
    for _ in range(1000):
        start = perf_counter_ns()
        end = perf_counter_ns()
        clock.append(end - start)
    recorder = profiling.Recorder()
    recorder.begin("calibration")
    for _ in range(1000):
        recorder.mark("calibration", "mark", None)
    spans = recorder.finish()
    return {
        "clock_pair_median_ns": median(clock),
        "boundary_mark_interval_median_ns": median(
            r["ns"] for r in spans[1:-1]
        ),
        "subtracted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=Path("cold-results.json"))
    parser.add_argument("--suite", choices=("basic", "sweep"), default="sweep")
    parser.add_argument("--case", dest="case_filter")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--cpu", type=int)
    parser.add_argument(
        "--retain-workspace",
        action="store_true",
        help="Retain workspace; default uses normal adapter release policy",
    )
    args = parser.parse_args()
    warm_cases = warm.load("cases.py")
    cases = (
        warm_cases.basic()
        if args.suite == "basic"
        else catalog.catalog(warm_cases)
    )
    if args.case_filter:
        cases = [c for c in cases if args.case_filter in c["name"]]
    if not cases:
        parser.error("No matching cases")
    if args.list_cases:
        for c in cases:
            print(
                f"{c['name']}: {c['public']} {c['shape']} "
                f"{c['dtype']} {c['layout']} {c['kwargs']}"
            )
        print(f"{len(cases)} cases")
        return
    if (
        args.blocks < 6
        or args.blocks % 6
        or args.repeats < 1
        or args.warmups < 2
    ):
        parser.error(
            "Require blocks divisible by six, repeats > 0, and warmups >= 2"
        )
    import numpy as np
    import cupy as cp
    import nvmath.fft as nf
    from cupy.cuda import driver
    from cupyx.profiler import benchmark
    from nvmath.internal._strides_helpers import element_strides
    from nvmath.internal.tensor_ifc import _numpy_dtype_name

    adapter = importlib.import_module("cupy.fft._fft_nvmath")
    fft_module = importlib.import_module("cupy.fft._fft")
    fft_source = Path(inspect.getfile(nf.FFT)).resolve()
    warm.check(
        hashlib.sha256(fft_source.read_bytes()).hexdigest()
        == profiling.FFT_SOURCE_SHA,
        "Requires nvmath candidate 84d59bc; inspect source to update anchors",
    )
    warm.check(
        hashlib.sha256(inspect.getsource(nf.FFT.execute).encode()).hexdigest()
        == warm.EXECUTE_SHA,
        "Wrong execute candidate",
    )
    warm.check(
        Path(adapter.__file__).resolve()
        == HERE.parents[1] / "cupy/fft/_fft_nvmath.py",
        "Install and run this CuPy source checkout",
    )
    warm.check(driver._is_cuda_python(), "Requires the CUDA-Python CuPy build")
    warm.check(
        _numpy_dtype_name.cache_parameters()
        == {"maxsize": 256, "typed": False},
        "Requires cached dtype names",
    )
    strides_module = importlib.import_module(element_strides.__module__)
    warm.check(
        any(
            strides_module.__file__.endswith(s)
            for s in importlib.machinery.EXTENSION_SUFFIXES
        ),
        "Requires compiled element strides",
    )
    warm.check(
        not logging.getLogger().isEnabledFor(logging.INFO),
        "Disable nvmath INFO/DEBUG logging",
    )
    affinity = os.sched_getaffinity(0)
    cpu = min(affinity) if args.cpu is None else args.cpu
    warm.check(cpu in affinity, "CPU outside process affinity")
    cache = cp.fft.config.get_plan_cache()
    warm.check(
        cache.is_enabled and cache.get_curr_size() == 0,
        "Requires empty enabled PlanCache in a fresh process",
    )
    warm.check(
        cp.fft.config.devices is None
        and not cp.fft.config.use_multi_gpus
        and cp.fft.config.get_current_callback_manager() is None,
        "Remove multi-GPU and callback contexts",
    )
    from cupy.cuda import cufft

    warm.check(
        cufft.get_current_plan() is None, "Remove explicit FFT plan context"
    )
    settings = (
        cp.fft.config.use_nvmath,
        adapter._RELEASE_WORKSPACE,
        fft_module._try_use_nvmath,
    )
    gc_enabled = gc.isenabled()
    stream = cp.cuda.get_current_stream()
    diagnostics = profiling.Diagnostics(nf.FFT, adapter)
    checkpoint = args.json.with_suffix(".samples.jsonl")
    warm.check(
        not checkpoint.exists() and not args.json.exists(),
        "Choose a new result path; refusing to overwrite",
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    source_dir = args.json.with_suffix(".sources")
    source_dir.mkdir(exist_ok=True)
    for name, meta in diagnostics.source_map.items():
        (source_dir / (name + ".py")).write_text(
            meta["instrumented_source"] + "\n"
        )
    files = {
        "benchmark": __file__,
        "instrumentation": profiling.__file__,
        "cases": catalog.__file__,
        "warm_helpers": warm.__file__,
        "warm_cases": warm_cases.__file__,
        "adapter": adapter.__file__,
        "nvmath_fft": str(fft_source),
        "profiler": inspect.getfile(benchmark),
        "stride_provider": strides_module.__file__,
    }
    for name in (
        "nvmath._internal.workspace",
        "nvmath._internal.cuda_scopes",
        "nvmath._internal.stream.resolution",
        "nvmath._internal.stream._resolver",
        "nvmath.fft._fft_impl",
        "nvmath.internal.tensor_ifc",
        "nvmath.internal.tensor_ifc_cupy",
        "nvmath.internal.tensor_wrapper",
        "nvmath.bindings.cufft",
        "cupy.fft._cache",
        "cupy.cuda.cufft",
    ):
        files[name] = importlib.import_module(name).__file__
    files["public_fft"] = inspect.getfile(cp.fft.fft)
    payload = {
        "status": "running",
        "configuration": vars(args) | {"json": str(args.json), "cpu": cpu},
        "measurement": (
            "fresh-plan public cupy.fft call; Ajay hook; "
            "insertion timed, drained eviction separate"
        ),
        "limitations": [
            "Library/JIT/allocator caches warm; not process startup",
            "Host wall times; GPU event intervals include host gaps",
            "Diagnostics include observer overhead; baseline is unmodified",
            "Sequential source intervals; missing branches count as zero",
            "Unattributed time includes request work and observer bookkeeping",
            "Curated cases have no production frequency weights",
        ],
        "environment": {
            "host": platform.uname()._asdict(),
            "python": sys.version,
            "cupy": cp.__version__,
            "numpy": np.__version__,
            "nvmath": importlib.import_module("nvmath").__version__,
            "device": str(cp.cuda.runtime.getDeviceProperties(0)["name"]),
            "runtime": cp.cuda.runtime.runtimeGetVersion(),
            "driver": cp.cuda.runtime.driverGetVersion(),
        },
        "sources": {name: warm.identity(path) for name, path in files.items()},
        "boundaries": diagnostics.source_map,
        "observer_calibration": calibration(),
        "cases": [],
    }
    checkpoint.touch()
    try:
        os.sched_setaffinity(0, {cpu})
        gc.disable()
        cp.fft.config.use_nvmath = True
        adapter._RELEASE_WORKSPACE = not args.retain_workspace
        fft_module._try_use_nvmath = adapter._try_use_nvmath_ajay
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                category=RuntimeWarning,
                message=r"nvmath-python rejected this FFT.*",
            )
            for index, case in enumerate(cases, 1):
                print(
                    f"[case] {index}/{len(cases)} {case['name']}", flush=True
                )
                kwargs = dict(case["kwargs"])
                if case["numpy_axes"]:
                    kwargs["axes"] = tuple(np.int64(x) for x in kwargs["axes"])
                public = getattr(cp.fft, case["public"])
                axes = warm.transform_axes(case, kwargs, len(case["shape"]))
                inputs = []
                for seed in (101, 202):
                    host = warm.host_input(np, case, kwargs, seed)
                    operand = warm.device_input(cp, host, case["layout"])
                    expected = getattr(np.fft, case["public"])(
                        host.astype(
                            np.complex128
                            if np.iscomplexobj(host)
                            else np.float64
                        ),
                        **kwargs,
                    )
                    inputs.append((operand, host, expected))
                operand, host, expected = inputs[1]

                def verify_cache():
                    entries = list(cache)
                    warm.check(
                        len(entries) == 1,
                        "Expected exactly one cache insertion",
                    )
                    key, node = entries[0]
                    plan = node.plan
                    warm.check(
                        key[:2] == ("nvmath", stream.ptr)
                        and isinstance(plan, adapter.CachedFFT),
                        "Public call fell back from nvmath",
                    )
                    warm.check(
                        plan._operand_released and plan.operand.tensor is None,
                        "Cached plan retains input",
                    )
                    workspace = plan.workspace
                    return {
                        "key": repr(key),
                        "workspace_bytes": workspace.size,
                        "allocated_workspace_bytes": workspace.allocated_size,
                    }

                def invoke(mode, input_value, validate=False):
                    warm.check(
                        cache.get_curr_size() == 0 and cache.is_enabled,
                        "Expected empty enabled cache",
                    )
                    before = warm.counts(cache)
                    with diagnostics.activate(mode):
                        if validate:
                            out = public(input_value, **kwargs)
                            result = None
                        else:
                            result = benchmark(
                                public,
                                args=(input_value,),
                                kwargs=kwargs,
                                name=mode,
                                n_repeat=1,
                                n_warmup=0,
                                devices=(0,),
                            )
                        records = diagnostics.take()
                        metadata = verify_cache()
                        warm.check(
                            warm.counts(cache) == (before[0], before[1] + 1),
                            "Call was not exactly one cold nvmath cache miss",
                        )
                        if mode != "baseline":
                            phases = phase_totals(records)
                            warm.check(
                                all(
                                    phases.get(p, 0) > 0
                                    for p in ("__init__", "plan", "execute")
                                ),
                                "Missing lifecycle phase",
                            )
                        stream.synchronize()
                        start = perf_counter_ns()
                        cache.clear()
                        eviction_ns = perf_counter_ns() - start
                        eviction_records = diagnostics.take()
                        warm.check(
                            cache.get_curr_size() == 0, "Eviction failed"
                        )
                    row = {
                        "public_records": records,
                        "eviction_ns": eviction_ns,
                        "eviction_records": eviction_records,
                        **metadata,
                    }
                    if validate:
                        return out, row
                    row.update(
                        cpu_seconds=float(result.cpu_times[0]),
                        gpu_seconds=result.gpu_times.tolist(),
                    )
                    return row

                accuracy = []
                for mode in MODES:
                    for fresh, values, wanted in inputs:
                        stream.synchronize()
                        out, info = invoke(mode, fresh, validate=True)
                        accuracy.append(
                            {
                                "mode": mode,
                                **warm.reference_error(
                                    np, out.get(), wanted, axes, host.dtype
                                ),
                            }
                        )
                        warm.check(
                            out.data.ptr != fresh.data.ptr,
                            "Out-of-place output aliases input",
                        )
                        np.testing.assert_array_equal(fresh.get(), values)
                        del out
                # Ensure additional warmups still create new plans.
                for _ in range(args.warmups - 2):
                    stream.synchronize()
                    invoke("baseline", operand)
                entry = {
                    "case": case,
                    "operand_shape": operand.shape,
                    "operand_strides": operand.strides,
                    "runtime_kwargs": repr(kwargs),
                    "correctness": True,
                    "reference_error": accuracy,
                    "samples": [],
                }
                payload["cases"].append(entry)
                orders = list(itertools.permutations(MODES)) * (
                    args.blocks // 6
                )
                random.Random(0).shuffle(orders)
                for block, order in enumerate(orders):
                    print(
                        f"[progress] {case['name']} "
                        f"block {block + 1}/{args.blocks}",
                        flush=True,
                    )
                    for repeat in range(args.repeats):
                        for mode in order:
                            stream.synchronize()
                            sample = invoke(mode, operand)
                            sample.update(
                                mode=mode, block=block, repeat=repeat
                            )
                            entry["samples"].append(sample)
                            with checkpoint.open("a") as f:
                                f.write(
                                    json.dumps(
                                        {"case": case["name"], **sample}
                                    )
                                    + "\n"
                                )
                np.testing.assert_array_equal(operand.get(), host)
                entry["workspace_bytes"] = info["workspace_bytes"]
                entry["summary"] = summarize(entry["samples"])
                print(
                    json.dumps(
                        {
                            mode: entry["summary"][mode]["public_cpu_us"]
                            for mode in MODES
                        }
                    ),
                    flush=True,
                )
                warm.save(args.json, payload)
                write_tables(args.json, payload)
        payload["status"] = "complete"
    except BaseException as exc:
        payload.update(status="failed", error=repr(exc))
        raise
    finally:
        stream.synchronize()
        cache.clear()
        (
            cp.fft.config.use_nvmath,
            adapter._RELEASE_WORKSPACE,
            fft_module._try_use_nvmath,
        ) = settings
        os.sched_setaffinity(0, affinity)
        if gc_enabled:
            gc.enable()
        warm.save(args.json, payload)
        write_tables(args.json, payload)


if __name__ == "__main__":
    main()
