# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: MIT

"""Collect public ``cupy.fft`` native/nvmath integration timings.

The hot experiment measures repeated public calls after a verified plan-cache
hit has been established.  The cold experiment keeps the process, CUDA
context, libraries, and allocator warm, but clears the plan cache before each
public call.  Both experiments use the same CuPy build and switch backends
only through ``cupy.fft.config.use_nvmath``.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import importlib.metadata
import json
import pathlib
import platform
import socket
import subprocess
import sys
import warnings
from time import perf_counter_ns as perf


HOT_WARMUP = 20
HOT_ITERS = 300
HOT_ROUNDS = 11
HOT_COMPLETION_ITERS = 30
HOT_COMPLETION_ROUNDS = 5

COLD_WARMUP = 3
COLD_ITERS = 10
COLD_ROUNDS = 9

BACKENDS = ("native", "nvmath")
MODES = ("hot", "cold")


@dataclasses.dataclass(frozen=True)
class Fixture:
    """One supported public FFT call and operand description."""

    name: str
    api: str
    shape: tuple[int, ...]
    dtype: str
    axes: tuple[int, ...]
    description: str
    norm: str | None = None
    layout: str = "c"


def _fixture(
    name: str,
    api: str,
    shape: tuple[int, ...],
    dtype: str,
    axes: tuple[int, ...],
    description: str,
    *,
    norm: str | None = None,
    layout: str = "c",
) -> Fixture:
    return Fixture(
        name,
        api,
        shape,
        dtype,
        axes,
        description,
        norm,
        layout,
    )


# This is a curated performance matrix, not a correctness Cartesian product.
# Every fixture is expected to route through nvmath when that backend is
# selected.  Intentional fallback cases are excluded because they cannot
# answer whether the nvmath backend is competitive.
FIXTURES = {
    item.name: item
    for item in (
        _fixture(
            "c2c_len2_c64",
            "fft",
            (2,),
            "complex64",
            (-1,),
            "minimal 1-D C2C",
        ),
        _fixture(
            "c2c_len16_c64",
            "fft",
            (16,),
            "complex64",
            (-1,),
            "small 1-D C2C",
        ),
        _fixture(
            "c2c_len257_c64",
            "fft",
            (257,),
            "complex64",
            (-1,),
            "prime-length 1-D C2C",
        ),
        _fixture(
            "c2c_len1024_c64",
            "fft",
            (1024,),
            "complex64",
            (-1,),
            "practical power-of-two 1-D C2C",
        ),
        _fixture(
            "c2c_len8192_c64",
            "fft",
            (8192,),
            "complex64",
            (-1,),
            "larger 1-D C2C",
        ),
        _fixture(
            "c2c_len1024_c128_inverse",
            "ifft",
            (1024,),
            "complex128",
            (-1,),
            "double-precision inverse 1-D C2C",
        ),
        _fixture(
            "c2c_batch128_len16",
            "fft",
            (128, 16),
            "complex64",
            (-1,),
            "many tiny batched 1-D transforms",
        ),
        _fixture(
            "c2c_batch64_len1024",
            "fft",
            (64, 1024),
            "complex64",
            (-1,),
            "practical batched 1-D transforms",
        ),
        _fixture(
            "c2c_2d_tiny",
            "fftn",
            (2, 2),
            "complex64",
            (-2, -1),
            "minimal direct 2-D C2C",
        ),
        _fixture(
            "c2c_2d_mixed",
            "fftn",
            (32, 48),
            "complex64",
            (-2, -1),
            "mixed-radix direct 2-D C2C",
        ),
        _fixture(
            "c2c_2d_prime",
            "fftn",
            (31, 47),
            "complex64",
            (-2, -1),
            "prime-shape direct 2-D C2C",
        ),
        _fixture(
            "c2c_2d_practical",
            "fftn",
            (128, 128),
            "complex64",
            (-2, -1),
            "practical direct 2-D C2C",
        ),
        _fixture(
            "c2c_3d_tiny",
            "fftn",
            (2, 2, 2),
            "complex64",
            (-3, -2, -1),
            "minimal direct 3-D C2C",
        ),
        _fixture(
            "c2c_3d_mixed",
            "fftn",
            (8, 16, 24),
            "complex64",
            (-3, -2, -1),
            "mixed-radix direct 3-D C2C",
        ),
        _fixture(
            "c2c_3d_practical",
            "fftn",
            (16, 32, 64),
            "complex64",
            (-3, -2, -1),
            "practical direct 3-D C2C",
        ),
        _fixture(
            "c2c_4d_last2",
            "fftn",
            (8, 16, 32, 64),
            "complex64",
            (-2, -1),
            "two transformed axes with two batch dimensions",
        ),
        _fixture(
            "c2c_2d_fortran",
            "fftn",
            (48, 64),
            "complex64",
            (-2, -1),
            "Fortran-contiguous direct 2-D C2C",
            layout="f",
        ),
        _fixture(
            "c2c_batch_strided_last",
            "fft",
            (64, 256),
            "complex64",
            (-1,),
            "batched 1-D C2C with a strided transformed axis",
            layout="strided_last",
        ),
        _fixture(
            "r2c_len16_f32",
            "rfft",
            (16,),
            "float32",
            (-1,),
            "small 1-D R2C",
        ),
        _fixture(
            "r2c_len1024_f32",
            "rfft",
            (1024,),
            "float32",
            (-1,),
            "practical single-precision 1-D R2C",
        ),
        _fixture(
            "r2c_len1024_f64",
            "rfft",
            (1024,),
            "float64",
            (-1,),
            "practical double-precision 1-D R2C",
        ),
        _fixture(
            "r2c_2d_mixed",
            "rfftn",
            (32, 48),
            "float32",
            (-2, -1),
            "mixed-radix direct 2-D R2C",
        ),
        _fixture(
            "c2r_len9_c64",
            "irfft",
            (9,),
            "complex64",
            (-1,),
            "small 1-D C2R with inferred even result length",
        ),
        _fixture(
            "c2r_len513_c64",
            "irfft",
            (513,),
            "complex64",
            (-1,),
            "practical 1-D C2R with inferred even result length",
        ),
        _fixture(
            "c2r_2d_mixed",
            "irfftn",
            (32, 25),
            "complex64",
            (-2, -1),
            "mixed-radix direct 2-D C2R",
        ),
        _fixture(
            "c2c_len1024_ortho",
            "fft",
            (1024,),
            "complex64",
            (-1,),
            "1-D C2C with orthonormal scaling",
            norm="ortho",
        ),
        _fixture(
            "c2c_2d_forward_norm",
            "fftn",
            (32, 48),
            "complex64",
            (-2, -1),
            "2-D C2C with forward normalization",
            norm="forward",
        ),
    )
}

SMOKE_FIXTURES = (
    "c2c_len16_c64",
    "c2c_2d_mixed",
    "r2c_len16_f32",
    "c2r_len9_c64",
)

_SCRIPT_PATH = pathlib.Path(__file__).resolve()
_REPO_ROOT = _SCRIPT_PATH.parents[2]
DEFAULT_DATA_PATH = _SCRIPT_PATH.with_name("fft_nvmath_integration.json")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _git(*args: str) -> str | None:
    completed = subprocess.run(
        ["git", *args],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cpu_model() -> str | None:
    try:
        lines = pathlib.Path("/proc/cpuinfo").read_text().splitlines()
        for line in lines:
            if line.startswith(("model name", "Hardware")):
                return line.split(":", 1)[1].strip()
    except (OSError, IndexError):
        pass
    return platform.processor() or None


def _json_text(payload: dict) -> str:
    """Pretty-print metadata while keeping raw arrays on one line."""
    head = {key: value for key, value in payload.items() if key != "cases"}
    rows = []
    sample_keys = (
        "host_samples_ns",
        "host_clock_pairs_ns",
        "completion_samples_ns",
        "completion_clock_pairs_ns",
        "event_samples_ns",
    )
    for name, case in payload["cases"].items():
        metadata = {
            key: value for key, value in case.items() if key not in sample_keys
        }
        row = json.dumps(metadata, indent=2).replace("\n", "\n    ")[:-2]
        for key in sample_keys:
            row += f",\n      {json.dumps(key)}: {json.dumps(case[key])}"
        rows.append(f"    {json.dumps(name)}: {row}\n    }}")
    return (
        json.dumps(head, indent=2)[:-2]
        + ',\n  "cases": {\n'
        + ",\n".join(rows)
        + "\n  }\n}\n"
    )


def _make_operand(cp, fixture: Fixture):
    base_shape = list(fixture.shape)
    if fixture.layout == "strided_last":
        base_shape[-1] *= 2

    real_dtype = (
        cp.float64
        if fixture.dtype in {"float64", "complex128"}
        else cp.float32
    )
    size = 1
    for extent in base_shape:
        size *= extent
    index = cp.arange(size, dtype=real_dtype)
    values = (index % 17) / 17
    if fixture.dtype.startswith("complex"):
        values = values + 1j * ((index % 13) / 13)
    operand = values.astype(fixture.dtype).reshape(base_shape)

    if fixture.layout == "f":
        operand = cp.asfortranarray(operand)
    elif fixture.layout == "strided_last":
        operand = operand[..., ::2]
    elif fixture.layout != "c":
        raise ValueError(f"unknown layout: {fixture.layout}")

    if operand.shape != fixture.shape:
        raise RuntimeError(
            f"fixture {fixture.name} produced shape {operand.shape}, "
            f"expected {fixture.shape}"
        )
    return operand


def _call_fixture(cp, fixture: Fixture, operand):
    function = getattr(cp.fft, fixture.api)
    if fixture.api in {"fft", "ifft", "rfft", "irfft"}:
        if len(fixture.axes) != 1:
            raise RuntimeError(f"{fixture.api} requires exactly one axis")
        return function(
            operand,
            axis=fixture.axes[0],
            norm=fixture.norm,
        )
    return function(operand, axes=fixture.axes, norm=fixture.norm)


def _cache_state(cache) -> dict:
    entries = list(cache)
    plans = [node.plan for _, node in entries]
    key_families = [
        "nvmath"
        if isinstance(key, tuple) and key and key[0] == "nvmath"
        else "native"
        for key, _ in entries
    ]
    plan_types = [
        f"{type(plan).__module__}.{type(plan).__qualname__}" for plan in plans
    ]
    labels = [getattr(plan, "_cupy_fft_cache_name", None) for plan in plans]
    nvmath_entries = [
        family == "nvmath" or label == "nvmath FFT"
        for family, label in zip(key_families, labels)
    ]
    return {
        "plan_ids": frozenset(id(plan) for plan in plans),
        "plan_count": len(plans),
        "plan_types": tuple(plan_types),
        "plan_labels": tuple(labels),
        "key_families": tuple(key_families),
        "nvmath_entries": tuple(nvmath_entries),
        "cache_memory_bytes": cache.get_curr_memsize(),
    }


def _public_cache_state(state: dict) -> dict:
    return {
        "plan_count": state["plan_count"],
        "plan_types": list(state["plan_types"]),
        "plan_labels": list(state["plan_labels"]),
        "key_families": list(state["key_families"]),
        "cache_memory_bytes": state["cache_memory_bytes"],
    }


def _assert_route(state: dict, backend: str, fixture: Fixture) -> None:
    if state["plan_count"] == 0:
        raise RuntimeError(
            f"{fixture.name}:{backend} did not create a cached plan"
        )
    routed_to_nvmath = all(state["nvmath_entries"])
    if backend == "nvmath" and not routed_to_nvmath:
        raise RuntimeError(
            f"{fixture.name} silently fell back to native CuPy: "
            f"{_public_cache_state(state)}"
        )
    if backend == "native" and any(state["nvmath_entries"]):
        raise RuntimeError(
            f"{fixture.name} unexpectedly used nvmath while disabled: "
            f"{_public_cache_state(state)}"
        )


def _assert_hot_plan_unchanged(
    cache,
    initial_state: dict,
    backend: str,
    fixture: Fixture,
) -> None:
    current = _cache_state(cache)
    _assert_route(current, backend, fixture)
    if current["plan_ids"] != initial_state["plan_ids"]:
        raise RuntimeError(
            f"{fixture.name}:{backend} replaced its plans during hot sampling"
        )


def _clock_pair() -> int:
    start = perf()
    stop = perf()
    return stop - start


def _event_elapsed_ns(cp, start_event, stop_event) -> int:
    milliseconds = cp.cuda.get_elapsed_time(start_event, stop_event)
    return round(milliseconds * 1_000_000)


def _prepare_hot_case(
    cp,
    fixture: Fixture,
    backend: str,
    operand,
    cache,
    stream,
    warmup: int,
) -> tuple[dict, dict]:
    cp.fft.config.use_nvmath = backend == "nvmath"
    stream.synchronize()
    cache.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = _call_fixture(cp, fixture, operand)
    stream.synchronize()
    state = _cache_state(cache)
    _assert_route(state, backend, fixture)
    result_metadata = {
        "shape": list(result.shape),
        "dtype": result.dtype.name,
        "strides": list(result.strides),
        "c_contiguous": bool(result.flags.c_contiguous),
        "f_contiguous": bool(result.flags.f_contiguous),
    }
    del result

    for _ in range(warmup):
        result = _call_fixture(cp, fixture, operand)
        del result
    stream.synchronize()
    _assert_hot_plan_unchanged(cache, state, backend, fixture)
    return state, result_metadata


def _collect_hot_case(
    cp,
    fixture: Fixture,
    backend: str,
    operand,
    cache,
    stream,
    args,
) -> dict:
    state, result_metadata = _prepare_hot_case(
        cp,
        fixture,
        backend,
        operand,
        cache,
        stream,
        args.hot_warmup,
    )

    def call():
        return _call_fixture(cp, fixture, operand)

    host_samples = []
    host_clock_pairs = []
    gc.collect()
    gc.disable()
    try:
        for round_index in range(args.hot_rounds):
            for _ in range(args.hot_iters):
                host_clock_pairs.append(_clock_pair())
                start = perf()
                result = call()
                stop = perf()
                host_samples.append(stop - start)
                del result
            stream.synchronize()
            _assert_hot_plan_unchanged(cache, state, backend, fixture)
            print(
                f"hot:{fixture.name}:{backend}: host round "
                f"{round_index + 1}/{args.hot_rounds}",
                flush=True,
            )
    finally:
        gc.enable()

    completion_samples = []
    completion_clock_pairs = []
    event_samples = []
    start_event = cp.cuda.Event()
    stop_event = cp.cuda.Event()
    gc.collect()
    gc.disable()
    try:
        for round_index in range(args.hot_completion_rounds):
            for _ in range(args.hot_completion_iters):
                stream.synchronize()
                completion_clock_pairs.append(_clock_pair())
                start_event.record(stream)
                start = perf()
                result = call()
                stop_event.record(stream)
                stream.synchronize()
                stop = perf()
                completion_samples.append(stop - start)
                event_samples.append(
                    _event_elapsed_ns(cp, start_event, stop_event)
                )
                del result
            _assert_hot_plan_unchanged(cache, state, backend, fixture)
            print(
                f"hot:{fixture.name}:{backend}: completion round "
                f"{round_index + 1}/{args.hot_completion_rounds}",
                flush=True,
            )
    finally:
        gc.enable()

    return {
        "mode": "hot",
        "fixture": fixture.name,
        "backend": backend,
        "observed_backend": backend,
        "cache": _public_cache_state(state),
        "result": result_metadata,
        "host_samples_ns": host_samples,
        "host_clock_pairs_ns": host_clock_pairs,
        "completion_samples_ns": completion_samples,
        "completion_clock_pairs_ns": completion_clock_pairs,
        "event_samples_ns": event_samples,
    }


def _collect_cold_case(
    cp,
    fixture: Fixture,
    backend: str,
    operand,
    cache,
    stream,
    args,
) -> dict:
    cp.fft.config.use_nvmath = backend == "nvmath"

    def call():
        return _call_fixture(cp, fixture, operand)

    result_metadata = None
    expected_cache = None
    for _ in range(args.cold_warmup):
        stream.synchronize()
        cache.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = call()
        stream.synchronize()
        state = _cache_state(cache)
        _assert_route(state, backend, fixture)
        public_state = _public_cache_state(state)
        if expected_cache is None:
            expected_cache = public_state
            result_metadata = {
                "shape": list(result.shape),
                "dtype": result.dtype.name,
                "strides": list(result.strides),
                "c_contiguous": bool(result.flags.c_contiguous),
                "f_contiguous": bool(result.flags.f_contiguous),
            }
        elif public_state != expected_cache:
            raise RuntimeError(
                f"unstable cold cache state for {fixture.name}:{backend}"
            )
        del result

    host_samples = []
    completion_samples = []
    clock_pairs = []
    event_samples = []
    start_event = cp.cuda.Event()
    stop_event = cp.cuda.Event()
    warning_context = warnings.catch_warnings()
    warning_context.__enter__()
    warnings.simplefilter("error", RuntimeWarning)
    gc.collect()
    gc.disable()
    try:
        for round_index in range(args.cold_rounds):
            for _ in range(args.cold_iters):
                stream.synchronize()
                cache.clear()
                if cache.get_curr_size() != 0:
                    raise RuntimeError("plan cache did not clear")

                clock_pairs.append(_clock_pair())
                start_event.record(stream)
                start = perf()
                result = call()
                host_stop = perf()
                stop_event.record(stream)
                stream.synchronize()
                completion_stop = perf()

                state = _cache_state(cache)
                _assert_route(state, backend, fixture)
                public_state = _public_cache_state(state)
                if public_state != expected_cache:
                    raise RuntimeError(
                        f"unstable cold cache state for "
                        f"{fixture.name}:{backend}"
                    )
                host_samples.append(host_stop - start)
                completion_samples.append(completion_stop - start)
                event_samples.append(
                    _event_elapsed_ns(cp, start_event, stop_event)
                )
                del result
            print(
                f"cold:{fixture.name}:{backend}: round "
                f"{round_index + 1}/{args.cold_rounds}",
                flush=True,
            )
    finally:
        gc.enable()
        warning_context.__exit__(None, None, None)
        cache.clear()

    if expected_cache is None or result_metadata is None:
        raise RuntimeError("cold warmup did not produce metadata")
    return {
        "mode": "cold",
        "fixture": fixture.name,
        "backend": backend,
        "observed_backend": backend,
        "cache": expected_cache,
        "result": result_metadata,
        "host_samples_ns": host_samples,
        "host_clock_pairs_ns": clock_pairs,
        "completion_samples_ns": completion_samples,
        "completion_clock_pairs_ns": clock_pairs,
        "event_samples_ns": event_samples,
    }


def _check_correctness(cp, fixture: Fixture, operand, cache, stream) -> None:
    outputs = {}
    states = {}
    for backend in BACKENDS:
        cp.fft.config.use_nvmath = backend == "nvmath"
        stream.synchronize()
        cache.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            outputs[backend] = _call_fixture(cp, fixture, operand)
        stream.synchronize()
        states[backend] = _cache_state(cache)
        _assert_route(states[backend], backend, fixture)

    native = outputs["native"]
    candidate = outputs["nvmath"]
    if candidate.shape != native.shape or candidate.dtype != native.dtype:
        raise AssertionError(
            f"{fixture.name} result metadata differs: "
            f"native={native.shape}/{native.dtype}, "
            f"nvmath={candidate.shape}/{candidate.dtype}"
        )
    if fixture.dtype in {"float64", "complex128"}:
        rtol = atol = 1e-11
    else:
        rtol = atol = 5e-5
    cp.testing.assert_allclose(candidate, native, rtol=rtol, atol=atol)
    del candidate, native
    stream.synchronize()
    cache.clear()


def _warm_process(cp, cache, stream) -> None:
    seed = cp.arange(16, dtype=cp.float32).astype(cp.complex64)
    fixture = FIXTURES["c2c_len16_c64"]
    for backend in BACKENDS:
        cp.fft.config.use_nvmath = backend == "nvmath"
        cache.clear()
        result = _call_fixture(cp, fixture, seed)
        stream.synchronize()
        state = _cache_state(cache)
        _assert_route(state, backend, fixture)
        del result
    cache.clear()


def _selected_names(args) -> list[str]:
    if args.fixture:
        return list(dict.fromkeys(args.fixture))
    if args.suite == "smoke":
        return list(SMOKE_FIXTURES)
    return list(FIXTURES)


def _collect(args) -> dict:
    import cupy as cp

    try:
        import nvmath
    except ImportError as error:
        raise RuntimeError(
            "nvmath-python must be installed in the CuPy environment"
        ) from error

    if not hasattr(cp.fft.config, "use_nvmath"):
        raise RuntimeError(
            "This CuPy build does not contain Yang's nvmath FFT integration"
        )

    selected_modes = list(MODES) if args.mode == "all" else [args.mode]
    selected_backends = (
        list(BACKENDS) if args.backend == "all" else [args.backend]
    )
    selected_fixtures = _selected_names(args)
    stream = cp.cuda.get_current_stream()
    device_id = cp.cuda.runtime.getDevice()
    cache = cp.fft.config.get_plan_cache()

    old_use_nvmath = cp.fft.config.use_nvmath
    old_cache_size = cache.get_size()
    old_cache_memsize = cache.get_memsize()
    cache.clear()
    cache.set_size(-1)
    cache.set_memsize(-1)

    cases = {}
    operands = {}
    try:
        _warm_process(cp, cache, stream)
        for name in selected_fixtures:
            fixture = FIXTURES[name]
            operand = _make_operand(cp, fixture)
            operands[name] = operand
            _check_correctness(cp, fixture, operand, cache, stream)

        collectors = {
            "hot": _collect_hot_case,
            "cold": _collect_cold_case,
        }
        for mode in selected_modes:
            for fixture_name in selected_fixtures:
                fixture = FIXTURES[fixture_name]
                operand = operands[fixture_name]
                for backend in selected_backends:
                    case_name = f"{mode}:{fixture_name}:{backend}"
                    print(f"collecting {case_name}", flush=True)
                    cases[case_name] = collectors[mode](
                        cp,
                        fixture,
                        backend,
                        operand,
                        cache,
                        stream,
                        args,
                    )
                    cache.clear()
    finally:
        stream.synchronize()
        cache.clear()
        cache.set_size(old_cache_size)
        cache.set_memsize(old_cache_memsize)
        cp.fft.config.use_nvmath = old_use_nvmath

    if cp.cuda.runtime.getDevice() != device_id:
        raise RuntimeError("the current CUDA device changed during collection")

    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    device_name = properties["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode()

    status = _git("status", "--porcelain")
    return {
        "schema_version": 1,
        "benchmark": "CuPy public FFT native/nvmath integration latency",
        "scope": {
            "hot": (
                "verified cache hits through the complete public cupy.fft "
                "call; process and plan warm"
            ),
            "cold": (
                "complete public cupy.fft call; process warm; plan cache "
                "cleared before every sample"
            ),
        },
        "mode_order": selected_modes,
        "backend_order": selected_backends,
        "fixture_order": selected_fixtures,
        "fixtures": {
            name: dataclasses.asdict(FIXTURES[name])
            for name in selected_fixtures
        },
        "sampling": {
            "hot": {
                "warmup": args.hot_warmup,
                "host_iters": args.hot_iters,
                "host_rounds": args.hot_rounds,
                "completion_iters": args.hot_completion_iters,
                "completion_rounds": args.hot_completion_rounds,
            },
            "cold": {
                "warmup": args.cold_warmup,
                "iters": args.cold_iters,
                "rounds": args.cold_rounds,
            },
        },
        "methodology": {
            "backend_switch": "cupy.fft.config.use_nvmath",
            "routing_check": (
                "cache entry family and plan type checked outside timed "
                "intervals; silent fallbacks fail collection"
            ),
            "host_clock": "time.perf_counter_ns",
            "cuda_event_span": (
                "CUDA events bracket the public call; cold spans include "
                "device-idle time while host planning runs and are not "
                "pure kernel execution time"
            ),
            "gc": "disabled while collecting samples",
            "cache": (
                "same CuPy PlanCache; cleared between backends and before "
                "every cold sample"
            ),
        },
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu": _cpu_model(),
            "python": sys.version.split()[0],
            "cupy": cp.__version__,
            "cupy_file": cp.__file__,
            "nvmath_python": _distribution_version("nvmath-python"),
            "nvmath_file": nvmath.__file__,
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
            "cuda_driver": cp.cuda.runtime.driverGetVersion(),
            "device_id": device_id,
            "device_name": device_name,
        },
        "repository": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "status": status,
            "dirty": bool(status),
        },
        "cases": cases,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", *MODES), default="all")
    parser.add_argument("--backend", choices=("all", *BACKENDS), default="all")
    parser.add_argument("--suite", choices=("smoke", "full"), default="full")
    parser.add_argument(
        "--fixture",
        action="append",
        choices=tuple(FIXTURES),
        help="repeat to select fixtures explicitly; overrides --suite",
    )
    parser.add_argument("--hot-warmup", type=_positive_int, default=HOT_WARMUP)
    parser.add_argument("--hot-iters", type=_positive_int, default=HOT_ITERS)
    parser.add_argument("--hot-rounds", type=_positive_int, default=HOT_ROUNDS)
    parser.add_argument(
        "--hot-completion-iters",
        type=_positive_int,
        default=HOT_COMPLETION_ITERS,
    )
    parser.add_argument(
        "--hot-completion-rounds",
        type=_positive_int,
        default=HOT_COMPLETION_ROUNDS,
    )
    parser.add_argument(
        "--cold-warmup", type=_positive_int, default=COLD_WARMUP
    )
    parser.add_argument("--cold-iters", type=_positive_int, default=COLD_ITERS)
    parser.add_argument(
        "--cold-rounds", type=_positive_int, default=COLD_ROUNDS
    )
    parser.add_argument("--json", type=pathlib.Path, default=DEFAULT_DATA_PATH)
    return parser


def main() -> None:
    args = _parser().parse_args()
    payload = _collect(args)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(_json_text(payload))
    print(f"Raw samples: {args.json}")


if __name__ == "__main__":
    main()
