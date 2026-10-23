"""Measure warm CuPy FFT paths and bound nvmath execution across operands.

The sweep measures CPU host latency and GPU event intervals, with the exact
minimal cached lifecycle
included only for comparable C2C default-forward cases. GPU event intervals
include host submission gaps. This is coverage, not a workload frequency model.
Original uses the reference adapter. Axes adds common-axis normalization and
no-op scaling skips. Fused adds runtime-divisor NVRTC fusion to Axes; C2R adds
prepared operand reset to Axes. Ajay combines both. Native uses CuPy's existing
backend.
Use --case irfft --compare-c2r-reset --blocks 24 to isolate prepared C2R
reset against the same fused adapter using the original operand reset.
Use --compare-fusion --blocks 24 to compare separate CuPy scaling with
NVRTC fusion while holding the Ajay request/reset path fixed.
Install this checkout with prepared copying in nvmath reset_operand_unchecked;
its execute implementation remains pinned to candidate 84d59bc.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import inspect
import itertools
import json
import logging
import math
import os
import platform
import random
import re
import sys
import time
import warnings
from pathlib import Path
from statistics import median

EXECUTE_SHA = (
    "6c69df10b7762d6e04ad70b24c12857dcd7cbd360fe2ae2f8805a0eae6630779"
)
MINIMAL_SHA = (
    "f436d4536041b066494a2d8ba487ef192514d9159d1d2023c3331245cc7a4cf1"
)
REFERENCE_RESET_SHA = (
    "546f6caa821abf6a0a14a3bbb2e795e0421335a9c7aad5d0cf03fda543f99a19"
)
PAIRS = {
    "original_minus_minimal": ("original", "minimal"),
    "ajay_minus_minimal": ("ajay", "minimal"),
    "original_minus_ajay": ("original", "ajay"),
    "original_minus_axes": ("original", "axes"),
    "axes_minus_ajay": ("axes", "ajay"),
    "axes_minus_fused": ("axes", "fused"),
    "axes_minus_c2r": ("axes", "c2r"),
    "axes_minus_native": ("axes", "native"),
    "axes_minus_minimal": ("axes", "minimal"),
    "c2r_minus_ajay": ("c2r", "ajay"),
    "c2r_minus_native": ("c2r", "native"),
    "original_minus_native": ("original", "native"),
    "ajay_minus_native": ("ajay", "native"),
    "fused_minus_ajay": ("fused", "ajay"),
    "fused_minus_native": ("fused", "native"),
    "unfused_minus_ajay": ("unfused", "ajay"),
    "minimal_minus_native": ("minimal", "native"),
    "minimal_minus_stateful": ("minimal", "stateful"),
    "ajay_minus_stateful": ("ajay", "stateful"),
    "native_minus_stateful": ("native", "stateful"),
}


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def identity(path):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def check_source_copy(imported, expected):
    """Accept editable or regular installs of the exact benchmark candidate."""
    imported, expected = Path(imported).resolve(), Path(expected).resolve()
    check(
        imported.is_file() and expected.is_file()
        and imported.read_bytes() == expected.read_bytes(),
        "Installed CuPy source differs from the benchmark candidate. "
        f"Imported: {imported}; expected matching contents: {expected}",
    )


def installed_version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def cpu_model():
    """Report the processor model for host-latency comparisons."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            field, separator, value = line.partition(":")
            if separator and field.strip().lower() in (
                "model name", "hardware"
            ):
                return value.strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def prepare_call(function, operand, kwargs, adapter, nf):
    """Record miss preparation; remove all wrappers before warm timing."""
    elapsed = {name: 0 for name in ("construct", "epilog_prepare", "plan")}
    missing, previous = object(), []

    def measured(name, target):
        def call(*args, **keywords):
            start = time.perf_counter_ns()
            try:
                return target(*args, **keywords)
            finally:
                elapsed[name] += time.perf_counter_ns() - start
        return call

    try:
        for owner, attribute, phase in (
            (adapter.CachedFFT, "__init__", "construct"),
            (adapter, "_normalization_epilog", "epilog_prepare"),
            (adapter.CachedFFT, "plan", "plan"),
        ):
            previous.append(
                (owner, attribute, vars(owner).get(attribute, missing)))
            setattr(owner, attribute,
                    measured(phase, getattr(owner, attribute)))
        output = function(operand, **kwargs)
    finally:
        for owner, attribute, original in reversed(previous):
            if original is missing:
                delattr(owner, attribute)
            else:
                setattr(owner, attribute, original)
    return output, {name + "_cpu_us": value / 1000
                    for name, value in elapsed.items()}


def load(filename):
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(
        "_fft_benchmark_" + path.stem, path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def select_reset(fft_class, method, plans=()):
    """Select a row's reset outside timing, without changing nvmath's class."""
    fft_class.reset_operand_unchecked = method
    for plan in plans:
        if getattr(plan, "fft_abstract_type", None) == "C2R":
            # Shared plans can retain a recipe from the preceding Ajay row.
            # This diagnostic describes the next row's reset, not that recipe.
            plan._c2r_reset_prepared_last_used = False


def counts(cache):
    match = re.search(r"hits / misses: (\d+) / (\d+)", repr(cache))
    check(match is not None, "Cannot read actual PlanCache counters")
    return tuple(map(int, match.groups()))


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def row_orders(labels, blocks):
    rng, orders = random.Random(0), []
    if len(labels) >= 6 and len(labels) % 2 == 0:
        # An even-sized Williams design balances positions and every ordered
        # adjacent pair over n blocks. Six and eight rows both fit 24 blocks.
        size = len(labels)
        base = [0]
        for index in range(1, size // 2):
            base.extend((index, size - index))
        base.append(size // 2)
        permutations = [
            tuple(labels[(x + offset) % size] for x in base)
            for offset in range(size)
        ]
    elif len(labels) == 5:
        # Williams design: each row occupies each position twice, and every
        # ordered adjacent pair occurs twice over ten blocks.
        base = (0, 1, 4, 2, 3)
        permutations = [
            tuple(labels[(x + offset) % 5] for x in order)
            for offset in range(5) for order in (base, base[::-1])
        ]
    else:
        permutations = list(itertools.permutations(labels))
    check(blocks >= len(permutations) and blocks % len(permutations) == 0,
          f"Require blocks to be a positive multiple of {len(permutations)}")
    for _ in range(blocks // len(permutations)):
        cycle = permutations.copy()
        rng.shuffle(cycle)
        orders.extend(cycle)
    return orders


def host_input(np, spec, kwargs, seed):
    dtype, shape = np.dtype(spec["dtype"]), spec["shape"]
    if spec["arange"]:
        host = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
        host = host.astype(dtype)
        return host + np.complex64(1 - 0.5j) if seed == 101 else host
    double = dtype.name in ("float64", "complex128")
    real_dtype = np.float64 if double else np.float32
    rng = np.random.default_rng(seed)
    real = rng.uniform(-0.5, 0.5, shape).astype(real_dtype)
    if spec["public"].startswith("irfft"):
        if spec["public"] == "irfft":
            axes = (kwargs.get("axis", -1),)
        elif spec["public"] == "irfft2":
            axes = kwargs.get("axes", (-2, -1))
        else:
            axes = kwargs.get("axes")
        host = np.fft.rfftn(real, axes=axes, norm="backward").astype(dtype)
    elif dtype.kind == "c":
        imag = rng.uniform(-0.5, 0.5, shape).astype(real_dtype)
        host = (real + 1j * imag).astype(dtype)
    else:
        host = real.astype(dtype)
    return np.asfortranarray(host) if spec["layout"] == "F" else host


def device_input(cp, host, layout):
    if layout != "strided":
        return cp.array(host, order=layout, copy=True)
    backing = cp.empty(
        (*host.shape[:-1], 2 * host.shape[-1]), dtype=host.dtype
    )
    operand = backing[..., ::2]
    operand[...] = cp.asarray(host)
    return operand


def transform_axes(spec, kwargs, ndim):
    if spec["public"] in ("fft", "ifft", "rfft", "irfft"):
        axes = (kwargs.get("axis", -1),)
    elif spec["public"].endswith("2"):
        axes = kwargs.get("axes", (-2, -1))
    else:
        axes = kwargs.get("axes", tuple(range(ndim)))
    if axes is None:
        axes = tuple(range(ndim))
    return tuple(int(axis) % ndim for axis in axes)


def row_labels(spec, *, compare_c2r_reset=False, compare_fusion=False):
    if compare_fusion:
        return ("original", "unfused", "ajay", "native")
    if compare_c2r_reset:
        return ("original", "fused", "ajay", "native")
    return (("original", "axes", "fused", "c2r", "ajay", "minimal",
             "native", "stateful") if spec["minimal"] else
            ("original", "axes", "fused", "c2r", "ajay", "native"))


def callback_free_key(adapter, operand, spec, kwargs, stream_ptr):
    """Describe Original's plan without deriving it from a callback key."""
    public = spec["public"]
    kind = ("C2R" if public.startswith("irfft") else
            "R2C" if public.startswith("rfft") else "C2C")
    axes = transform_axes(spec, kwargs, operand.ndim)
    options = adapter._FFT_OPTIONS[kind]
    try:
        key = adapter._create_key_from_operand(operand, axes, options)
    except adapter.nvmath_fft.UnsupportedLayoutError as error:
        key = adapter._create_key_from_operand(
            operand, tuple(error.axes), options, tuple(error.permutation))
    return ("nvmath", stream_ptr, key)


def normalization_metadata(np, spec, kwargs, operand_shape, output_shape,
                           output_dtype):
    """Record effective scaling, including C2R's full real transform size."""
    norm = kwargs.get("norm")
    effective = "backward" if norm is None else norm
    inverse = spec["public"].startswith("i")
    axes = transform_axes(spec, kwargs, len(operand_shape))
    shape = (output_shape if spec["public"].startswith("irfft")
             else operand_shape)
    size = math.prod(shape[axis] for axis in axes)
    divisor = (math.sqrt(size) if effective == "ortho" else size
               if (effective == "backward" and inverse
                   or effective == "forward" and not inverse) else 1)
    real_dtype = np.finfo(output_dtype).dtype
    return {
        "requested_norm": norm,
        "effective_norm": effective,
        "direction": "inverse" if inverse else "forward",
        "transformed_points": size,
        "divisor": divisor,
        "divisor_dtype": real_dtype.name,
        "typed_divisor": float(real_dtype.type(divisor)),
        "scaling_applies": divisor != 1,
    }


def fusion_report(payload):
    """Render same-run fusion contrasts separately from historical results."""
    cases = sorted(payload["cases"], key=lambda case: (
        not case["normalization"]["scaling_applies"],
        -case["summary"]["unfused_minus_ajay"], case["case_id"]))
    lines = [
        "# Isolated normalization fusion", "",
        "Unfused and Fused use the same Ajay request path, nvmath reset, "
        "metadata helpers and workspace policy. Unfused scales through a "
        "separate CuPy division; Fused uses an NVRTC store callback.", "",
        "Times are median CPU or CUDA-event µs. Saved is the median paired "
        "block difference, Unfused minus Fused. GPU intervals include host "
        "submission gaps. Scaled requests precede unscaled controls.", "",
        "| ID | Request | Operand | Effective norm | Divisor | Unfused CPU | "
        "Fused CPU | Saved CPU | Unfused GPU | Fused GPU | Saved GPU |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: |",
    ]
    for case in cases:
        spec, normal, summary = (case["case"], case["normalization"],
                                  case["summary"])
        arguments = ", ".join(f"{key}={value!r}"
                              for key, value in spec["kwargs"].items())
        request = f"{spec['public']}(a{', ' if arguments else ''}{arguments})"
        norm = ("None→backward" if normal["requested_norm"] is None
                else normal["effective_norm"])
        operand = (f"{tuple(case['operand_shape'])} {spec['dtype']} "
                   f"{spec['layout']}")
        cpu, gpu = summary["rows_cpu_us"], summary["rows_gpu_us"]
        lines.append(
            f"| {case['case_id']} | `{request}` | `{operand}` | {norm} | "
            f"{normal['typed_divisor']:.9g} | {cpu['unfused']:.3f} | "
            f"{cpu['ajay']:.3f} | {summary['unfused_minus_ajay']:.3f} | "
            f"{gpu['unfused']:.3f} | {gpu['ajay']:.3f} | "
            f"{summary['paired_gpu_us']['unfused_minus_ajay']:.3f} |")
    return "\n".join(lines) + "\n"


def reference_error(np, actual, wanted, axes, input_dtype):
    """Apply nvmath's 100-epsilon L2 criterion per transformed batch.

    The reference uses double precision, but the tolerance follows the
    actual operand's precision. Zero references use an absolute L2 bound.
    See tests/nvmath_tests/fft/test_fft_with_hypothesis.py::verify_result
    and utils/check_helpers.py::assert_norm_close in the nvmath candidate.
    """
    real_dtype = np.empty((), dtype=input_dtype).real.dtype
    output_dtype = (
        np.result_type(real_dtype, np.complex64)
        if np.iscomplexobj(wanted) else real_dtype
    )
    check(wanted.dtype.name in ("float64", "complex128"),
          "Reference must use double precision")
    check(actual.shape == wanted.shape, "Incorrect output shape")
    check(actual.dtype == output_dtype, "Incorrect output dtype")
    check(np.isfinite(actual).all() and np.isfinite(wanted).all(),
          "Nonfinite FFT output or reference")
    difference = np.abs(actual - wanted)
    error = np.sqrt(np.sum(difference ** 2, axis=axes, dtype=np.float64))
    magnitude = np.sqrt(np.sum(np.abs(wanted) ** 2, axis=axes,
                               dtype=np.float64))
    relative = np.divide(error, magnitude, out=np.zeros_like(error),
                         where=magnitude != 0)
    zero_error = np.where(magnitude == 0, error, 0)
    bound = float(100 * np.finfo(real_dtype).eps)
    worst_relative = float(np.max(relative))
    worst_zero = float(np.max(zero_error))
    check(worst_relative <= bound and worst_zero <= bound,
          f"FFT norm error: relative L2={worst_relative}, "
          f"zero-reference absolute L2={worst_zero}, limit={bound}")
    return {"max_batch_relative_l2": worst_relative,
            "max_zero_reference_l2": worst_zero,
            "max_absolute_error": float(np.max(difference)),
            "l2_limit": bound}


def summarize(blocks, labels):
    rows = {
        label: median(b["rows"][label]["cpu_median_us"] for b in blocks)
        for label in labels
    }
    pairs = {
        name: (a, b)
        for name, (a, b) in PAIRS.items()
        if a in labels and b in labels
    }
    values = {
        name: [b["paired_cpu_us"][name] for b in blocks] for name in pairs
    }
    result = {
        "rows_cpu_us": rows,
        "rows_gpu_us": {
            label: median(b["rows"][label]["gpu_median_us"] for b in blocks)
            for label in labels
        },
        "paired_gpu_us": {
            name: median(b["paired_gpu_us"][name] for b in blocks)
            for name in pairs
        },
        **{name: median(v) for name, v in values.items()},
    }
    result["sign_counts"] = {
        name: {
            "positive": sum(x > 0 for x in v),
            "negative": sum(x < 0 for x in v),
            "zero": sum(x == 0 for x in v),
        }
        for name, v in values.items()
    }
    delta = result["original_minus_ajay"]
    percent = 100 * delta / rows["original"]
    result["original_minus_ajay_percent"] = percent
    result["negative_median"] = delta < 0
    slower = result["sign_counts"]["original_minus_ajay"]["negative"]
    result["material_regression_candidate"] = (
        delta <= -0.2
        and percent <= -2
        and slower >= math.ceil(len(blocks) * 10 / 12)
    )
    if "fused" in labels:
        delta = result["fused_minus_ajay"]
        percent = 100 * delta / rows["fused"]
        result["fused_minus_ajay_percent"] = percent
        result["fused_negative_median"] = delta < 0
        result["fused_material_regression_candidate"] = (
            delta <= -0.2 and percent <= -2
            and result["sign_counts"]["fused_minus_ajay"]["negative"]
            >= math.ceil(len(blocks) * 10 / 12)
        )
    if "unfused" in labels:
        delta = result["unfused_minus_ajay"]
        percent = 100 * delta / rows["unfused"]
        result["unfused_minus_ajay_percent"] = percent
        result["fusion_negative_median"] = delta < 0
        result["fusion_material_regression_candidate"] = (
            delta <= -0.2 and percent <= -2
            and result["sign_counts"]["unfused_minus_ajay"]["negative"]
            >= math.ceil(len(blocks) * 10 / 12)
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("basic", "sweep"), default="sweep")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument(
        "--case", dest="case_filter", help="Case-name substring"
    )
    parser.add_argument(
        "--json", type=Path, default=Path("adapter-overhead.json")
    )
    parser.add_argument(
        "--cpu", type=int, help="Allowed CPU; lowest by default"
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--blocks", type=int, default=24,
        help="Positive multiple of 24 for balanced row orders",
    )
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--release-workspace", action="store_true")
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument(
        "--compare-c2r-reset", action="store_true",
        help="C2R cases only: compare the fused adapter with original versus "
             "prepared operand reset; requires blocks divisible by 24",
    )
    comparison.add_argument(
        "--compare-fusion", action="store_true",
        help="Compare separate CuPy scaling with NVRTC fusion using the "
             "same Ajay request/reset path; requires blocks divisible by 24",
    )
    args = parser.parse_args()
    catalog = load("cases.py")
    cases = getattr(catalog, args.suite)()
    case_ids = {spec["name"]: index for index, spec in enumerate(cases, 1)}
    if args.case_filter:
        cases = [s for s in cases if args.case_filter in s["name"]]
    if not cases:
        parser.error("No cases match --case")
    if args.compare_c2r_reset and not all(
        spec["public"].startswith("irfft") for spec in cases
    ):
        parser.error("--compare-c2r-reset requires only C2R cases; "
                     "use --case irfft")
    if args.list_cases:
        for spec in cases:
            rows = "/".join(row_labels(
                spec, compare_c2r_reset=args.compare_c2r_reset,
                compare_fusion=args.compare_fusion))
            print(
                f"{spec['name']}: {spec['public']} {spec['shape']} "
                f"{spec['dtype']} {spec['layout']} {spec['kwargs']} "
                f"numpy_axes={spec['numpy_axes']}; {rows}"
            )
        print(f"{len(cases)} cases")
        return
    permutation_count = 24
    if (
        args.blocks < permutation_count
        or args.blocks % permutation_count
        or args.repeats < 1
        or args.warmups < 0
        or args.device < 0
    ):
        parser.error(
            f"Require blocks >= {permutation_count} and divisible by it, "
            "repeats >=1, "
            "warmups >=0, device >=0"
        )
    affinity = os.sched_getaffinity(0)
    cpu = min(affinity) if args.cpu is None else args.cpu
    if cpu not in affinity:
        parser.error(f"CPU {cpu} is outside {sorted(affinity)}")

    # Help and case listing never import NumPy, CuPy, or nvmath.
    import numpy as np
    import nvmath.fft as nf
    from nvmath.internal._strides_helpers import element_strides
    from nvmath.internal.tensor_ifc import _numpy_dtype_name

    import cupy as cp
    from cupy.cuda import cufft, driver
    from cupyx.profiler import benchmark

    adapter = importlib.import_module("cupy.fft._fft_nvmath")
    expected_adapter = (
        Path(__file__).resolve().parents[2] / "cupy" / "fft" / "_fft_nvmath.py"
    )
    check_source_copy(adapter.__file__, expected_adapter)
    minimal = load("minimal_lifecycle.py")
    execute = hashlib.sha256(
        inspect.getsource(nf.FFT.execute).encode()
    ).hexdigest()
    check(
        execute == EXECUTE_SHA, "Requires the exact nvmath candidate execute"
    )
    minimal_sha = hashlib.sha256(
        inspect.getsource(minimal.execute_cached).encode()
    ).hexdigest()
    check(
        minimal_sha == MINIMAL_SHA, "Minimal lifecycle differs from reference"
    )
    for module in (adapter, minimal):
        namespace = module.__dict__
        check(
            namespace.get("cp", namespace.get("cupy")) is cp
            and namespace.get("nf", namespace.get("nvmath_fft")) is nf
            and namespace.get("_numpy_dtype_name") is _numpy_dtype_name
            and namespace.get(
                "element_strides", namespace.get("_element_strides")
            )
            is element_strides,
            "Nvmath rows must use identical backend and metadata providers",
        )
    check(
        _numpy_dtype_name.cache_parameters()
        == {"maxsize": 256, "typed": False},
        "Requires cached dtype names",
    )
    stride_module = importlib.import_module(element_strides.__module__)
    check(
        any(
            stride_module.__file__.endswith(s)
            for s in importlib.machinery.EXTENSION_SUFFIXES
        ),
        "Requires compiled element_strides without substitution",
    )
    check(adapter.CachedFFT.execute is nf.FFT.execute, "Different FFT execute")
    check(hasattr(nf.FFT, "_try_reset_c2r_operand"),
          "Requires prepared C2R copying in nvmath reset_operand_unchecked")
    reset_reference = load("reference_reset.py")
    reference_sha = hashlib.sha256(
        inspect.getsource(reset_reference.reset_operand_unchecked).encode()
    ).hexdigest()
    check(reference_sha == REFERENCE_RESET_SHA,
          "Reference reset differs from the retained nvmath implementation")
    nvmath_reset = nf.FFT.reset_operand_unchecked
    check(adapter.CachedFFT.reset_operand_unchecked is nvmath_reset,
          "Ajay must use nvmath's existing reset API directly")
    missing = object()
    previous_reset = vars(adapter.CachedFFT).get(
        "reset_operand_unchecked", missing)
    fft_module = importlib.import_module("cupy.fft._fft")
    old_hook = fft_module._try_use_nvmath
    check(
        old_hook is adapter._try_use_nvmath, "Public FFT must start original"
    )
    check(driver._is_cuda_python(), "Requires the CuPy CUDA-Python build")
    check(type(adapter._RELEASE_WORKSPACE) is bool, "Workspace policy missing")
    settings = (
        cp.fft.config.use_nvmath,
        cp.fft.config.use_multi_gpus,
        adapter._RELEASE_WORKSPACE,
    )
    gc_enabled, cache, cache_owned = gc.isenabled(), None, False
    files = {
        "benchmark": __file__,
        "report": Path(__file__).with_name("report.py"),
        "cases": catalog.__file__,
        "adapter": adapter.__file__,
        "normalization": inspect.getfile(adapter._normalization_epilog),
        "minimal_lifecycle": minimal.__file__,
        "reference_reset": reset_reference.__file__,
        "nvmath_fft": inspect.getfile(nf.FFT),
        "public_fft": inspect.getfile(cp.fft.fft),
        "profiler": inspect.getfile(benchmark),
        "plan_cache": importlib.import_module("cupy.fft._cache").__file__,
        "native_cufft": cufft.__file__,
        "dtype_provider": importlib.import_module(
            _numpy_dtype_name.__module__
        ).__file__,
    }
    scaling_module = importlib.import_module("cupy.fft._nvmath_scaling")
    check_source_copy(scaling_module.__file__,
                      expected_adapter.with_name("_nvmath_scaling.py"))
    check_source_copy(Path(scaling_module.__file__).with_suffix(".cu"),
                      expected_adapter.with_name("_nvmath_scaling.cu"))
    files["nvrtc_scaling"] = scaling_module.__file__
    files["nvrtc_scaling_cuda"] = str(
        Path(scaling_module.__file__).with_suffix(".cu"))
    files["cupy_compiler"] = importlib.import_module(
        "cupy.cuda.compiler").__file__
    for name in (
        "nvmath.internal.tensor_ifc_cupy",
        "nvmath.internal._strides_helpers",
        "nvmath.fft._fft_impl",
        "nvmath._internal.cuda_scopes",
        "nvmath.internal._layout._layout",
        "nvmath._internal.stream._resolver",
    ):
        files[name] = importlib.import_module(name).__file__
    block_path = args.json.with_name(args.json.stem + "-blocks.jsonl")
    payload = {
        "status": "running",
        "execute_sha256": execute,
        "reference_reset": {
            "nvmath_revision": "eafc59f5f7f335957c34f8aa5a1a0d449694efd0",
            "method_sha256": reference_sha,
        },
        "cases": [],
        "configuration": {
            **vars(args),
            "json": str(args.json),
            "cpu": cpu,
            "order_seed": 0,
            "selected_cases": len(cases),
            "native_enable_nd_planning": cp.fft.config.enable_nd_planning,
            "native_workspace_policy": "retained with cached native plans",
            "block_checkpoint": str(block_path),
        },
        "sources": {name: identity(path) for name, path in files.items()},
        "measurement": "cupyx.profiler.benchmark CPU host latency and "
        "CUDA-event stream elapsed time; output disposal timed. "
        "GPU intervals include host submission gaps, "
        "not isolated kernel time.",
        "preparation_measurement": (
            "Single first-miss host intervals for construction, "
            "epilog preparation and plan; execution excluded. "
            "Wrappers removed before warm timing. "
            "Compiler first use and driver caches affect these observations."
        ),
        "limitations": "Curated cases have no production frequency weights. "
        "Only marked C2C default-forward cases use minimal and stateful. "
        "Native contrasts "
        "include backend plan, copy, scaling and output-layout differences; "
        "release_workspace applies only to the nvmath rows.",
        "candidate_threshold": {
            "slower_us": 0.2,
            "slower_percent": 2,
            "negative_block_fraction": 10 / 12,
        },
        "validation": {
            "row_agreement": (
                "Shared unscaled nvmath rows agree bitwise; "
                "fused Ajay and native checked independently against reference"
            ),
            "reference": "NumPy FFT of the same input promoted to double",
            "accuracy": "per-batch relative L2 <= 100 * input dtype epsilon; "
                        "absolute L2 for zero references",
        },
    }
    try:
        os.sched_setaffinity(0, {cpu})
        cp.fft.config.use_nvmath, cp.fft.config.use_multi_gpus = True, False
        adapter._RELEASE_WORKSPACE = args.release_workspace
        with cp.cuda.Device(args.device), warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                category=RuntimeWarning,
                message=r"nvmath-python rejected this FFT.*",
            )
            cache = cp.fft.config.get_plan_cache()
            check(
                cache.is_enabled and cache.get_curr_size() == 0,
                "Requires a fresh process with an empty enabled PlanCache",
            )
            cache_owned = True
            check(
                cp.fft.config.devices is None
                and cp.fft.config.get_current_callback_manager() is None
                and cufft.get_current_plan() is None,
                "Remove multi-GPU, callback and explicit-plan contexts",
            )
            stream = cp.cuda.get_current_stream()
            name = cp.cuda.runtime.getDeviceProperties(args.device)["name"]
            payload["environment"] = {
                "gpu": name.decode() if isinstance(name, bytes) else str(name),
                "host": platform.uname()._asdict(),
                "node": platform.node(),
                "cpu_model": cpu_model(),
                "python": sys.version,
                "python_executable": sys.executable,
                "numpy": np.__version__,
                "cupy": cp.__version__,
                "nvmath": importlib.import_module("nvmath").__version__,
                "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
                "cuda_driver": cp.cuda.runtime.driverGetVersion(),
                "stream_ptr": stream.ptr,
                "callback_compiler": {
                    package: installed_version(package)
                    for package in (
                        "nvidia-cuda-nvrtc", "cuda-bindings", "numba")
                },
            }
            block_path.parent.mkdir(parents=True, exist_ok=True)
            block_path.write_text("")
            save(args.json, payload)
            for case_index, spec in enumerate(cases, 1):
                payload["current_case"] = spec["name"]
                print(
                    f"[case] {case_index}/{len(cases)} {spec['name']}",
                    flush=True,
                )
                kwargs = dict(spec["kwargs"])
                if spec["numpy_axes"]:
                    kwargs["axes"] = tuple(np.int64(x) for x in kwargs["axes"])
                host = host_input(np, spec, kwargs, 202)
                warm_host = host_input(np, spec, kwargs, 101)
                operand = device_input(cp, host, spec["layout"])
                warm_operand = device_input(cp, warm_host, spec["layout"])
                public = getattr(cp.fft, spec["public"])
                check(
                    public.__globals__ is fft_module.__dict__,
                    "Public API does not read the selected hook",
                )
                hooks = {
                    "original": adapter._try_use_nvmath,
                    "axes": adapter._try_use_nvmath_ajay_unfused,
                    "fused": adapter._try_use_nvmath_ajay,
                    "c2r": adapter._try_use_nvmath_ajay_unfused,
                    "ajay": adapter._try_use_nvmath_ajay,
                    "unfused": adapter._try_use_nvmath_ajay_unfused,
                }
                labels = row_labels(
                    spec, compare_c2r_reset=args.compare_c2r_reset,
                    compare_fusion=args.compare_fusion)
                include_minimal = "minimal" in labels
                axes = tuple(range(operand.ndim))
                options = adapter._FFT_OPTIONS["C2C"]
                outputs, minimal_cache = [], {}
                stateful = None
                native_outputs, native_plans, native_plan_info = [], {}, []
                nvmath_plans = {}
                reset_observations = {}
                hits_per_call = {
                    "original": 1, "axes": 1, "fused": 1,
                    "c2r": 1, "ajay": 1, "unfused": 1,
                    "minimal": 0, "stateful": 0}

                def select(label):
                    # All rows use ordinary CachedFFT instances. Changing
                    # only this subclass's method keeps control selection
                    # outside the timed public call and leaves nf.FFT intact.
                    method = (reset_reference.reset_operand_unchecked
                              if label in ("original", "axes", "fused")
                              else nvmath_reset)
                    select_reset(adapter.CachedFFT, method,
                                 (node.plan for _, node in cache))
                    cp.fft.config.use_nvmath = label != "native"
                    fft_module._try_use_nvmath = hooks.get(
                        label, hooks["original"]
                    )

                reference_axes = transform_axes(spec, kwargs, operand.ndim)
                reference_dtype = (
                    np.complex128 if np.iscomplexobj(host) else np.float64
                )
                try:
                    select("ajay")
                    warm, ajay_preparation = prepare_call(
                        public, warm_operand, kwargs, adapter, nf)
                    preparation = {"ajay": ajay_preparation}
                    reference = getattr(np.fft, spec["public"])
                    warm_expected = reference(
                        warm_host.astype(reference_dtype), **kwargs)
                    expected = reference(
                        host.astype(reference_dtype), **kwargs)
                    outputs.append((warm, warm_expected))
                    entries = list(cache)
                    check(len(entries) == 1, "Expected one warm nvmath plan")
                    cache_key, node = entries[0]
                    check(
                        isinstance(cache_key, tuple)
                        and cache_key[:2] == ("nvmath", stream.ptr)
                        and (len(cache_key) == 3 or (
                            len(cache_key) == 4 and cache_key[3] in (
                                ("normalization", "N"),
                                ("normalization", "sqrtN")))),
                        "Warm request did not route to nvmath",
                    )
                    plan = node.plan
                    check(
                        isinstance(plan, adapter.CachedFFT)
                        and not plan.inplace,
                        "Wrong cached FFT plan",
                    )
                    check(
                        not plan.logger.isEnabledFor(logging.INFO),
                        "Disable nvmath INFO/DEBUG logging",
                    )
                    # Scaling-policy keys distinguish Ajay from Original;
                    # reset selection never changes the production cache key.
                    original_warm = None
                    original_key = callback_free_key(
                        adapter, warm_operand, spec, kwargs, stream.ptr)
                    if cache_key != original_key:
                        select("original")
                        original_warm, original_preparation = prepare_call(
                            public, warm_operand, kwargs, adapter, nf)
                        preparation["original"] = original_preparation
                        outputs.append((original_warm, warm_expected))
                    else:
                        preparation["original"] = {"shared_plan_with": "ajay"}
                    fused_warm, fused_plan, fused_key = None, None, None
                    if "fused" in labels:
                        before_fused = {key: node.plan for key, node in cache}
                        select("fused")
                        fused_warm, fused_preparation = prepare_call(
                            public, warm_operand, kwargs, adapter, nf)
                        preparation["fused"] = {
                            **fused_preparation, "shared_plan_with": "ajay"}
                        outputs.append((fused_warm, warm_expected))
                        after_fused = {key: node.plan for key, node in cache}
                        check(after_fused == before_fused
                              and after_fused.get(cache_key) is plan,
                              "Fused and Ajay must share one cached plan")
                        fused_key, fused_plan = cache_key, plan
                    nvmath_plans = {key: node.plan for key, node in cache}
                    original_plan = nvmath_plans.get(original_key)
                    check(
                        all(isinstance(value, adapter.CachedFFT)
                            and not value.inplace
                            for value in nvmath_plans.values())
                        and original_plan is not None
                        and nvmath_plans.get(cache_key) is plan,
                        "Original and Ajay warmups must use nvmath",
                    )
                    separate_scale_label = (
                        "unfused" if args.compare_fusion else
                        None if args.compare_c2r_reset else "axes")
                    separate_scale_warm, separate_scale_plan = None, None
                    if separate_scale_label is not None:
                        select(separate_scale_label)
                        (separate_scale_warm,
                         separate_preparation) = prepare_call(
                             public, warm_operand, kwargs, adapter, nf)
                        preparation[separate_scale_label] = {
                            **separate_preparation,
                            "shared_plan_with": "original"}
                        outputs.append((separate_scale_warm, warm_expected))
                        after_separate = {
                            key: node.plan for key, node in cache}
                        check(after_separate == nvmath_plans,
                              "Separate scaling must reuse Original's "
                              "callback-free plan without inserting "
                              "another plan")
                        separate_scale_plan = after_separate[original_key]
                        check(separate_scale_plan is original_plan,
                              "Separate scaling and Original use different "
                              "base plans")
                    c2r_warm = None
                    if "c2r" in labels:
                        select("c2r")
                        c2r_warm, c2r_preparation = prepare_call(
                            public, warm_operand, kwargs, adapter, nf)
                        preparation["c2r"] = {
                            **c2r_preparation,
                            "shared_plan_with": "original"}
                        outputs.append((c2r_warm, warm_expected))
                        after_c2r = {key: node.plan for key, node in cache}
                        check(after_c2r == nvmath_plans
                              and after_c2r.get(original_key)
                              is separate_scale_plan,
                              "C2R and Axes must share Original's "
                              "callback-free plan")
                    if include_minimal:
                        check(original_plan is plan and len(cache_key) == 3,
                              "Unscaled rows must share one cached plan")
                        key = adapter._create_key_from_operand(
                            operand, axes, options
                        )
                        check(
                            key == cache_key[2], "Minimal metadata key differs"
                        )
                        minimal_cache[(stream.ptr, key)] = plan
                        stateful = nf.FFT(
                            operand, axes=axes, options=options,
                            execution="cuda", stream=stream,
                        )
                        stateful.plan(stream=stream)

                    # Native owns its own Plan1d/PlanNd entries. Its public
                    # call may use several FFT passes and cache lookups.
                    select("native")
                    native_warm = public(warm_operand, **kwargs)
                    native_outputs.append((native_warm, warm_expected))
                    for native_key, native_node in cache:
                        if native_key in nvmath_plans:
                            continue
                        native_plan = native_node.plan
                        check(
                            isinstance(native_plan,
                                       (cufft.Plan1d, cufft.PlanNd))
                            and native_plan.gpus is None,
                            "Native warmup created an unexpected plan",
                        )
                        native_plans[native_key] = native_plan
                        native_plan_info.append({
                            "key": repr(native_key),
                            "type": type(native_plan).__name__,
                            "workspace_bytes": native_node.memsize,
                        })
                    check(native_plans, "Native warmup did not cache a plan")
                    del native_plan, native_node, native_key

                    def verify():
                        entries = {key: node.plan for key, node in cache}
                        check(
                            len(entries)
                            == len(nvmath_plans) + len(native_plans)
                            and all(entries.get(key) is value
                                    for key, value in nvmath_plans.items())
                            and all(entries.get(key) is value
                                    for key, value in native_plans.items()),
                            "CuPy cache identities changed; both backend "
                            "families must fit without eviction",
                        )
                        if include_minimal:
                            check(
                                len(minimal_cache) == 1
                                and minimal_cache[(stream.ptr, key)] is plan,
                                "Minimal dictionary changed",
                            )
                        for cached in nvmath_plans.values():
                            check(
                                cached._operand_released
                                and cached.operand.tensor is None,
                                "Cached plan retained its operand",
                            )
                            check(
                                cached.workspace.size == 0
                                or (cached.workspace.ptr is None)
                                == args.release_workspace,
                                "Wrong workspace policy",
                            )

                    def verify_c2r_reset(label):
                        if label not in ("original", "axes", "fused", "c2r",
                                         "ajay", "unfused"):
                            return
                        cached = (original_plan if label == "original"
                                  else fused_plan if label == "fused"
                                  else separate_scale_plan
                                  if label in ("axes", "c2r", "unfused")
                                  else plan)
                        method = cached.reset_operand_unchecked.__func__
                        prepared = (cached.fft_abstract_type == "C2R"
                                    and method is nvmath_reset)
                        info = getattr(
                            cached, "_c2r_reset_prepared_info", None)
                        last_used = getattr(
                            cached, "_c2r_reset_prepared_last_used", False)
                        check(type(cached) is adapter.CachedFFT,
                              "Unexpected nvmath cached plan class")
                        if label in ("original", "axes", "fused"):
                            check(method is (
                                      reset_reference.reset_operand_unchecked)
                                  and not last_used,
                                  "Reference control used prepared C2R reset")
                        else:
                            check(method is nvmath_reset,
                                  "Ajay, C2R and Unfused must use nvmath's "
                                  "existing reset API")
                        if prepared:
                            check(info is not None and info["eligible"]
                                  and last_used,
                                  "Prepared C2R reset was not used: "
                                  f"{info!r}")
                        reset_observations[label] = {
                            "reset_method": method.__qualname__,
                            "c2r_preparation": {
                                "selected": prepared,
                                "info": (dict(info)
                                         if prepared and info else None),
                                "last_used": last_used,
                            },
                        }

                    verify()
                    comparisons = []
                    for input_index, (fresh, values, wanted) in enumerate((
                        (operand, host, expected),
                        (warm_operand, warm_host, warm_expected),
                    )):
                        previous = None
                        for label in labels:
                            select(label)
                            before = counts(cache)
                            if label == "stateful":
                                # Rebind for untimed two-input validation.
                                stateful.reset_operand_unchecked(
                                    fresh, stream=stream)
                                out = stateful.execute(
                                    direction=nf.FFTDirection.FORWARD,
                                    stream=stream,
                                    release_workspace=args.release_workspace,
                                )
                            elif label == "minimal":
                                out = minimal.execute_cached(
                                    fresh,
                                    axes,
                                    options,
                                    minimal_cache,
                                    release_workspace=args.release_workspace,
                                )
                            else:
                                out = public(fresh, **kwargs)
                            verify_c2r_reset(label)
                            if label == "native":
                                native_outputs.append((out, wanted))
                            else:
                                outputs.append((out, wanted))
                            actual = out.get()
                            if label != "native":
                                if (previous is not None
                                        and (original_plan is plan
                                             or args.compare_c2r_reset
                                             and label == "ajay")):
                                    check(
                                        actual.dtype == previous.dtype
                                        and actual.shape == previous.shape
                                        and actual.tobytes()
                                        == previous.tobytes(),
                                        f"FFT rows differ bitwise for {label}",
                                    )
                                previous = actual
                            comparisons.append(
                                (input_index, label, actual, wanted))
                            np.testing.assert_array_equal(fresh.get(), values)
                            after = counts(cache)
                            if (label == "native"
                                    and label not in hits_per_call):
                                check(after[1] == before[1]
                                      and after[0] > before[0],
                                      "Native preflight was not a warm hit")
                                hits_per_call[label] = after[0] - before[0]
                            check(
                                after
                                == (
                                    before[0] + hits_per_call[label],
                                    before[1],
                                ),
                                "Wrong preflight cache path",
                            )
                            verify()
                    # Shared unscaled rows must agree bitwise. A fused
                    # scaling plan may round differently; validate every row
                    # independently against the same numerical reference.
                    accuracy = [
                        {"input": input_index, "row": label,
                         **reference_error(np, actual, wanted,
                                           reference_axes, host.dtype)}
                        for input_index, label, actual, wanted in comparisons
                    ]
                    accuracy.append({
                        "input": "planning-warmup", "row": "ajay",
                        **reference_error(np, warm.get(), warm_expected,
                                          reference_axes, host.dtype),
                    })
                    accuracy.append({
                        "input": "planning-warmup", "row": "native",
                        **reference_error(np, native_warm.get(), warm_expected,
                                          reference_axes, host.dtype),
                    })
                    comparisons.clear()
                    del actual, previous
                    pointers = []
                    for out, wanted in outputs:
                        check(
                            out.shape == warm.shape
                            and out.dtype == warm.dtype
                            and out.strides == warm.strides,
                            "Different output metadata",
                        )
                        pointers.append(out.data.ptr)
                    # Native may use another legal output layout and need not
                    # agree bitwise with nvmath. Its reference check is above.
                    pointers.extend(out.data.ptr for out, _ in native_outputs)
                    check(
                        len(set(pointers)) == len(pointers)
                        and not {
                            operand.data.ptr,
                            warm_operand.data.ptr,
                        }.intersection(pointers),
                        "Outputs alias",
                    )
                    output_metadata = {
                        backend: {
                            "shape": value.shape,
                            "dtype": str(value.dtype),
                            "strides": value.strides,
                            "c_contiguous": bool(value.flags.c_contiguous),
                            "f_contiguous": bool(value.flags.f_contiguous),
                        }
                        for backend, value in (
                            ("nvmath", warm), ("native", native_warm)
                        )
                    }
                    outputs.clear()
                    native_outputs.clear()
                    normalization = normalization_metadata(
                        np, spec, kwargs, operand.shape, warm.shape,
                        warm.dtype)
                    del (out, warm, original_warm, fused_warm,
                         separate_scale_warm, c2r_warm,
                         native_warm, fresh)
                    calls = {
                        label: (public, (operand,), kwargs)
                        for label in ("original", "axes", "fused", "c2r",
                                      "unfused", "ajay", "native")
                        if label in labels
                    }
                    if include_minimal:
                        calls["minimal"] = (
                            minimal.execute_cached,
                            (operand, axes, options, minimal_cache),
                            {"release_workspace": args.release_workspace},
                        )
                        # The timed object remains bound to this operand.
                        stateful.reset_operand_unchecked(
                            operand, stream=stream)
                        calls["stateful"] = (
                            stateful.execute, (),
                            {"direction": nf.FFTDirection.FORWARD,
                             "stream": stream,
                             "release_workspace": args.release_workspace},
                        )
                    plan_rows = [("original", original_key, original_plan),
                                 ("ajay", cache_key, plan)]
                    if "fused" in labels:
                        plan_rows.insert(1, ("fused", fused_key, fused_plan))
                    if separate_scale_label is not None:
                        plan_rows.insert(
                            1, (separate_scale_label, original_key,
                                separate_scale_plan))
                    if "c2r" in labels:
                        plan_rows.insert(
                            3, ("c2r", original_key, separate_scale_plan))
                    case = {
                        "case_id": case_ids[spec["name"]],
                        "case": spec,
                        "rows": labels,
                        "operand_shape": operand.shape,
                        "preparation": preparation,
                        "operand_strides": operand.strides,
                        "canonical_key_repr": repr(cache_key[2]),
                        "runtime_axes": repr(
                            kwargs.get("axes", kwargs.get("axis"))
                        ),
                        "normalization": normalization,
                        "correctness": True,
                        "reference_error": accuracy,
                        "route": {
                            label: "native_cupy" if label == "native"
                            else "nvmath" for label in labels
                        },
                        "cache_hits_per_call": {
                            label: hits_per_call[label] for label in labels
                        },
                        "nvmath_plans": {
                            label: {
                                "key": repr(plan_key),
                                "fused_scaling": getattr(
                                    cached, "_cupy_fft_epilog_ltoir", None
                                ) is not None,
                                "divisor": getattr(
                                    cached, "_cupy_fft_epilog_divisor", None
                                ),
                                **reset_observations[label],
                            }
                            for label, plan_key, cached in plan_rows
                        },
                        "native_plans": native_plan_info,
                        "output_metadata": output_metadata,
                        "blocks": [],
                    }
                    payload["cases"].append(case)
                    references = sys.getrefcount(operand)
                    gc.disable()
                    for index, order in enumerate(
                        row_orders(labels, args.blocks)
                    ):
                        print(
                            f"[progress] {spec['name']} "
                            f"block {index + 1}/{args.blocks}: "
                            + ", ".join(order),
                            flush=True,
                        )
                        block = {"index": index, "order": order, "rows": {}}
                        for label in order:
                            select(label)
                            function, positional, keywords = calls[label]
                            before = counts(cache)
                            result = benchmark(
                                function,
                                args=positional,
                                kwargs=keywords,
                                name=label,
                                n_repeat=args.repeats,
                                n_warmup=args.warmups,
                                devices=(args.device,),
                            )
                            cpu_times = result.cpu_times.tolist()
                            hits = hits_per_call[label] * (
                                args.warmups + len(cpu_times)
                            )
                            check(
                                len(cpu_times) == args.repeats
                                and counts(cache)
                                == (before[0] + hits, before[1]),
                                "Timed cache path changed",
                            )
                            verify()
                            verify_c2r_reset(label)
                            if stateful is not None:
                                check(
                                    not stateful._operand_released
                                    and stateful.operand.tensor is operand,
                                    "Stateful row lost its continuously "
                                    "bound operand",
                                )
                            block["rows"][label] = {
                                "cpu_seconds": cpu_times,
                                "gpu_seconds": result.gpu_times.tolist(),
                                "cpu_median_us": median(cpu_times) * 1e6,
                                "gpu_median_us": median(
                                    result.gpu_times[0].tolist()) * 1e6,
                            }
                        check(
                            sys.getrefcount(operand) == references,
                            "Timed call retained an input reference",
                        )
                        block["paired_cpu_us"] = {
                            name: block["rows"][a]["cpu_median_us"]
                            - block["rows"][b]["cpu_median_us"]
                            for name, (a, b) in PAIRS.items()
                            if a in labels and b in labels
                        }
                        block["paired_gpu_us"] = {
                            name: block["rows"][a]["gpu_median_us"]
                            - block["rows"][b]["gpu_median_us"]
                            for name, (a, b) in PAIRS.items()
                            if a in labels and b in labels
                        }
                        block["workspace"] = {
                            "required_bytes": plan.workspace.size,
                            "allocated_bytes": plan.workspace.allocated_size,
                            "retained": plan.workspace.ptr is not None,
                        }
                        case["blocks"].append(block)
                        # Incremental raw checkpoint avoids rewriting all prior
                        # cases' samples after every block on NFS.
                        with block_path.open("a") as log:
                            log.write(
                                json.dumps(
                                    {"case": spec["name"], "block": block}
                                )
                                + "\n"
                            )
                    case["summary"] = summarize(case["blocks"], labels)
                    np.testing.assert_array_equal(operand.get(), host)
                    summary = case["summary"]
                    reference_pair = (
                        "unfused_minus_ajay" if args.compare_fusion else
                        "fused_minus_ajay" if args.compare_c2r_reset
                        else "original_minus_ajay")
                    negative_key = (
                        "fusion_negative_median" if args.compare_fusion else
                        "fused_negative_median" if args.compare_c2r_reset
                        else "negative_median")
                    material_key = (
                        "fusion_material_regression_candidate"
                        if args.compare_fusion else
                        "fused_material_regression_candidate"
                        if args.compare_c2r_reset
                        else "material_regression_candidate")
                    percent_key = reference_pair + "_percent"
                    tag = (
                        "REGRESSION"
                        if summary[negative_key]
                        else "saving"
                    )
                    if summary[material_key]:
                        tag += " candidate"
                    slower = summary["sign_counts"][reference_pair][
                        "negative"
                    ]
                    print(
                        f"{spec['name']}: {tag} "
                        f"{reference_pair} {summary[reference_pair]:+.3f} us "
                        f"({summary[percent_key]:+.2f}%); "
                        f"ajay slower in {slower}/{args.blocks} blocks",
                        flush=True,
                    )
                    if include_minimal:
                        print(
                            "  original-minus-minimal "
                            f"{summary['original_minus_minimal']:.3f} us; "
                            "ajay-minus-minimal "
                            f"{summary['ajay_minus_minimal']:.3f} us",
                            flush=True,
                        )
                    print(
                        "  CPU us: "
                        + ", ".join(
                            f"{label}={summary['rows_cpu_us'][label]:.3f}"
                            for label in labels
                        )
                        + "; ajay-minus-native "
                        f"{summary['ajay_minus_native']:+.3f} us",
                        flush=True,
                    )
                    print(
                        "  GPU event us (includes host gaps): "
                        + ", ".join(
                            f"{label}={summary['rows_gpu_us'][label]:.3f}"
                            for label in labels
                        )
                        + "; ajay-minus-native "
                        f"{summary['paired_gpu_us']['ajay_minus_native']:+.3f}"
                        " us",
                        flush=True,
                    )
                finally:
                    fft_module._try_use_nvmath = old_hook
                    cp.fft.config.use_nvmath = True
                    stream.synchronize()
                    if stateful is not None:
                        stateful.free()
                    cache.clear()
                    minimal_cache.clear()
                    native_plans.clear()
                    nvmath_plans.clear()
                    outputs.clear()
                    native_outputs.clear()
                    if gc_enabled:
                        gc.enable()
                save(args.json, payload)
            payload["negative_cases"] = [
                c["case"]["name"]
                for c in payload["cases"]
                if c["summary"][
                    "fusion_negative_median" if args.compare_fusion else
                    "fused_negative_median" if args.compare_c2r_reset
                    else "negative_median"]
            ]
            print(
                "Negative paired medians: "
                + (", ".join(payload["negative_cases"]) or "none"),
                flush=True,
            )
            payload["status"] = "complete"
            if args.compare_fusion:
                report = args.json.with_suffix(".md")
                report.write_text(fusion_report(payload))
                print(f"Fusion table: {report}", flush=True)
            elif not args.compare_c2r_reset:
                report = args.json.with_suffix(".md")
                report.write_text(load("report.py").render(payload))
                print(f"Comparison table: {report}", flush=True)
    except BaseException as exc:
        payload["status"], payload["error"] = "failed", repr(exc)
        raise
    finally:
        fft_module._try_use_nvmath = old_hook
        if previous_reset is missing:
            if "reset_operand_unchecked" in vars(adapter.CachedFFT):
                del adapter.CachedFFT.reset_operand_unchecked
        else:
            adapter.CachedFFT.reset_operand_unchecked = previous_reset
        if cache_owned:
            cache.clear()
        (
            cp.fft.config.use_nvmath,
            cp.fft.config.use_multi_gpus,
            adapter._RELEASE_WORKSPACE,
        ) = settings
        os.sched_setaffinity(0, affinity)
        if gc_enabled:
            gc.enable()
        save(args.json, payload)


if __name__ == "__main__":
    main()
