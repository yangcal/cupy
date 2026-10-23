"""CPU checks for compiler isolation, typed code reuse and plan cleanup."""

from __future__ import annotations

import ast
import importlib.util
import math
import operator
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy


DIRECTORY = Path(__file__).resolve().parent
BASE = DIRECTORY.parents[1] / "cupy/fft/_fft_nvmath.py"
SPEC = importlib.util.spec_from_file_location(
    "_comparison_compilation_cache", DIRECTORY / "compilation_cache.py"
)
cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache)
DRIVER_SPEC = importlib.util.spec_from_file_location(
    "_epilog_comparison_driver", DIRECTORY / "main.py"
)
driver = importlib.util.module_from_spec(DRIVER_SPEC)
DRIVER_SPEC.loader.exec_module(driver)


def functions(path, namespace, names=None):
    """Load the real function bodies without importing either GPU package."""
    nodes = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and (names is None or node.name in names)
    ]
    namespace = dict(namespace, __file__=str(path))
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


class Array:
    def __init__(self, shape=(32,), dtype="complex64"):
        self.shape, self.dtype = shape, numpy.dtype(dtype)
        self.ndim, self.itemsize = len(shape), self.dtype.itemsize
        self.size = math.prod(shape)
        strides, stride = [], self.itemsize
        for extent in reversed(shape):
            strides.append(stride)
            stride *= extent
        self.strides = tuple(reversed(strides))


class PlanCache(dict):
    is_enabled = True
    fail_insert = False

    def __setitem__(self, key, value):
        if self.fail_insert:
            raise RuntimeError("insert failed")
        super().__setitem__(key, value)


class Comparison(unittest.TestCase):
    def setUp(self):
        self.target, self.version, self.nvrtc_version = "100", 12040, (13, 4)
        self.disk, self.plans, self.warnings = {}, [], []
        self.plan_cache = PlanCache()
        self.stream = SimpleNamespace(ptr=7)
        self.fail_plan = self.fail_execute = False
        self.callback_compiles = []
        self.uploads = []
        owner = self

        class Device:
            def __init__(self, device_id=None):
                self.compute_capability = owner.target

            def __enter__(self):
                return self

            def __exit__(self, *error):
                pass

        def compile_python(function, dtype, info_dtype, **kwargs):
            owner.callback_compiles.append(
                (dtype, info_dtype, function, kwargs)
            )
            return b"numba-" + dtype.encode()

        def asarray(values, dtype=None, *, blocking=False):
            class Scalar:
                def __init__(self):
                    self.values = numpy.asarray(values, dtype=dtype).copy()
                    self.dtype = self.values.dtype
                    self.shape = self.values.shape
                    self.data = SimpleNamespace(ptr=id(self))

            scalar = Scalar()
            owner.uploads.append(
                dict(
                    values=scalar.values.copy(),
                    blocking=blocking,
                    reference=weakref.ref(scalar),
                )
            )
            return scalar

        class Plan:
            def __init__(self, operand, *, axes, options, **kwargs):
                self.device_id = 0
                self.fft_abstract_type = options["fft_type"]
                self.result_data_type = operand.dtype.name
                shape = list(operand.shape)
                if self.fft_abstract_type == "C2R":
                    shape[axes[-1]] = 2 * (shape[axes[-1]] - 1)
                    self.result_data_type = {
                        "complex64": "float32",
                        "complex128": "float64",
                    }[operand.dtype.name]
                elif self.fft_abstract_type == "R2C":
                    shape[axes[-1]] = shape[axes[-1]] // 2 + 1
                    self.result_data_type = {
                        "float32": "complex64",
                        "float64": "complex128",
                    }[operand.dtype.name]
                self.output = Array(tuple(shape), self.result_data_type)
                self.freed = self.released = self.resets = 0
                self.execute_options = []
                owner.plans.append(self)

            @staticmethod
            def create_key_from_metadata(shape, dtype, memory_space, **kwargs):
                return (
                    shape,
                    dtype,
                    tuple(kwargs["axes"]),
                    kwargs["strides"],
                    kwargs["options"]["fft_type"],
                )

            def get_output_layout(self):
                return self.output.shape, ()

            def plan(self, *, stream, epilog):
                self.epilog = epilog
                if owner.fail_plan:
                    raise RuntimeError("planning failed")

            def execute(self, **kwargs):
                self.execute_options.append(kwargs)
                if owner.fail_execute:
                    raise RuntimeError("execution failed")
                return self.output

            def reset_operand_unchecked(self, operand, **kwargs):
                self.resets += 1

            def release_operand(self):
                self.released += 1

            def free(self):
                self.freed += 1
                self._cupy_fft_epilog_data = None

        self.Plan = Plan
        self.compile_nvrtc = Mock(return_value=b"nvrtc-ir")
        compiler = SimpleNamespace(
            _get_nvrtc_version=lambda: self.nvrtc_version,
            _compile_module_with_cache=self.compile_nvrtc,
            _kernel_cache_backend=SimpleNamespace(
                load=lambda name: self.disk.get(name),
                save=lambda name, code, source: self.disk.__setitem__(
                    name, code
                ),
            ),
        )
        cuda = ModuleType("cupy.cuda")
        cuda.Device, cuda.compiler = Device, compiler
        cuda.runtime = SimpleNamespace(runtimeGetVersion=lambda: 13040)
        cuda.cufft = SimpleNamespace(get_current_plan=lambda: None)
        cuda.get_current_stream = lambda: self.stream
        cupy = ModuleType("cupy")
        cupy.cuda, cupy.ndarray = cuda, Array
        cupy.asarray = asarray
        bindings = ModuleType("nvmath.bindings")
        bindings.cufft = SimpleNamespace(get_version=lambda: self.version)
        nvmath = ModuleType("nvmath")
        nvmath.bindings = bindings
        self.modules = patch.dict(
            sys.modules,
            {
                "cupy": cupy,
                "cupy.cuda": cuda,
                "nvmath": nvmath,
                "nvmath.bindings": bindings,
            },
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)
        search_path = patch.object(sys, "path", [str(DIRECTORY), *sys.path])
        search_path.start()
        self.addCleanup(search_path.stop)
        cache.clear_memory_cache()
        cache._versions.cache_clear()
        self.addCleanup(cache.clear_memory_cache)
        self.addCleanup(cache._versions.cache_clear)
        config = SimpleNamespace(
            use_nvmath=True,
            devices=None,
            get_current_callback_manager=lambda: None,
        )
        nf = ModuleType("nvmath.fft")
        nf.FFT, nf.compile_epilog = Plan, compile_python
        nf.UnsupportedLayoutError = type(
            "UnsupportedLayoutError", (Exception,), {}
        )
        nvmath.fft = nf
        common = functions(
            BASE,
            {"operator": operator, "nvmath_fft": nf},
            {
                "_normalize_axes",
                "_create_key_from_operand",
                "_c_contiguous_element_strides",
                "_invert_permutation",
            },
        )
        common.update(
            CachedFFT=Plan,
            get_plan_cache=lambda: self.plan_cache,
            _DIRECTIONS={"forward": -1, "inverse": 1},
            _FFT_OPTIONS={
                kind: {"fft_type": kind} for kind in ("C2C", "R2C", "C2R")
            },
            _VALID_NORMS=("backward", "forward", "ortho"),
            _warn_fallback=self.warnings.append,
        )
        base = ModuleType("cupy.fft._fft_nvmath")
        base.__dict__.update(common)
        fft = ModuleType("cupy.fft")
        fft.config = config
        cupy.fft = fft
        plan_cache_module = ModuleType("cupy.fft._cache")
        plan_cache_module.get_plan_cache = lambda: self.plan_cache
        sys.modules.update(
            {
                "nvmath.fft": nf,
                "cupy.fft": fft,
                "cupy.fft._cache": plan_cache_module,
                "cupy.fft._fft_nvmath": base,
                "compilation_cache": cache,
            }
        )
        self.siblings = {}
        for name in ("numba", "nvrtc"):
            adapter = importlib.import_module(f"{name}_backend.adapter")
            epilog = importlib.import_module(f"{name}_backend.epilog")
            compiler_module = importlib.import_module(
                f"{name}_backend.compiler"
            )
            self.siblings[name] = {
                **compiler_module.__dict__,
                **epilog.__dict__,
                "try_use_nvmath": adapter.try_use_nvmath,
            }

    def invoke(self, backend, *, norm=None, direction="inverse", operand=None):
        return self.siblings[backend]["try_use_nvmath"](
            operand or Array(),
            None,
            (-1,),
            norm,
            fft_type="C2C",
            fft_direction=direction,
        )

    def test_request_and_normalization_bodies_are_identical(self):
        for name, filename in (
            ("try_use_nvmath", "adapter.py"),
            ("compile_normalization_epilog", "epilog.py"),
        ):
            trees = [
                ast.parse(
                    (DIRECTORY / f"{backend}_backend" / filename).read_text()
                )
                for backend in ("numba", "nvrtc")
            ]
            bodies = [
                next(
                    node
                    for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == name
                )
                for tree in trees
            ]
            self.assertEqual(ast.dump(bodies[0]), ast.dump(bodies[1]))

    def test_both_compilers_reuse_code_in_memory_and_on_disk(self):
        for backend in self.siblings:
            compile_ = Mock(return_value=b"compiled")
            path = DIRECTORY / f"{backend}_backend/compiler.py"
            a = cache.get_code(backend, "complex64", compile_, path)
            b = cache.get_code(backend, "complex64", compile_, path)
            self.assertIs(a, b)
            self.assertEqual(
                cache.cache_info()["last_lookup"]["level"], "memory"
            )
            cache.clear_memory_cache()
            self.assertEqual(
                cache.get_code(backend, "complex64", compile_, path), a
            )
            self.assertEqual(
                cache.cache_info()["last_lookup"]["level"], "disk"
            )
            compile_.assert_called_once_with("complex64", "100", 12040)
            self.assertNotIn(
                "divisor", cache.cache_info()["last_lookup"]["identity"]
            )

    def test_target_precision_and_compiler_versions_do_not_alias(self):
        compile_ = Mock(return_value=b"compiled")
        path = DIRECTORY / "nvrtc_backend/compiler.py"
        cache.get_code("nvrtc", "complex64", compile_, path)
        self.target = "90"
        cache.get_code("nvrtc", "complex64", compile_, path)
        self.version = 11300
        cache.get_code("nvrtc", "complex64", compile_, path)
        self.nvrtc_version = (13, 5)
        cache.get_code("nvrtc", "complex64", compile_, path)
        cache.get_code("nvrtc", "complex128", compile_, path)
        self.assertEqual(compile_.call_count, 5)
        self.assertEqual(len(self.disk), 5)

    def test_compile_failure_is_not_cached_and_retry_succeeds(self):
        compile_ = Mock(side_effect=[RuntimeError("compile failed"), b"ok"])
        path = DIRECTORY / "numba_backend/compiler.py"
        with self.assertRaisesRegex(RuntimeError, "compile failed"):
            cache.get_code("numba", "float32", compile_, path)
        self.assertEqual(cache.cache_info()["entries"], 0)
        self.assertEqual(self.disk, {})
        self.assertEqual(
            cache.get_code("numba", "float32", compile_, path), b"ok"
        )
        self.assertEqual(compile_.call_count, 2)

    def test_divisor_uses_transformed_axes_and_full_real_extent(self):
        for name, ns in self.siblings.items():
            for kind, input_shape, output_shape, expected in (
                ("C2C", (4, 3, 8), (4, 3, 8), math.sqrt(32)),
                ("R2C", (4, 3, 8), (4, 3, 5), math.sqrt(32)),
                ("C2R", (4, 3, 5), (4, 3, 8), math.sqrt(32)),
            ):
                with self.subTest(backend=name, kind=kind):
                    p = SimpleNamespace(
                        fft_abstract_type=kind,
                        device_id=0,
                        result_data_type="float32"
                        if kind == "C2R"
                        else "complex64",
                        get_output_layout=lambda: (output_shape, ()),
                    )
                    compile_ = Mock(return_value=b"ir")
                    helper = ns["compile_normalization_epilog"]
                    with patch.dict(helper.__globals__, compile_code=compile_):
                        ns["compile_normalization_epilog"](
                            p, Array(input_shape), (0, 2), "ortho", "inverse"
                        )
                        compile_.assert_called_once_with(p.result_data_type)
                        self.assertEqual(p._cupy_fft_epilog_ltoir, b"ir")
                        self.assertEqual(
                            p._cupy_fft_epilog_data.values.dtype,
                            numpy.finfo(p.result_data_type).dtype,
                        )
                        self.assertEqual(
                            p._cupy_fft_epilog_data.values[0],
                            numpy.float32(expected),
                        )

    def test_identity_normalization_never_requests_compilation(self):
        for name, ns in self.siblings.items():
            for norm, direction, shape in (
                ("backward", "forward", (32,)),
                ("forward", "inverse", (32,)),
                ("ortho", "forward", (1,)),
            ):
                with self.subTest(backend=name, norm=norm, shape=shape):
                    p = SimpleNamespace(fft_abstract_type="C2C")
                    compile_ = Mock(side_effect=AssertionError("compiled"))
                    helper = ns["compile_normalization_epilog"]
                    with patch.dict(helper.__globals__, compile_code=compile_):
                        self.assertIsNone(
                            ns["compile_normalization_epilog"](
                                p, Array(shape), (0,), norm, direction
                            )
                        )
                    self.assertEqual(p._cupy_fft_epilog_divisor, 1)
                    self.assertIsNone(p._cupy_fft_epilog_ltoir)
                    self.assertIsNone(p._cupy_fft_epilog_data)
        self.assertEqual(self.uploads, [])

    def test_warm_hits_skip_compile_and_use_ordinary_release_workspace(self):
        for name in self.siblings:
            self.invoke(name)
            self.invoke(name)
        self.assertEqual(len(self.plans), 2)
        self.assertEqual(len(self.callback_compiles), 1)
        self.assertEqual(self.compile_nvrtc.call_count, 1)
        self.assertEqual(len(self.uploads), 2)
        for p in self.plans:
            self.assertEqual((p.resets, p.released, p.freed), (1, 2, 0))
            self.assertTrue(
                all(call["release_workspace"] for call in p.execute_options)
            )
        self.stream = SimpleNamespace(ptr=9)
        self.invoke("numba")
        self.assertEqual(len(self.plans), 3)
        self.assertEqual(len(self.callback_compiles), 1)
        self.assertEqual(len(self.uploads), 3)

    def test_norms_directions_and_backends_keep_plans_separate(self):
        class UnhashableNorm(str):
            __hash__ = None

        for name in self.siblings:
            self.invoke(name, norm=None)
            self.invoke(name, norm=UnhashableNorm("backward"))
            self.invoke(name, norm="ortho")
            self.invoke(name, norm=None, direction="forward")
            self.invoke(name, norm="forward", direction="inverse")
        self.assertEqual(len(self.plans), 6)
        self.assertEqual(len(self.plan_cache), 6)

    def test_plan_and_execute_failure_cleanup_matches_baseline(self):
        for name in self.siblings:
            with self.subTest(backend=name):
                self.fail_plan = True
                self.assertIsNone(self.invoke(name))
                self.assertEqual(self.plans[-1].freed, 1)
                self.fail_plan = False
                self.fail_execute = True
                with self.assertRaisesRegex(RuntimeError, "execution failed"):
                    self.invoke(name)
                self.assertEqual(self.plans[-1].freed, 1)
                self.assertEqual(self.plans[-1].released, 1)
                self.fail_execute = False
        self.assertEqual(self.plan_cache, {})

    def test_nvrtc_pointer_precision_callback_types_and_offset_abi(self):
        source = self.siblings["nvrtc"]["source_for_dtype"]
        cuda_source = DIRECTORY / "nvrtc_backend/scaling.cu"
        for type_id, (dtype, result, scalar, suffix) in enumerate(
            (
                ("float32", "float", "float", "Real"),
                ("float64", "double", "double", "DoubleReal"),
                ("complex64", "float2", "float", "Complex"),
                ("complex128", "double2", "double", "DoubleComplex"),
            )
        ):
            for version, uint64 in ((11299, 0), (11300, 1)):
                code, options = source(dtype, version)
                self.assertEqual(code, cuda_source.read_text())
                self.assertIn(f"-DCUPY_FFT_CALLBACK_TYPE={type_id}", options)
                self.assertIn(
                    f"-DCUPY_FFT_CALLBACK_OFFSET_UINT64={uint64}", options
                )
                self.assertIn(f"using CallbackOutput = {result};", code)
                self.assertIn(f"using CallbackReal = {scalar};", code)
                self.assertIn(f"cufftJITCallbackStore{suffix}", code)
                self.assertRegex(
                    code,
                    rf"#(?:if|elif) CUPY_FFT_CALLBACK_TYPE == {type_id}\s+"
                    rf"using CallbackOutput = {result};\s+"
                    rf"using CallbackReal = {scalar};\s+"
                    "#define CUPY_FFT_CALLBACK_NAME "
                    f"cufftJITCallbackStore{suffix}",
                )
                self.assertIn(
                    "*static_cast<const CallbackReal*>(user_info)", code
                )
                self.assertNotIn('extern "C"', code)
                self.assertNotIn("constexpr", code)
                self.assertEqual(code.count("__device__ void"), 1)
        compiler_source = DIRECTORY / "nvrtc_backend/compiler.py"
        self.assertFalse(
            any(
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and "__device__" in node.value
                for node in ast.walk(ast.parse(compiler_source.read_text()))
            )
        )

    def test_cuda_source_edits_invalidate_memory_and_disk_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            compiler_source = Path(temporary) / "compiler.py"
            cuda_source = Path(temporary) / "scaling.cu"
            compiler_source.write_text("# fixed compiler wrapper\n")
            cuda_source.write_text("// CUDA revision 1\n")
            compile_ = Mock(side_effect=[b"IR-version-1", b"IR-version-2"])
            first = cache.get_code(
                "nvrtc", "complex64", compile_, compiler_source, cuda_source
            )
            # Same path and length: content must change the process key.
            cuda_source.write_text("// CUDA revision 2\n")
            second = cache.get_code(
                "nvrtc", "complex64", compile_, compiler_source, cuda_source
            )
            self.assertNotEqual(first, second)
            self.assertEqual(compile_.call_count, 2)
            cache.clear_memory_cache()
            for revision, expected in ((1, first), (2, second)):
                cuda_source.write_text(f"// CUDA revision {revision}\n")
                self.assertEqual(
                    cache.get_code(
                        "nvrtc",
                        "complex64",
                        compile_,
                        compiler_source,
                        cuda_source,
                    ),
                    expected,
                )
                self.assertEqual(
                    cache.cache_info()["last_lookup"]["level"], "disk"
                )
            self.assertEqual(compile_.call_count, 2)

    def test_numba_reads_matching_real_precision_runtime_data(self):
        for dtype in ("float32", "float64", "complex64", "complex128"):
            self.siblings["numba"]["compile_code"](dtype)
            element, info, function, kwargs = self.callback_compiles[-1]
            real_dtype = numpy.finfo(dtype).dtype
            self.assertEqual((element, info), (dtype, real_dtype.name))
            self.assertEqual(function.__code__.co_freevars, ())
            self.assertEqual(kwargs, {"compute_capability": self.target})
            data = numpy.dtype(dtype).type(
                6 + 8j if dtype.startswith("complex") else 6
            )
            output = numpy.empty(1, dtype=dtype)
            for divisor in (2, 4):
                user_info = numpy.asarray([divisor], dtype=real_dtype)
                function(output, 0, data, user_info, None)
                self.assertEqual(output[0], data / user_info[0])

    def test_different_plan_divisors_share_code_but_own_distinct_data(self):
        for backend in self.siblings:
            for size in (32, 64):
                self.invoke(backend, operand=Array((size,)))
            cache.clear_memory_cache()
            self.invoke(backend, operand=Array((128,)))
        self.assertEqual(len(self.plans), 6)
        self.assertEqual(len(self.callback_compiles), 1)
        self.assertEqual(self.compile_nvrtc.call_count, 1)
        self.assertEqual(len(self.uploads), 6)
        self.assertTrue(all(upload["blocking"] for upload in self.uploads))
        pointers = []
        for plan, size in zip(self.plans, (32, 64, 128) * 2):
            scalar = plan._cupy_fft_epilog_data
            self.assertEqual(scalar.values.tolist(), [size])
            self.assertEqual(scalar.dtype, numpy.dtype("float32"))
            self.assertEqual(plan.epilog["data"], scalar.data.ptr)
            pointers.append(scalar.data.ptr)
        self.assertEqual(len(set(pointers)), 6)

    def test_epilog_pointer_depends_on_retained_plan_owner(self):
        for backend, ns in self.siblings.items():
            with self.subTest(backend=backend):
                plan = SimpleNamespace(
                    fft_abstract_type="C2C",
                    device_id=0,
                    result_data_type="complex64",
                )
                epilog = ns["compile_normalization_epilog"](
                    plan, Array(), (0,), "backward", "inverse"
                )
                scalar = plan._cupy_fft_epilog_data
                reference = weakref.ref(scalar)
                self.assertEqual(epilog["data"], scalar.data.ptr)
                del scalar
                self.assertIsNotNone(reference())
                del plan
                self.assertIsNone(reference())


class DriverReference(unittest.TestCase):
    def test_case_counts_and_names_are_unambiguous(self):
        for smoke, count in ((False, 67), (True, 11)):
            cases = driver.cases(smoke)
            self.assertEqual(len(cases), count)
            self.assertEqual(len({case["name"] for case in cases}), count)
            self.assertEqual(
                {case["precision"] for case in cases}, {"single", "double"}
            )

    def test_full_case_inputs_shapes_precisions_and_numpy_references(self):
        for case in driver.cases():
            with self.subTest(case=case["name"]):
                host, wanted, dtype = driver.inputs(case, 1234)
                real = (
                    "float32" if case["precision"] == "single" else "float64"
                )
                complex_type = (
                    "complex64" if real == "float32" else "complex128"
                )
                c2r = case["public"].startswith("irfft")
                r2c = case["public"].startswith("rfft")
                packed_shape = list(case["shape"])
                packed_shape[case["axes"][-1]] = (
                    packed_shape[case["axes"][-1]] // 2 + 1
                )
                self.assertEqual(
                    host.dtype.name, real if r2c else complex_type
                )
                self.assertEqual(
                    host.shape, tuple(packed_shape) if c2r else case["shape"]
                )
                self.assertEqual(
                    wanted.shape, tuple(packed_shape) if r2c else case["shape"]
                )
                self.assertEqual(dtype, real if c2r else complex_type)
                self.assertEqual(
                    wanted.dtype.name, "float64" if c2r else "complex128"
                )
                reference_input = host.astype(
                    numpy.complex128
                    if numpy.iscomplexobj(host)
                    else numpy.float64
                )
                expected = getattr(numpy.fft, case["public"])(
                    reference_input, **driver.kwargs_for(case)
                )
                numpy.testing.assert_array_equal(wanted, expected)

    def test_public_kwargs_keep_batch_dimensions_out_of_axes(self):
        for case in driver.cases():
            kwargs = driver.kwargs_for(case)
            self.assertEqual(kwargs["norm"], case["norm"])
            if len(case["axes"]) == 1:
                self.assertEqual(
                    kwargs, {"axis": case["axes"][0], "norm": case["norm"]}
                )
            else:
                self.assertEqual(
                    kwargs, {"axes": case["axes"], "norm": case["norm"]}
                )
            if case["shape"] == (8, 32):
                self.assertEqual(kwargs["axis"], 1)
                self.assertTrue(case["public"].endswith("fft"))

    def test_numerical_guard_rejects_missing_inverse_normalization(self):
        case = next(
            case
            for case in driver.cases(True)
            if case["public"] == "ifft" and case["precision"] == "double"
        )
        _, wanted, dtype = driver.inputs(case, 1234)

        class Result:
            def __init__(self, values):
                self.values = values
                self.shape, self.dtype = values.shape, values.dtype

            def get(self):
                return self.values

        self.assertEqual(
            driver.check_result(
                Result(wanted.copy()), wanted, dtype, case["axes"]
            ),
            0,
        )
        with self.assertRaisesRegex(RuntimeError, "normalization error: 31"):
            driver.check_result(
                Result(wanted * 32), wanted, dtype, case["axes"]
            )
        with self.assertRaisesRegex(RuntimeError, "Nonfinite FFT output"):
            driver.check_result(
                Result(numpy.full_like(wanted, numpy.inf)),
                wanted,
                dtype,
                case["axes"],
            )
        zero = numpy.zeros_like(wanted)
        with self.assertRaisesRegex(RuntimeError, "normalization error"):
            driver.check_result(
                Result(numpy.ones_like(wanted)), zero, dtype, case["axes"]
            )


if __name__ == "__main__":
    unittest.main()
