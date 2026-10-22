"""CPU-only checks of Ajay's epilog, reset selection and cache lifecycle.

The real functions are extracted from the adapter so these checks do not import
CuPy or initialize CUDA. Numerical correctness is checked by the GPU benchmark.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from statistics import median
from types import ModuleType, SimpleNamespace
import time
import unittest
from unittest.mock import Mock, patch

import numpy


ADAPTER = Path(__file__).resolve().parents[2] / 'cupy/fft/_fft_nvmath.py'
FUNCTIONS = {'_compile_normalization_epilog', '_try_use_nvmath_ajay'}


class Array:
    def __init__(self, shape=(1024,)):
        self.shape = shape
        self.ndim = len(shape)
        self.size = math.prod(shape)


class Cache(dict):
    is_enabled = True

    def __setitem__(self, key, value):
        if getattr(self, 'fail_insert', False):
            raise RuntimeError('insertion failed')
        super().__setitem__(key, value)


class TestEpilog(unittest.TestCase):
    def setUp(self):
        self.compilations = []
        self.callbacks = []
        self.current_device = 3
        self.plans = []
        self.fallbacks = []
        self.cache = Cache()
        self.fail_plan = self.fail_execute = False
        self.stream = SimpleNamespace(ptr=7)
        self.cupy = ModuleType('cupy')
        self.cupy.ndarray = Array
        cuda = ModuleType('cupy.cuda')
        cuda.get_current_stream = lambda: self.stream
        owner = self

        class Device:
            def __init__(self, device_id):
                self.device_id = device_id

            def __enter__(self):
                self.previous = owner.current_device
                owner.current_device = self.device_id
                return self

            def __exit__(self, *error):
                owner.current_device = self.previous

        cuda.Device = Device
        cufft = ModuleType('cupy.cuda.cufft')
        cufft.get_current_plan = lambda: None
        cuda.cufft = cufft
        self.cupy.cuda = cuda

        def compile_callback(function, element_dtype, user_info_dtype):
            self.assertEqual(element_dtype, user_info_dtype)
            captured = dict(zip(function.__code__.co_freevars,
                                function.__closure__))
            divisor = captured['divisor'].cell_contents
            self.compilations.append(
                (divisor, element_dtype, self.current_device))
            self.callbacks.append(function)
            return b'compiled-epilog'

        self.nvmath_fft = SimpleNamespace(
            UnsupportedLayoutError=type(
                'UnsupportedLayoutError', (Exception,), {}),
            compile_epilog=compile_callback)
        self.modules = patch.dict('sys.modules', {
            'cupy': self.cupy, 'cupy.cuda': cuda,
            'cupy.cuda.cufft': cufft,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        owner = self

        class Plan:
            def __init__(self, operand, *, axes, options, execution, stream):
                self.freed = self.released = self.resets = 0
                self.prepared_resets = 0
                self.device_id = 0
                self.fft_abstract_type = options
                shape = list(operand.shape)
                if options == 'C2R':
                    shape[axes[-1]] = 2 * (shape[axes[-1]] - 1)
                self.result_data_type = (
                    'float32' if options == 'C2R' else 'complex64')
                self.output = Array(tuple(shape))
                owner.plans.append(self)

            def get_output_layout(self):
                return self.output.shape, ()

            def plan(self, *, stream, epilog):
                self.epilog = epilog
                if owner.fail_plan:
                    raise RuntimeError('planning failed')

            def execute(self, **kwargs):
                if owner.fail_execute:
                    raise RuntimeError('execution failed')
                return self.output

            def reset_operand_unchecked(self, operand, **kwargs):
                self.resets += 1

            def _reset_operand_unchecked_prepared(self, operand, *, stream):
                self.prepared_resets += 1
                self.prepared_operand = operand
                self.prepared_stream = stream

            def release_operand(self):
                self.released += 1

            def free(self):
                self.freed += 1

        config = SimpleNamespace(
            use_nvmath=True, devices=None,
            get_current_callback_manager=lambda: None)
        self.ns = {
            'math': math, 'cupy': self.cupy, 'config': config,
            'nvmath_fft': self.nvmath_fft,
            'CachedFFT': Plan, 'get_plan_cache': lambda: self.cache,
            '_FFT_OPTIONS': {kind: kind for kind in ('C2C', 'R2C', 'C2R')},
            '_DIRECTIONS': {'forward': -1, 'inverse': 1},
            '_VALID_NORMS': ('backward', 'forward', 'ortho'),
            '_RELEASE_WORKSPACE': False,
            '_normalize_axes_ajay': lambda ndim, axes: axes,
            '_create_key_from_operand': lambda operand, axes, options: (
                operand.shape, axes, options),
            '_warn_fallback': self.fallbacks.append,
        }
        tree = ast.parse(ADAPTER.read_text())
        selected = ast.Module(body=[
            node for node in tree.body
            if (isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS
                or isinstance(node, ast.ClassDef)
                and node.name == '_PreparedC2RCachedFFT'
                or isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name)
                        and target.id == '_C2R_FFT'
                        for target in node.targets))
        ], type_ignores=[])
        self.assertEqual(len(selected.body), len(FUNCTIONS) + 2)
        exec(compile(selected, str(ADAPTER), 'exec'), self.ns)

    def invoke(self, *, norm='backward', direction='inverse', kind='C2C',
               operand=None, axes=(0,)):
        return self.ns['_try_use_nvmath_ajay'](
            operand or Array(), None, axes, norm,
            fft_type=kind, fft_direction=direction)

    def test_prepares_once_and_keeps_norm_direction_and_stream_separate(self):
        self.invoke()
        self.invoke()
        self.assertEqual(len(self.compilations), 1)
        self.assertEqual(self.plans[0].resets, 1)
        self.assertEqual(self.plans[0].released, 2)
        self.assertEqual(self.plans[0].freed, 0)
        self.assertEqual(next(iter(self.cache)), (
            'nvmath', 7, ((1024,), (0,), 'C2C'),
            'backward', 'inverse'))
        self.invoke(norm='ortho')
        self.invoke(direction='forward')
        self.stream = SimpleNamespace(ptr=8)
        self.invoke()
        self.assertEqual(len(self.plans), 4)
        self.assertEqual(len(self.compilations), 3)

    def test_unscaled_requests_share_original_key_without_callback(self):
        self.invoke(direction='forward')
        self.invoke(norm='forward', direction='inverse')
        self.assertEqual(list(self.cache), [
            ('nvmath', 7, ((1024,), (0,), 'C2C'))])
        self.assertEqual(len(self.plans), 1)
        self.assertEqual(self.plans[0].resets, 1)
        self.assertEqual(self.compilations, [])
        self.assertIsNone(self.plans[0].epilog)

    def test_norm_direction_matrix_uses_transform_size_not_batch_size(self):
        # Each request transforms the last two dimensions; the leading four
        # batches do not belong to the normalization size, 8 * 16 = 128.
        expected = (
            ('C2C', (4, 8, 16), None, 'forward', 1),
            ('C2C', (4, 8, 16), None, 'inverse', 128),
            ('C2C', (4, 8, 16), 'backward', 'forward', 1),
            ('C2C', (4, 8, 16), 'backward', 'inverse', 128),
            ('C2C', (4, 8, 16), 'forward', 'forward', 128),
            ('C2C', (4, 8, 16), 'forward', 'inverse', 1),
            ('C2C', (4, 8, 16), 'ortho', 'forward', math.sqrt(128)),
            ('C2C', (4, 8, 16), 'ortho', 'inverse', math.sqrt(128)),
            ('R2C', (4, 8, 16), None, 'forward', 1),
            ('R2C', (4, 8, 16), 'backward', 'forward', 1),
            ('R2C', (4, 8, 16), 'forward', 'forward', 128),
            ('R2C', (4, 8, 16), 'ortho', 'forward', math.sqrt(128)),
            ('C2R', (4, 8, 9), None, 'inverse', 128),
            ('C2R', (4, 8, 9), 'backward', 'inverse', 128),
            ('C2R', (4, 8, 9), 'forward', 'inverse', 1),
            ('C2R', (4, 8, 9), 'ortho', 'inverse', math.sqrt(128)),
        )
        for kind, shape, norm, direction, divisor in expected:
            with self.subTest(kind=kind, norm=norm, direction=direction):
                output = self.invoke(
                    kind=kind, operand=Array(shape), axes=(1, 2),
                    norm=norm, direction=direction)
                plan = next(p for p in self.plans if p.output is output)
                self.assertEqual(plan._cupy_fft_epilog_divisor, divisor)
                self.assertEqual(plan.epilog is None, divisor == 1)

    def test_alternating_norms_revisit_warm_plans_without_cold_work(self):
        requests = (
            (None, 'forward'), ('backward', 'forward'),
            ('forward', 'inverse'), ('forward', 'forward'),
            (None, 'inverse'), ('backward', 'inverse'),
            ('ortho', 'forward'), ('ortho', 'inverse'),
        )
        # A fused result must never go through the separate scaling helper.
        self.ns['_scale_result_ajay'] = Mock(
            side_effect=AssertionError('separate scaling after fused output'))
        for norm, direction in requests:
            self.invoke(norm=norm, direction=direction)
        self.assertEqual(len(self.plans), 5)
        self.assertEqual(len(self.compilations), 4)
        warmed = dict(self.cache)
        with patch.dict(self.ns, {
            '_compile_normalization_epilog': Mock(
                side_effect=AssertionError('cold helper on a warm hit')),
        }), patch.object(
                self.nvmath_fft, 'compile_epilog',
                side_effect=AssertionError('compiler on a warm hit')):
            for norm, direction in requests[::-1] + requests:
                self.invoke(norm=norm, direction=direction)
        self.assertEqual(len(self.plans), 5)
        self.assertEqual(len(self.compilations), 4)
        self.assertEqual(self.cache, warmed)
        self.assertEqual(sum(plan.released for plan in self.plans), 24)
        self.assertEqual(sum(plan.freed for plan in self.plans), 0)
        self.ns['_scale_result_ajay'].assert_not_called()

    def test_canonical_batch_aliases_keep_same_transform_scale(self):
        # A 1D length-eight transform with six batches can be represented by
        # different batch shapes. Model that same backend key explicitly.
        self.ns['_create_key_from_operand'] = lambda operand, axes, options: (
            (8,), 6, options)
        for norm, divisor in (('forward', 8), ('ortho', math.sqrt(8))):
            first = self.invoke(
                operand=Array((2, 3, 8)), axes=(2,),
                norm=norm, direction='forward')
            second = self.invoke(
                operand=Array((6, 8)), axes=(1,),
                norm=norm, direction='forward')
            plan = next(p for p in self.plans if p.output is first)
            self.assertIs(first, second)
            self.assertEqual(plan._cupy_fft_epilog_divisor, divisor)
            self.assertEqual(plan.resets, 1)
        self.assertEqual(len(self.plans), 2)
        self.assertEqual(len(self.compilations), 2)
        self.assertEqual(len(self.cache), 2)

    def test_string_norm_subclasses_reuse_plain_norm_plan(self):
        class UnhashableNorm(str):
            __hash__ = None

        class MisleadingStringNorm(str):
            def __str__(self):
                return 'forward'

        expected = self.invoke(norm='ortho')
        expected_key = next(iter(self.cache))
        for norm in (UnhashableNorm('ortho'), MisleadingStringNorm('ortho')):
            with self.subTest(norm_type=type(norm).__name__):
                self.assertIs(self.invoke(norm=norm), expected)
                self.assertEqual(list(self.cache), [expected_key])
        self.assertEqual(len(self.plans), 1)
        self.assertEqual(self.plans[0].resets, 2)
        self.assertEqual(len(self.compilations), 1)

    def test_scaled_request_does_not_reuse_original_plan(self):
        self.invoke(direction='forward')
        original = self.plans[0]
        self.invoke(norm='ortho')
        self.assertEqual(len(self.cache), 2)
        self.assertEqual(original.resets, 0)
        self.assertIsNone(original.epilog)
        self.assertIsNotNone(self.plans[1].epilog)

    def test_c2r_divisor_uses_logical_real_output(self):
        self.invoke(kind='C2R', operand=Array((8, 9)), axes=(0, 1))
        self.assertEqual(self.plans[0]._cupy_fft_epilog_divisor, 128)
        self.assertEqual(self.compilations, [(128, 'float32', 0)])

    def test_c2r_subclass_keeps_execute_and_selects_prepared_reset(self):
        candidate = self.ns['_PreparedC2RCachedFFT']
        self.assertIs(candidate.execute, self.ns['CachedFFT'].execute)
        first, second = Array((17,)), Array((17,))
        self.invoke(kind='C2R', operand=first)
        plan = self.plans[0]
        self.assertIs(type(plan), candidate)
        self.assertIs(next(iter(self.cache))[-1], candidate)
        self.invoke(kind='C2R', operand=second)
        self.assertEqual(plan.resets, 0)
        self.assertEqual(plan.prepared_resets, 1)
        self.assertIs(plan.prepared_operand, second)
        self.assertIs(plan.prepared_stream, self.stream)
        self.assertEqual(plan.released, 2)
        self.assertEqual(len(self.compilations), 1)

    def test_c2r_plans_do_not_reuse_original_even_without_scaling(self):
        operand = Array((17,))
        original = self.ns['CachedFFT'](
            operand, axes=(0,), options='C2R', execution='cuda',
            stream=self.stream)
        original_key = ('nvmath', 7, ((17,), (0,), 'C2R'))
        self.cache[original_key] = original
        for norm, key_length in (
                ('forward', 4), ('backward', 6), ('ortho', 6)):
            with self.subTest(norm=norm):
                output = self.invoke(kind='C2R', operand=operand, norm=norm)
                key, plan = next((key, plan)
                                for key, plan in self.cache.items()
                                if plan.output is output)
                self.assertEqual(key[:3], original_key)
                self.assertEqual(len(key), key_length)
                self.assertIs(key[-1], self.ns['_PreparedC2RCachedFFT'])
                self.assertIsNot(plan, original)
        self.assertEqual(original.resets, 0)
        self.assertEqual(original.prepared_resets, 0)
        self.assertIs(self.cache[original_key], original)
        self.assertEqual(len(self.cache), 4)

    def test_c2r_fused_control_and_prepared_use_separate_plans(self):
        operand = Array((17,))
        control = self.ns['CachedFFT']
        candidate = self.ns['_PreparedC2RCachedFFT']
        for norm in ('forward', 'backward'):
            with self.subTest(norm=norm):
                self.ns['_C2R_FFT'] = control
                control_out = self.invoke(
                    kind='C2R', operand=operand, norm=norm)
                control_plan = self.plans[-1]
                self.ns['_C2R_FFT'] = candidate
                candidate_out = self.invoke(
                    kind='C2R', operand=operand, norm=norm)
                candidate_plan = self.plans[-1]
                self.assertIsNot(control_out, candidate_out)
                self.assertIs(type(control_plan), control)
                self.assertIs(type(candidate_plan), candidate)
                self.ns['_C2R_FFT'] = control
                self.assertIs(self.invoke(
                    kind='C2R', operand=operand, norm=norm), control_out)
                self.ns['_C2R_FFT'] = candidate
                self.assertIs(self.invoke(
                    kind='C2R', operand=operand, norm=norm), candidate_out)
                self.assertEqual(control_plan.resets, 1)
                self.assertEqual(control_plan.prepared_resets, 0)
                self.assertEqual(candidate_plan.resets, 0)
                self.assertEqual(candidate_plan.prepared_resets, 1)
        self.assertEqual(len(self.cache), 4)
        self.assertEqual(len(self.compilations), 2)

    def test_other_fft_families_keep_the_base_class_and_key(self):
        for kind in ('C2C', 'R2C'):
            with self.subTest(kind=kind):
                self.invoke(kind=kind, direction='forward')
                plan = self.plans[-1]
                self.assertIs(type(plan), self.ns['CachedFFT'])
                self.assertIn(
                    ('nvmath', 7, ((1024,), (0,), kind)), self.cache)
                self.assertEqual(plan.prepared_resets, 0)

    def test_r2c_divisor_uses_real_input(self):
        self.invoke(kind='R2C', direction='forward', norm='forward',
                    operand=Array((33,)))
        self.assertEqual(self.plans[0]._cupy_fft_epilog_divisor, 33)

    def test_partial_axes_exclude_batch_dimensions(self):
        self.invoke(operand=Array((4, 3, 8)), axes=(0, 2), norm='ortho')
        self.assertEqual(
            self.plans[0]._cupy_fft_epilog_divisor, math.sqrt(32))

    def test_divisor_one_avoids_callback(self):
        self.invoke(operand=Array((1,)), norm='ortho')
        self.assertEqual(self.compilations, [])
        self.assertIsNone(self.plans[0].epilog)

    def test_delegates_output_dtype_and_device_to_nvmath(self):
        for dtype in ('float32', 'float64', 'complex64', 'complex128'):
            with self.subTest(dtype=dtype):
                plan = SimpleNamespace(
                    fft_abstract_type='C2C', result_data_type=dtype,
                    device_id=2)
                self.ns['_compile_normalization_epilog'](
                    plan, Array((32,)), (0,), 'forward', 'forward')
                self.assertEqual(self.compilations[-1], (32, dtype, 2))
                self.assertEqual(self.current_device, 3)
                self.assertEqual(
                    self.compilations[-1][0].dtype,
                    numpy.finfo(dtype).dtype)
                output = [None]
                value = numpy.dtype(dtype).type(64)
                self.callbacks[-1](output, 0, value, None, None)
                self.assertEqual(output[0], 2)
                self.assertEqual(output[0].dtype, numpy.dtype(dtype))

    def test_warm_hit_never_calls_nvmath_compiler(self):
        self.invoke()
        self.assertEqual(self.compilations, [(1024, 'complex64', 0)])
        self.invoke()
        self.assertEqual(self.compilations, [(1024, 'complex64', 0)])

    def test_preparation_collection_restores_hooks_before_warm_calls(self):
        path = ADAPTER.parents[2] / 'benchmarks/fft_nvmath_adapter/main.py'
        function = next(node for node in ast.parse(path.read_text()).body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == 'prepare_call')
        namespace = {'time': time}
        exec(compile(ast.Module(body=[function], type_ignores=[]),
                     str(path), 'exec'), namespace)
        prepare = namespace['prepare_call']
        derived = type('DerivedPlan', (self.ns['CachedFFT'],), {})
        self.ns['CachedFFT'] = derived
        adapter = SimpleNamespace(CachedFFT=derived)
        original_compiler = self.nvmath_fft.compile_epilog

        def public(operand):
            return self.invoke(operand=operand, norm='ortho')

        result, phases = prepare(public, Array(), {}, adapter, self.nvmath_fft)
        self.assertIs(result, self.plans[0].output)
        self.assertEqual(set(phases), {
            'construct_cpu_us', 'compile_epilog_cpu_us', 'plan_cpu_us'})
        self.assertTrue(all(value > 0 for value in phases.values()))
        self.assertNotIn('__init__', vars(derived))
        self.assertNotIn('plan', vars(derived))
        self.assertIs(self.nvmath_fft.compile_epilog, original_compiler)
        public(Array())
        self.assertEqual(len(self.compilations), 1)

        def fail(operand):
            raise RuntimeError('preparation failed')

        with self.assertRaisesRegex(RuntimeError, 'preparation failed'):
            prepare(fail, Array(), {}, adapter, self.nvmath_fft)
        self.assertNotIn('__init__', vars(derived))
        self.assertNotIn('plan', vars(derived))
        self.assertIs(self.nvmath_fft.compile_epilog, original_compiler)

    def test_summary_keeps_cpu_results_and_pairs_gpu_block_medians(self):
        path = ADAPTER.parents[2] / 'benchmarks/fft_nvmath_adapter/main.py'
        tree = ast.parse(path.read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == 'summarize')
        pairs = ast.literal_eval(next(
            node.value for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == 'PAIRS'
                    for target in node.targets)))
        labels = ('original', 'ajay', 'native')
        pairs = {name: (a, b) for name, (a, b) in pairs.items()
                 if a in labels and b in labels}
        namespace = {'median': median, 'math': math, 'PAIRS': pairs}
        exec(compile(ast.Module(body=[function], type_ignores=[]),
                     str(path), 'exec'), namespace)
        blocks = []
        for index, gpu_values in enumerate(((10, 9, 20),
                                             (100, 99, 21),
                                             (101, 0, 22))):
            rows = {
                label: {'cpu_median_us': cpu + index, 'gpu_median_us': gpu}
                for label, cpu, gpu in zip(labels, (20, 10, 9), gpu_values)
            }
            blocks.append({
                'rows': rows,
                'paired_cpu_us': {
                    name: rows[a]['cpu_median_us'] - rows[b]['cpu_median_us']
                    for name, (a, b) in pairs.items()},
                'paired_gpu_us': {
                    name: rows[a]['gpu_median_us'] - rows[b]['gpu_median_us']
                    for name, (a, b) in pairs.items()},
            })
        result = namespace['summarize'](blocks, labels)
        self.assertEqual(result['rows_cpu_us'], {
            'original': 21, 'ajay': 11, 'native': 10})
        self.assertEqual(result['original_minus_ajay'], 10)
        self.assertEqual(result['rows_gpu_us'], {
            'original': 100, 'ajay': 9, 'native': 21})
        # Preserve paired-block statistics; subtracting row medians gives 91.
        self.assertEqual(result['paired_gpu_us']['original_minus_ajay'], 1)
        self.assertEqual(result['paired_gpu_us']['ajay_minus_native'], -11)

    def test_key_rejection_keeps_existing_fallback_policy(self):
        def fail(*args):
            raise ValueError('unsupported layout')
        self.ns['_create_key_from_operand'] = fail
        self.assertIsNone(self.invoke())
        self.assertEqual(str(self.fallbacks[0]), 'unsupported layout')
        self.assertEqual(self.plans, [])

    def test_planning_failure_frees_and_does_not_cache(self):
        self.fail_plan = True
        self.assertIsNone(self.invoke())
        self.assertEqual(str(self.fallbacks[0]), 'planning failed')
        self.assertEqual(self.plans[0].freed, 1)
        self.assertFalse(self.cache)

    def test_execution_failure_releases_and_frees_cold_plan(self):
        self.fail_execute = True
        with self.assertRaisesRegex(RuntimeError, 'execution failed'):
            self.invoke()
        self.assertEqual(self.plans[0].released, 1)
        self.assertEqual(self.plans[0].freed, 1)
        self.assertFalse(self.cache)

    def test_warm_execution_failure_preserves_cache_ownership(self):
        self.invoke()
        self.fail_execute = True
        with self.assertRaisesRegex(RuntimeError, 'execution failed'):
            self.invoke()
        self.assertEqual(self.plans[0].released, 2)
        self.assertEqual(self.plans[0].freed, 0)
        self.assertEqual(len(self.cache), 1)

    def test_compilation_failure_frees_cold_plan(self):
        def fail(*args, **kwargs):
            raise RuntimeError('compilation failed')
        self.nvmath_fft.compile_epilog = fail
        self.assertIsNone(self.invoke())
        self.assertEqual(str(self.fallbacks[0]), 'compilation failed')
        self.assertEqual(self.current_device, 3)
        self.assertEqual(self.plans[0].freed, 1)
        self.assertFalse(self.cache)

    def test_disabled_cache_releases_plan_after_execution(self):
        self.cache.is_enabled = False
        self.invoke()
        self.assertFalse(self.cache)
        self.assertEqual(self.plans[0].released, 1)
        self.assertEqual(self.plans[0].freed, 1)

    def test_insertion_failure_frees_plan(self):
        self.cache.fail_insert = True
        with self.assertRaisesRegex(RuntimeError, 'insertion failed'):
            self.invoke()
        self.assertEqual(self.plans[0].freed, 1)


if __name__ == '__main__':
    unittest.main()
