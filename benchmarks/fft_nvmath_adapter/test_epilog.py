"""GPU-free checks of semantic normalization keys and matched controls."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy


ADAPTER = Path(__file__).resolve().parents[2] / 'cupy/fft/_fft_nvmath.py'
FUNCTIONS = {
    '_normalization_policy', '_normalization_divisor',
    '_normalization_epilog', '_scale_result',
    '_scale_result_ajay', '_create_key_from_operand', '_invert_permutation',
    '_c_contiguous_element_strides', '_try_use_nvmath',
    '_try_use_nvmath_ajay', '_try_use_nvmath_ajay_unfused',
}


class Array:
    def __init__(self, shape=(1024,), dtype='complex64'):
        self.shape, self.dtype = shape, numpy.dtype(dtype)
        self.ndim, self.itemsize = len(shape), self.dtype.itemsize
        self.size = math.prod(shape)
        self.data = SimpleNamespace(device_id=0)
        self.divisions = []
        stride, strides = self.itemsize, []
        for length in reversed(shape):
            strides.append(stride)
            stride *= length
        self.strides = tuple(reversed(strides))

    def __itruediv__(self, divisor):
        self.divisions.append(divisor)
        return self

    def transpose(self, permutation):
        return Array(tuple(self.shape[axis] for axis in permutation),
                     self.dtype)

    def copy(self):
        return Array(self.shape, self.dtype)


class Cache(dict):
    is_enabled = True
    fail_insert = False

    def __setitem__(self, key, value):
        if self.fail_insert:
            raise RuntimeError('insertion failed')
        super().__setitem__(key, value)


class TestEpilog(unittest.TestCase):
    def setUp(self):
        self.cache, self.plans, self.bindings = Cache(), [], {}
        self.fallbacks, self.key_callbacks, self.key_metadata = [], [], []
        self.uploads = self.synchronizations = 0
        self.current_device = 0
        self.fail_plan = self.fail_execute = False
        self.canonicalize_batches = False
        self.reject_layout = False
        self.stream = SimpleNamespace(ptr=7, synchronize=self.synchronize)
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

        class FFT:
            def __init__(self, operand, *, axes, options, **kwargs):
                self.device_id = owner.current_device
                self.inplace = False
                self.fft_abstract_type = options
                self.freed = self.released = self.resets = 0
                self.execute_options = []
                shape = list(operand.shape)
                dtype = operand.dtype.name
                if options == 'C2R':
                    shape[axes[-1]] = 2 * (shape[axes[-1]] - 1)
                    dtype = {'complex64': 'float32',
                             'complex128': 'float64'}[dtype]
                elif options == 'R2C':
                    shape[axes[-1]] = shape[axes[-1]] // 2 + 1
                    dtype = {'float32': 'complex64',
                             'float64': 'complex128'}[dtype]
                self.output = Array(tuple(shape), dtype)
                self.result_data_type = dtype
                owner.plans.append(self)

            @staticmethod
            def create_key_from_metadata(shape, dtype, space, **kwargs):
                epilog = kwargs['epilog']
                owner.key_callbacks.append(epilog)
                owner.key_metadata.append((shape, kwargs))
                if owner.reject_layout and shape == (4, 3, 8):
                    error = UnsupportedLayoutError('layout')
                    error.permutation, error.axes = (1, 0, 2), (1, 2)
                    raise error
                axes, options = tuple(kwargs['axes']), kwargs['options']
                if owner.canonicalize_batches:
                    problem = (math.prod(shape[a] for a in axes),
                               math.prod(shape) // math.prod(
                                   shape[a] for a in axes), dtype, options)
                else:
                    problem = shape, dtype, axes, kwargs['strides'], options
                callback = (None if epilog is None
                            else (epilog.ltoir, epilog.data))
                return problem, callback

            def plan(self, *, stream, epilog=None):
                self.epilog = epilog
                if owner.fail_plan:
                    raise RuntimeError('planning failed')

            def execute(self, **kwargs):
                self.execute_options.append(kwargs)
                if owner.fail_execute:
                    raise RuntimeError('execution failed')
                return self.output

            def reset_operand_unchecked(self, operand, *, stream):
                self.resets += 1
                self.bound = operand

            def release_operand(self):
                self.released += 1

            def free(self):
                self.freed += 1
                self._cupy_fft_epilog = None

        class UnsupportedLayoutError(Exception):
            pass

        def binding(dtype, divisor, stream_ptr):
            real = numpy.finfo(dtype).dtype.type
            key = dtype, real(divisor), stream_ptr, self.current_device
            if key not in self.bindings:
                self.uploads += 1
                self.bindings[key] = SimpleNamespace(
                    ltoir=b'IR-' + dtype.encode(), data=self.uploads,
                    _cupy_divisor=real(divisor))
            return self.bindings[key]

        cupy = ModuleType('cupy')
        cupy.ndarray = Array
        cuda = ModuleType('cupy.cuda')
        cufft = ModuleType('cupy.cuda.cufft')
        cufft.get_current_plan = lambda: None
        cuda.cufft, cuda.Device = cufft, Device
        cuda.runtime = SimpleNamespace(getDevice=lambda: self.current_device)
        cuda.get_current_stream = lambda: self.stream
        cupy.cuda = cuda
        self.modules = patch.dict('sys.modules', {
            'cupy': cupy, 'cupy.cuda': cuda, 'cupy.cuda.cufft': cufft})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.nvmath_fft = SimpleNamespace(
            FFT=FFT, UnsupportedLayoutError=UnsupportedLayoutError,
            compile_epilog=Mock(side_effect=AssertionError('Numba compiler')))
        self.ns = {
            'math': math, 'cupy': cupy, 'nvmath_fft': self.nvmath_fft,
            '_nvmath_scaling': SimpleNamespace(get_scaling_epilog=binding),
            'config': SimpleNamespace(
                use_nvmath=True, devices=None,
                get_current_callback_manager=lambda: None),
            'get_plan_cache': lambda: self.cache,
            '_FFT_OPTIONS': {x: x for x in ('C2C', 'R2C', 'C2R')},
            '_RESULT_DTYPES': {
                'R2C': {'float32': 'complex64', 'float64': 'complex128'},
                'C2R': {'complex64': 'float32', 'complex128': 'float64'}},
            '_DIRECTIONS': {'forward': -1, 'inverse': 1},
            '_VALID_NORMS': ('backward', 'ortho', 'forward'),
            '_RELEASE_WORKSPACE': False,
            '_normalize_axes': lambda ndim, axes: axes,
            '_normalize_axes_ajay': lambda ndim, axes: axes,
            '_element_strides': lambda s, size: tuple(x // size for x in s),
            '_numpy_dtype_name': lambda dtype: dtype.name,
            '_warn_fallback': self.fallbacks.append,
        }
        tree = ast.parse(ADAPTER.read_text())
        selected = ast.Module(body=[node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS
            or isinstance(node, ast.ClassDef) and node.name == 'CachedFFT'],
            type_ignores=[])
        self.assertEqual(len(selected.body), len(FUNCTIONS) + 1)
        exec(compile(selected, str(ADAPTER), 'exec'), self.ns)

    def synchronize(self):
        self.synchronizations += 1

    def invoke(self, *, norm=None, direction='inverse', kind='C2C',
               operand=None, axes=(0,), row='ajay'):
        name = {'ajay': '_try_use_nvmath_ajay',
                'unfused': '_try_use_nvmath_ajay_unfused',
                'original': '_try_use_nvmath'}[row]
        if operand is None:
            operand = Array(dtype='float32' if kind == 'R2C' else 'complex64')
        return self.ns[name](operand, None, axes, norm,
                             fft_type=kind, fft_direction=direction)

    def test_semantic_key_and_plan_owned_descriptor(self):
        self.invoke()
        plan = self.plans[0]
        epilog = plan.epilog
        key = next(iter(self.cache))
        self.assertEqual(len(key), 4)
        self.assertEqual(key[:2], ('nvmath', 7))
        self.assertEqual(key[3], ('normalization', 'N'))
        self.assertIsNone(self.key_callbacks[0])
        self.assertIs(plan._cupy_fft_epilog, epilog)
        self.assertIsNone(key[2][1])
        self.assertEqual(plan._cupy_fft_epilog_ltoir, epilog.ltoir)
        self.assertEqual(plan._cupy_fft_epilog_divisor, 1024)
        self.nvmath_fft.compile_epilog.assert_not_called()

    def test_warm_hits_skip_all_normalization_preparation(self):
        self.invoke()
        prepare = self.ns['_normalization_epilog'] = Mock(
            side_effect=AssertionError('warm normalization preparation'))
        divisor = self.ns['_normalization_divisor'] = Mock(
            side_effect=AssertionError('warm divisor resolution'))
        binding = self.ns['_nvmath_scaling'].get_scaling_epilog = Mock(
            side_effect=AssertionError('warm callback binding'))
        self.invoke()
        prepare.assert_not_called()
        divisor.assert_not_called()
        binding.assert_not_called()
        self.assertEqual(self.uploads, 1)
        self.assertEqual(len(self.plans), 1)
        self.assertEqual(self.plans[0].resets, 1)
        self.assertEqual(self.plans[0].released, 2)
        self.assertEqual(self.key_callbacks, [None, None])
        self.assertEqual(self.synchronizations, 0)

    def test_policy_describes_effective_scaling(self):
        policy = self.ns['_normalization_policy']
        cases = ((None, 'forward', 'none'),
                 ('backward', 'forward', 'none'),
                 ('forward', 'inverse', 'none'),
                 (None, 'inverse', 'N'),
                 ('backward', 'inverse', 'N'),
                 ('forward', 'forward', 'N'),
                 ('ortho', 'forward', 'sqrtN'),
                 ('ortho', 'inverse', 'sqrtN'))
        for norm, direction, expected in cases:
            with self.subTest(norm=norm, direction=direction):
                self.assertEqual(policy(norm, direction), expected)

    def test_equivalent_scaling_norms_and_directions_share_plan(self):
        for norm, direction in ((None, 'inverse'), ('backward', 'inverse'),
                                ('forward', 'forward')):
            self.invoke(norm=norm, direction=direction)
        self.assertEqual(len(self.plans), 1)
        self.invoke(norm='ortho', direction='forward')
        self.invoke(norm='ortho', direction='inverse')
        self.assertEqual(len(self.plans), 2)
        self.assertEqual(self.uploads, 2)
        self.assertEqual(len(self.cache), 2)
        self.assertEqual([p.resets for p in self.plans], [2, 1])

    def test_unscaled_rows_share_callback_free_plan(self):
        for row, norm, direction in (
                ('original', None, 'forward'), ('unfused', None, 'forward'),
                ('ajay', None, 'forward'), ('ajay', 'forward', 'inverse')):
            self.invoke(row=row, norm=norm, direction=direction)
        self.assertEqual(len(self.plans), 1)
        self.assertEqual(self.uploads, 0)
        self.assertIsNone(self.plans[0].epilog)
        self.assertEqual(self.plans[0].output.divisions, [])

    def test_control_keeps_separate_division_and_same_workspace_policy(self):
        for row in ('unfused', 'ajay'):
            self.invoke(row=row)
            self.invoke(row=row)
        unfused, fused = self.plans
        self.assertEqual(unfused.output.divisions, [1024, 1024])
        self.assertEqual(fused.output.divisions, [])
        self.assertEqual(unfused.execute_options, fused.execute_options)
        self.assertEqual(unfused.resets, fused.resets)
        self.assertEqual(unfused.released, fused.released)

    def test_family_norm_matrix_excludes_batch_axes_and_matches_precision(self):
        cases = (
            ('C2C', (4, 8, 16), 'complex64', 'forward', None, 1),
            ('C2C', (4, 8, 16), 'complex64', 'inverse', None, 128),
            ('C2C', (4, 8, 16), 'complex128', 'forward', 'forward', 128),
            ('C2C', (4, 8, 16), 'complex64', 'inverse', 'forward', 1),
            ('C2C', (4, 8, 16), 'complex64', 'inverse', 'ortho', math.sqrt(128)),
            ('R2C', (4, 8, 16), 'float32', 'forward', 'forward', 128),
            ('R2C', (4, 8, 16), 'float64', 'forward', 'ortho', math.sqrt(128)),
            ('C2R', (4, 8, 9), 'complex64', 'inverse', None, 128),
            ('C2R', (4, 8, 9), 'complex128', 'inverse', 'ortho', math.sqrt(128)),
            ('C2R', (4, 8, 9), 'complex64', 'inverse', 'forward', 1),
        )
        for kind, shape, dtype, direction, norm, divisor in cases:
            with self.subTest(kind=kind, dtype=dtype, norm=norm):
                operand = Array(shape, dtype)
                result = self.invoke(kind=kind, operand=operand, axes=(1, 2),
                                     direction=direction, norm=norm)
                plan = next(p for p in self.plans if p.output is result)
                real = numpy.finfo(result.dtype).dtype.type
                self.assertEqual(self.ns['_normalization_divisor'](
                    operand, (1, 2), norm, kind, direction), divisor)
                self.assertEqual(getattr(plan, '_cupy_fft_epilog_divisor', 1),
                                 real(divisor))
                if divisor != 1:
                    self.assertEqual(plan.epilog.ltoir,
                                     b'IR-' + result.dtype.name.encode())

    def test_unit_transform_avoids_callback(self):
        self.invoke(operand=Array((1,)), norm='ortho')
        self.assertEqual(self.uploads, 0)
        self.assertIsNone(self.plans[0].epilog)

    def test_canonical_batch_aliases_reuse_same_semantic_key(self):
        self.canonicalize_batches = True
        for norm in ('forward', 'ortho'):
            first = self.invoke(operand=Array((2, 3, 8)), axes=(2,),
                                norm=norm, direction='forward')
            second = self.invoke(operand=Array((6, 8)), axes=(1,),
                                 norm=norm, direction='forward')
            self.assertIs(first, second)
        self.assertEqual(len(self.plans), 2)
        self.assertEqual(self.uploads, 2)

    def test_layout_recovery_resolves_divisor_after_materializing_layout(self):
        self.reject_layout = True
        prepare = self.ns['_normalization_epilog'] = Mock(
            wraps=self.ns['_normalization_epilog'])
        self.invoke(operand=Array((4, 3, 8)), axes=(0, 2), norm='ortho')
        plan = self.plans[0]
        self.assertEqual(len(self.key_callbacks), 2)
        self.assertEqual(self.key_callbacks, [None, None])
        prepare.assert_called_once()
        arguments = prepare.call_args.args
        self.assertEqual(arguments[0].shape, (3, 4, 8))
        self.assertEqual(arguments[1], (1, 2))
        self.assertEqual(plan.output.shape, (3, 4, 8))
        self.assertEqual(plan.epilog._cupy_divisor, numpy.float32(math.sqrt(32)))

    def test_c2r_layout_recovery_uses_logical_unpacked_axis_length(self):
        self.reject_layout = True
        operand = Array((4, 3, 8))
        self.invoke(operand=operand, axes=(0, 2), kind='C2R', norm='ortho')
        plan = self.plans[0]
        self.assertEqual(self.key_callbacks, [None, None])
        self.assertEqual(plan.output.shape, (3, 4, 14))
        self.assertEqual(plan.epilog._cupy_divisor, numpy.float32(math.sqrt(56)))
        self.assertIs(plan._cupy_fft_epilog, plan.epilog)

    def test_c2r_uses_existing_reset_api(self):
        first, second = Array((17,)), Array((17,))
        self.invoke(kind='C2R', operand=first)
        self.invoke(kind='C2R', operand=second)
        plan = self.plans[0]
        self.assertIs(type(plan), self.ns['CachedFFT'])
        self.assertIs(plan.bound, second)
        self.assertEqual(plan.resets, 1)
        self.assertEqual(plan._cupy_fft_epilog_divisor, 32)

    def test_string_norm_subclasses_use_validated_meaning(self):
        class UnhashableNorm(str):
            __hash__ = None

        class MisleadingNorm(str):
            def __str__(self):
                return 'forward'

        expected = self.invoke(norm='ortho')
        for norm in (UnhashableNorm('ortho'), MisleadingNorm('ortho')):
            self.assertIs(self.invoke(norm=norm), expected)
        self.assertEqual(len(self.plans), 1)

    def test_streams_keep_separate_bindings_and_plans(self):
        self.invoke()
        self.stream = SimpleNamespace(ptr=8, synchronize=self.synchronize)
        self.invoke()
        self.assertEqual(len(self.plans), 2)
        self.assertEqual(self.uploads, 2)
        self.assertNotEqual(self.plans[0].epilog.data, self.plans[1].epilog.data)

    def test_operand_device_scopes_callback_key_and_plan_and_restores_caller(self):
        for row in ('ajay', 'unfused'):
            with self.subTest(row=row):
                operand = Array()
                operand.data.device_id = 1
                self.invoke(operand=operand, row=row)
                self.assertEqual(self.current_device, 0)
                self.assertEqual(self.plans[-1].device_id, 1)
        self.assertTrue(all(key[-1] == 1 for key in self.bindings))

    def test_planning_failure_waits_before_releasing_scalar(self):
        self.fail_plan = True
        self.assertIsNone(self.invoke())
        self.assertEqual(str(self.fallbacks[0]), 'planning failed')
        self.assertEqual(self.plans[0].freed, 1)
        self.assertEqual(self.synchronizations, 1)
        self.assertIsNone(self.plans[0]._cupy_fft_epilog)
        self.assertFalse(self.cache)

    def test_compilation_failure_does_not_create_a_plan(self):
        self.ns['_nvmath_scaling'].get_scaling_epilog = Mock(
            side_effect=RuntimeError('compilation failed'))
        self.assertIsNone(self.invoke())
        self.assertEqual(str(self.fallbacks[0]), 'compilation failed')
        self.assertFalse(self.plans)
        self.assertFalse(self.cache)

    def test_execution_failures_preserve_existing_ownership_policy(self):
        self.fail_execute = True
        with self.assertRaisesRegex(RuntimeError, 'execution failed'):
            self.invoke()
        cold = self.plans[0]
        self.assertEqual((cold.released, cold.freed), (1, 1))
        self.assertFalse(self.cache)
        self.fail_execute = False
        self.invoke()
        warm = self.plans[1]
        self.fail_execute = True
        with self.assertRaisesRegex(RuntimeError, 'execution failed'):
            self.invoke()
        self.assertEqual((warm.released, warm.freed), (2, 0))
        self.assertEqual(len(self.cache), 1)

    def test_disabled_cache_and_failed_insertion_wait_on_eviction(self):
        self.cache.is_enabled = False
        self.invoke()
        self.assertFalse(self.cache)
        self.assertEqual(self.synchronizations, 1)
        self.cache.is_enabled = True
        self.cache.fail_insert = True
        with self.assertRaisesRegex(RuntimeError, 'insertion failed'):
            self.invoke()
        self.assertEqual(self.synchronizations, 2)
        self.assertTrue(all(p.freed == 1 for p in self.plans))

    def test_original_cleanup_does_not_add_a_stream_wait(self):
        self.invoke(row='original')
        self.plans[0].free()
        self.assertEqual(self.synchronizations, 0)


if __name__ == '__main__':
    unittest.main()
