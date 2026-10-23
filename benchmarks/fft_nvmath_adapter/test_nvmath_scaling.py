"""GPU-free checks of callback compilation, identity and scalar ownership."""

from concurrent.futures import ThreadPoolExecutor
import gc
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy


SOURCE = (Path(__file__).resolve().parents[2]
          / 'cupy/fft/_nvmath_scaling.py')


class DeviceCallable:
    def __init__(self, *, ltoir, data):
        self.ltoir = ltoir
        self.data = data


class TestNvmathScaling(unittest.TestCase):
    def setUp(self):
        self.device_id = 0
        self.uploads = []
        self.scalar_refs = []
        self.next_ptr = 100
        self.device = Mock(return_value=SimpleNamespace(
            compute_capability='100'))
        self.compiler = Mock(side_effect=lambda source, options, **kwargs:
                             (str((options, kwargs))).encode())
        self.version = Mock(return_value=12400)

        cupy = ModuleType('cupy')
        cuda = ModuleType('cupy.cuda')
        cuda.runtime = SimpleNamespace(getDevice=lambda: self.device_id)
        cuda.Device = self.device
        cuda.compiler = SimpleNamespace(
            _compile_module_with_cache=self.compiler)
        cupy.cuda = cuda
        cupy.asarray = self.upload = Mock(side_effect=self._upload)
        nvmath = ModuleType('nvmath')
        fft = ModuleType('nvmath.fft')
        fft.DeviceCallable = DeviceCallable
        nvmath.fft = fft
        bindings = ModuleType('nvmath.bindings')
        bindings.cufft = SimpleNamespace(get_version=self.version)
        nvmath.bindings = bindings
        self.modules = patch.dict(sys.modules, {
            'cupy': cupy, 'cupy.cuda': cuda, 'nvmath': nvmath,
            'nvmath.fft': fft, 'nvmath.bindings': bindings,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        spec = importlib.util.spec_from_file_location(
            '_nvmath_scaling_test', SOURCE)
        self.scaling = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.scaling)

    def _upload(self, host, *, blocking):
        self.assertTrue(blocking)
        self.uploads.append(host.copy())
        self.next_ptr += 1

        class Scalar:
            pass

        scalar = Scalar()
        scalar.data = SimpleNamespace(ptr=self.next_ptr)
        self.scalar_refs.append(weakref.ref(scalar))
        return scalar

    def get(self, divisor=32, dtype='complex64', stream=7):
        return self.scaling.get_scaling_epilog(dtype, divisor, stream)

    def test_matching_requests_reuse_actual_descriptor_and_pointer(self):
        first = self.get()
        self.assertIs(first, self.get())
        self.assertEqual(first.data, first._cupy_scalar.data.ptr)
        self.assertEqual(first._cupy_divisor, numpy.float32(32))
        self.assertEqual(self.compiler.call_count, 1)
        self.assertEqual(self.upload.call_count, 1)
        self.assertEqual(self.device.call_count, 1)
        self.assertEqual(self.version.call_count, 1)

    def test_different_divisors_reuse_code_and_bind_immutable_scalars(self):
        first = self.get(32)
        second = self.get(64)
        self.assertIsNot(first, second)
        self.assertNotEqual(first.data, second.data)
        self.assertIs(first.ltoir, second.ltoir)
        numpy.testing.assert_array_equal(self.uploads[0], [32])
        numpy.testing.assert_array_equal(self.uploads[1], [64])
        self.assertEqual(self.compiler.call_count, 1)

    def test_unit_divisor_avoids_compiler_device_and_allocation(self):
        self.assertIsNone(self.get(1))
        self.assertIsNone(self.get(1 + 1e-9))  # rounds to float32 one
        self.compiler.assert_not_called()
        self.upload.assert_not_called()
        self.device.assert_not_called()
        self.version.assert_not_called()

    def test_output_types_use_real_component_precision(self):
        for dtype, real_type in (
                ('float32', numpy.float32), ('complex64', numpy.float32),
                ('float64', numpy.float64), ('complex128', numpy.float64)):
            with self.subTest(dtype=dtype):
                callback = self.get(numpy.sqrt(32), dtype)
                self.assertIsInstance(callback._cupy_divisor, real_type)
                self.assertEqual(self.uploads[-1].dtype,
                                 numpy.dtype(real_type))

    def test_divisors_that_round_identically_share_binding(self):
        first = self.get(32)
        self.assertIs(first, self.get(32 + 1e-8))
        self.assertEqual(self.upload.call_count, 1)

    def test_weak_interning_does_not_retain_device_allocations(self):
        callback = self.get()
        ref = weakref.ref(callback)
        owner = SimpleNamespace(epilog=callback)
        del callback
        gc.collect()
        self.assertIsNotNone(ref())
        self.assertIs(owner.epilog, self.get())
        del owner
        gc.collect()
        self.assertIsNone(ref())
        self.assertIsNone(self.scalar_refs[0]())
        self.assertEqual(len(self.scaling._bindings.cache), 0)
        # A plan eviction may require another scalar upload, not compilation.
        callback = self.get()
        self.assertEqual(self.upload.call_count, 2)
        self.assertEqual(self.compiler.call_count, 1)

    def test_streams_and_devices_have_distinct_bindings_but_share_ir(self):
        first = self.get()
        other_stream = self.get(stream=8)
        self.device_id = 1
        other_device = self.get()
        self.assertEqual(len({first.data, other_stream.data,
                              other_device.data}), 3)
        self.assertEqual(self.compiler.call_count, 1)
        self.assertIs(other_device, self.get())
        self.device_id = 0
        self.assertIs(first, self.get())

    def test_threads_have_distinct_bindings_but_share_ir(self):
        main = self.get()
        with ThreadPoolExecutor(max_workers=1) as executor:
            other = executor.submit(self.get).result()
            repeated = executor.submit(self.get).result()
        self.assertIs(other, repeated)
        self.assertIsNot(main, other)
        self.assertNotEqual(main.data, other.data)
        self.assertEqual(self.compiler.call_count, 1)

    def test_architecture_and_offset_abi_distinguish_code_cache(self):
        compile_code = self.scaling._compile_code
        first = compile_code('complex64', '100', True)
        self.assertIs(first, compile_code('complex64', '100', True))
        compile_code('complex64', '90', True)
        compile_code('complex64', '100', False)
        self.assertEqual(self.compiler.call_count, 3)
        self.assertIn('-DCUPY_FFT_CALLBACK_OFFSET_UINT64=0',
                      self.compiler.call_args.args[1])
        self.assertTrue(self.compiler.call_args.kwargs['to_ltoir'])

    def test_code_cache_is_bounded(self):
        for index in range(130):
            self.scaling._compile_code('complex64', str(index), True)
        info = self.scaling._compile_code.cache_info()
        self.assertEqual(info.maxsize, 128)
        self.assertEqual(info.currsize, 128)

    def test_compilation_failure_does_not_poison_binding_cache(self):
        self.compiler.side_effect = RuntimeError('compile failed')
        with self.assertRaisesRegex(RuntimeError, 'compile failed'):
            self.get()
        self.upload.assert_not_called()
        self.assertEqual(len(self.scaling._bindings.cache), 0)
        self.compiler.side_effect = None
        self.compiler.return_value = b'valid-ir'
        callback = self.get()
        self.assertEqual(callback.ltoir, b'valid-ir')
        self.assertEqual(self.compiler.call_count, 2)

    def test_upload_failure_does_not_poison_binding_cache(self):
        self.upload.side_effect = RuntimeError('upload failed')
        with self.assertRaisesRegex(RuntimeError, 'upload failed'):
            self.get()
        self.assertEqual(len(self.scaling._bindings.cache), 0)
        self.upload.side_effect = self._upload
        callback = self.get()
        self.assertIsNotNone(callback)
        self.assertEqual(self.compiler.call_count, 1)

    def test_invalid_values_fail_before_device_work(self):
        for value in (0, -1, numpy.inf, numpy.nan):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.get(value)
        with self.assertRaisesRegex(ValueError, 'Unsupported callback dtype'):
            self.get(dtype='float16')
        self.compiler.assert_not_called()
        self.upload.assert_not_called()

    def test_external_cuda_source_is_part_of_distribution(self):
        root = SOURCE.parents[2]
        self.assertTrue(SOURCE.with_suffix('.cu').is_file())
        self.assertIn("'cupy/fft/_nvmath_scaling.cu'",
                      (root / 'setup.py').read_text())


if __name__ == '__main__':
    unittest.main()
