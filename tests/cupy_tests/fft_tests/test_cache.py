from __future__ import annotations

import contextlib
import io
import queue
import threading
import unittest

import pytest

import cupy
from cupy import testing
from cupy.cuda import cufft
from cupy.cuda import device
from cupy.cuda import driver
from cupy.cuda import runtime
from cupy.fft import config

from .test_fft import multi_gpu_config


def intercept_stdout(func):
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        func()
        stdout = buf.getvalue()
    return stdout


n_devices = runtime.getDeviceCount()
is_cuda_python = driver._is_cuda_python()


class _ExplicitFreePlan:
    gpus = None
    work_area = None

    def __init__(self, fail_on_free=False):
        self.free_count = 0
        self.fail_on_free = fail_on_free

    def free(self):
        self.free_count += 1
        if self.fail_on_free:
            raise RuntimeError('free failed')

    def _cupy_fft_cache_cleanup(self):
        self.free()


class TestPlanCache(unittest.TestCase):
    @contextlib.contextmanager
    @staticmethod
    def prepare_and_restore_caches():
        old_sizes = []
        for i in range(n_devices):
            with device.Device(i):
                cache = config.get_plan_cache()
                old_sizes.append(cache.get_size())
                cache.clear()
                cache.set_memsize(-1)
                cache.set_size(2)

        try:
            yield
        finally:
            for i in range(n_devices):
                with device.Device(i):
                    cache = config.get_plan_cache()
                    cache.clear()
                    cache.set_size(old_sizes[i])
                    cache.set_memsize(-1)

    @prepare_and_restore_caches()
    def test_LRU_cache1(self):
        # test if insertion and clean-up works
        cache = config.get_plan_cache()
        assert cache.get_curr_size() == 0 <= cache.get_size()

        a = testing.shaped_random((10,), cupy, cupy.float32)
        cupy.fft.fft(a)
        assert cache.get_curr_size() == 1 <= cache.get_size()

        cache.clear()
        assert cache.get_curr_size() == 0 <= cache.get_size()

    @prepare_and_restore_caches()
    def test_LRU_cache2(self):
        # test if plan is reused
        cache = config.get_plan_cache()
        assert cache.get_curr_size() == 0 <= cache.get_size()

        # run once and fetch the cached plan
        a = testing.shaped_random((10,), cupy, cupy.float32)
        cupy.fft.fft(a)
        assert cache.get_curr_size() == 1 <= cache.get_size()
        iterator = iter(cache)
        plan0 = next(iterator)[1].plan

        # repeat
        cupy.fft.fft(a)
        assert cache.get_curr_size() == 1 <= cache.get_size()
        iterator = iter(cache)
        plan1 = next(iterator)[1].plan

        # we should get the same plan
        assert plan0 is plan1

    @prepare_and_restore_caches()
    def test_LRU_cache3(self):
        # test if cache size is limited
        cache = config.get_plan_cache()
        assert cache.get_curr_size() == 0 <= cache.get_size()

        # run once and fetch the cached plan
        a = testing.shaped_random((10,), cupy, cupy.float32)
        cupy.fft.fft(a)
        assert cache.get_curr_size() == 1 <= cache.get_size()
        iterator = iter(cache)
        plan = next(iterator)[1].plan

        # run another two FFTs with different sizes so that the first
        # plan is discarded from the cache
        a = testing.shaped_random((20,), cupy, cupy.float32)
        cupy.fft.fft(a)
        assert cache.get_curr_size() == 2 <= cache.get_size()
        a = testing.shaped_random((30,), cupy, cupy.float32)
        cupy.fft.fft(a)
        assert cache.get_curr_size() == 2 <= cache.get_size()

        # check if the first plan is indeed not cached
        for _, node in cache:
            assert plan is not node.plan

    @prepare_and_restore_caches()
    def test_explicit_free_plan_lifecycle(self):
        cache = config.get_plan_cache()
        plan1 = _ExplicitFreePlan()
        plan2 = _ExplicitFreePlan()
        plan3 = _ExplicitFreePlan()

        cache[('explicit-free', 1)] = plan1
        cache[('explicit-free', 2)] = plan2
        assert cache.get_curr_memsize() == 0
        assert 'plan type: nvmath FFT' in str(cache)

        # Reassigning the same object only refreshes its LRU position.
        cache[('explicit-free', 2)] = plan2
        assert plan2.free_count == 0

        # Inserting a third plan evicts and explicitly frees the LRU plan.
        cache[('explicit-free', 3)] = plan3
        assert plan1.free_count == 1
        assert plan2.free_count == 0
        assert plan3.free_count == 0

        del cache[('explicit-free', 2)]
        assert plan2.free_count == 1

        plan4 = _ExplicitFreePlan()
        cache[('explicit-free', 3)] = plan4
        assert plan3.free_count == 1

        cache.clear()
        assert plan4.free_count == 1

    @prepare_and_restore_caches()
    def test_zero_memsize_frees_zero_weight_plan(self):
        cache = config.get_plan_cache()
        plan = _ExplicitFreePlan()
        cache[('explicit-free',)] = plan

        cache.set_memsize(0)

        assert plan.free_count == 1
        assert cache.get_curr_size() == 0
        assert cache.get_curr_memsize() == 0

    @prepare_and_restore_caches()
    def test_clear_recovers_from_explicit_free_failure(self):
        cache = config.get_plan_cache()
        bad_plan = _ExplicitFreePlan(fail_on_free=True)
        good_plan = _ExplicitFreePlan()
        cache[('explicit-free', 1)] = bad_plan
        cache[('explicit-free', 2)] = good_plan

        with pytest.raises(RuntimeError, match='free failed'):
            cache.clear()

        assert bad_plan.free_count == 1
        assert good_plan.free_count == 1
        assert cache.get_curr_size() == 0
        assert cache.get_curr_memsize() == 0

    @prepare_and_restore_caches()
    def test_LRU_cache4(self):
        # test if fetching the plan will reorder it to the top
        cache = config.get_plan_cache()
        assert cache.get_curr_size() == 0 <= cache.get_size()

        # this creates a Plan1d
        a = testing.shaped_random((10,), cupy, cupy.float32)
        cupy.fft.fft(a)
        assert cache.get_curr_size() == 1 <= cache.get_size()

        # this creates a PlanNd
        a = testing.shaped_random((10, 20), cupy, cupy.float32)
        cupy.fft.fftn(a)
        assert cache.get_curr_size() == 2 <= cache.get_size()

        # The first in the cache is the most recently used one;
        # using an iterator to access the linked list guarantees that
        # we don't alter the cache order
        iterator = iter(cache)
        assert isinstance(next(iterator)[1].plan, cufft.PlanNd)
        assert isinstance(next(iterator)[1].plan, cufft.Plan1d)
        with pytest.raises(StopIteration):
            next(iterator)

        # this brings Plan1d to the top
        a = testing.shaped_random((10,), cupy, cupy.float32)
        cupy.fft.fft(a)
        assert cache.get_curr_size() == 2 <= cache.get_size()
        iterator = iter(cache)
        assert isinstance(next(iterator)[1].plan, cufft.Plan1d)
        assert isinstance(next(iterator)[1].plan, cufft.PlanNd)
        with pytest.raises(StopIteration):
            next(iterator)

        # An LRU cache guarantees that such a silly operation never
        # raises StopIteration
        iterator = iter(cache)
        for i in range(100):
            cache[next(iterator)[0]]

    @testing.multi_gpu(2)
    @prepare_and_restore_caches()
    @pytest.mark.thread_unsafe(reason="intercepts stdout")
    def test_LRU_cache5(self):
        # test if the LRU cache is thread-local

        def init_caches(gpus):
            for i in gpus:
                with device.Device(i):
                    config.get_plan_cache()

        # Testing in the current thread: in setUp() we ensure all caches
        # are initialized
        stdout = intercept_stdout(config.show_plan_cache_info)
        assert 'uninitialized' not in stdout

        def thread_show_plan_cache_info(queue):
            # allow output from another thread to be accessed by the
            # main thread
            cupy.cuda.Device().use()
            stdout = intercept_stdout(config.show_plan_cache_info)
            queue.put(stdout)

        # When starting a new thread, the cache is uninitialized there
        # (for both devices)
        q = queue.Queue()
        thread = threading.Thread(target=thread_show_plan_cache_info,
                                  args=(q,))
        thread.start()
        thread.join()
        stdout = q.get()
        assert stdout.count('uninitialized') == n_devices

        def thread_init_caches(gpus, queue):
            cupy.cuda.Device().use()
            init_caches(gpus)
            thread_show_plan_cache_info(queue)

        # Now let's try initializing device 0 on another thread
        thread = threading.Thread(target=thread_init_caches,
                                  args=([0], q,))
        thread.start()
        thread.join()
        stdout = q.get()
        assert stdout.count('uninitialized') == n_devices - 1

        # ...and this time both devices
        thread = threading.Thread(target=thread_init_caches,
                                  args=([0, 1], q,))
        thread.start()
        thread.join()
        stdout = q.get()
        assert stdout.count('uninitialized') == n_devices - 2

    @testing.multi_gpu(2)
    @prepare_and_restore_caches()
    def test_LRU_cache6(self, gpus=None):
        # test if each device has a separate cache
        with device.Device(0):
            cache0 = config.get_plan_cache()
        with device.Device(1):
            cache1 = config.get_plan_cache()

        # ensure a fresh state
        assert cache0.get_curr_size() == 0 <= cache0.get_size()
        assert cache1.get_curr_size() == 0 <= cache1.get_size()

        # do some computation on GPU 0
        with device.Device(0):
            a = testing.shaped_random((10,), cupy, cupy.float32)
            cupy.fft.fft(a)
        assert cache0.get_curr_size() == 1 <= cache0.get_size()
        assert cache1.get_curr_size() == 0 <= cache1.get_size()

        # do some computation on GPU 1
        with device.Device(1):
            c = testing.shaped_random((16,), cupy, cupy.float64)
            cupy.fft.fft(c)
        assert cache0.get_curr_size() == 1 <= cache0.get_size()
        assert cache1.get_curr_size() == 1 <= cache1.get_size()

        # reset device 0
        cache0.clear()
        assert cache0.get_curr_size() == 0 <= cache0.get_size()
        assert cache1.get_curr_size() == 1 <= cache1.get_size()

        # reset device 1
        cache1.clear()
        assert cache0.get_curr_size() == 0 <= cache0.get_size()
        assert cache1.get_curr_size() == 0 <= cache1.get_size()

    @testing.multi_gpu(2)
    @pytest.mark.skipif(runtime.is_hip,
                        reason="hipFFT doesn't support multi-GPU")
    @prepare_and_restore_caches()
    def test_LRU_cache7(self, gpus=None):
        # test accessing a multi-GPU plan
        with device.Device(0):
            cache0 = config.get_plan_cache()
        with device.Device(1):
            cache1 = config.get_plan_cache()

        # ensure a fresh state
        assert cache0.get_curr_size() == 0 <= cache0.get_size()
        assert cache1.get_curr_size() == 0 <= cache1.get_size()

        # do some computation on GPU 0
        with device.Device(0):
            a = testing.shaped_random((10,), cupy, cupy.float32)
            cupy.fft.fft(a)
        assert cache0.get_curr_size() == 1 <= cache0.get_size()
        assert cache1.get_curr_size() == 0 <= cache1.get_size()

        # do a multi-GPU FFT
        config.use_multi_gpus = True
        config.set_cufft_gpus([0, 1])
        c = testing.shaped_random((128,), cupy, cupy.complex64)
        cupy.fft.fft(c)
        assert cache0.get_curr_size() == 2 <= cache0.get_size()
        assert cache1.get_curr_size() == 1 <= cache1.get_size()

        # check both devices' caches see the same multi-GPU plan
        plan0 = next(iter(cache0))[1].plan
        plan1 = next(iter(cache1))[1].plan
        assert plan0 is plan1

        # reset
        config.use_multi_gpus = False
        config._device = None

        # do some computation on GPU 1
        with device.Device(1):
            e = testing.shaped_random((20,), cupy, cupy.complex128)
            cupy.fft.fft(e)
        assert cache0.get_curr_size() == 2 <= cache0.get_size()
        assert cache1.get_curr_size() == 2 <= cache1.get_size()

        # by this time, the multi-GPU plan remains the most recently
        # used one on GPU 0, but not on GPU 1
        assert plan0 is next(iter(cache0))[1].plan
        assert plan1 is not next(iter(cache1))[1].plan

        # now use it again to make it the most recent
        config.use_multi_gpus = True
        config.set_cufft_gpus([0, 1])
        c = testing.shaped_random((128,), cupy, cupy.complex64)
        cupy.fft.fft(c)
        assert cache0.get_curr_size() == 2 <= cache0.get_size()
        assert cache1.get_curr_size() == 2 <= cache1.get_size()
        assert plan0 is next(iter(cache0))[1].plan
        assert plan1 is next(iter(cache1))[1].plan
        # reset
        config.use_multi_gpus = False
        config._device = None

        # Do 2 more different FFTs on one of the devices, and the
        # multi-GPU plan would be discarded from both caches
        with device.Device(1):
            x = testing.shaped_random((30,), cupy, cupy.complex128)
            cupy.fft.fft(x)
            y = testing.shaped_random((40, 40), cupy, cupy.complex64)
            cupy.fft.fftn(y)
        for _, node in cache0:
            assert plan0 is not node.plan
        for _, node in cache1:
            assert plan1 is not node.plan
        assert cache0.get_curr_size() == 1 <= cache0.get_size()
        assert cache1.get_curr_size() == 2 <= cache1.get_size()

    @prepare_and_restore_caches()
    def test_LRU_cache8(self):
        # test if Plan1d and PlanNd can coexist in the same cache
        cache = config.get_plan_cache()
        assert cache.get_curr_size() == 0 <= cache.get_size()

        # do a 1D FFT
        a = testing.shaped_random((10,), cupy, cupy.float32)
        cupy.fft.fft(a)
        assert cache.get_curr_size() == 1 <= cache.get_size()
        assert isinstance(next(iter(cache))[1].plan, cufft.Plan1d)

        # then a 3D FFT
        a = testing.shaped_random((8, 8, 8), cupy, cupy.complex128)
        cupy.fft.fftn(a)
        assert cache.get_curr_size() == 2 <= cache.get_size()
        iterator = iter(cache)

        # The N-D entry is most recent, followed by the 1-D entry.
        plan = next(iterator)[1].plan
        if is_cuda_python:
            assert callable(plan._cupy_fft_cache_cleanup)
        else:
            assert isinstance(plan, cufft.PlanNd)
        assert isinstance(next(iterator)[1].plan, cufft.Plan1d)

    @prepare_and_restore_caches()
    def test_LRU_cache9(self):
        # test if memsizes in the cache adds up
        cache = config.get_plan_cache()
        assert cache.get_curr_size() == 0 <= cache.get_size()

        memsize = 0
        a = testing.shaped_random((10,), cupy, cupy.float32)
        cupy.fft.fft(a)
        assert cache.get_curr_size() == 1 <= cache.get_size()
        memsize += next(iter(cache))[1].memsize

        a = testing.shaped_random((48,), cupy, cupy.complex64)
        cupy.fft.fft(a)
        assert cache.get_curr_size() == 2 <= cache.get_size()
        memsize += next(iter(cache))[1].memsize

        assert memsize == cache.get_curr_memsize()

    @prepare_and_restore_caches()
    @pytest.mark.thread_unsafe(reason="intercepts stdout")
    def test_LRU_cache10(self):
        # test if deletion works and if show_info() is consistent with data
        cache = config.get_plan_cache()
        assert cache.get_curr_size() == 0 <= cache.get_size()

        curr_size = 0
        size = 2
        curr_memsize = 0
        memsize = '(unlimited)'  # default

        a = testing.shaped_random((16, 16), cupy, cupy.float32)
        cupy.fft.fft2(a)
        assert cache.get_curr_size() == 1 <= cache.get_size()
        node1 = next(iter(cache))[1]
        curr_size += 1
        curr_memsize += node1.memsize
        stdout = intercept_stdout(cache.show_info)
        assert '{} / {} (counts)'.format(curr_size, size) in stdout
        assert '{} / {} (bytes)'.format(curr_memsize, memsize) in stdout
        assert str(node1) in stdout

        a = testing.shaped_random((1024,), cupy, cupy.complex64)
        cupy.fft.ifft(a)
        assert cache.get_curr_size() == 2 <= cache.get_size()
        node2 = next(iter(cache))[1]
        curr_size += 1
        curr_memsize += node2.memsize
        stdout = intercept_stdout(cache.show_info)
        assert '{} / {} (counts)'.format(curr_size, size) in stdout
        assert '{} / {} (bytes)'.format(curr_memsize, memsize) in stdout
        assert str(node2) + '\n' + str(node1) in stdout

        # test deletion
        key = node2.key
        del cache[key]
        assert cache.get_curr_size() == 1 <= cache.get_size()
        curr_size -= 1
        curr_memsize -= node2.memsize
        stdout = intercept_stdout(cache.show_info)
        assert '{} / {} (counts)'.format(curr_size, size) in stdout
        assert '{} / {} (bytes)'.format(curr_memsize, memsize) in stdout
        assert str(node2) not in stdout

    @multi_gpu_config(gpu_configs=[[0, 1], [1, 0]])
    @testing.multi_gpu(2)
    @pytest.mark.skipif(runtime.is_hip,
                        reason="hipFFT doesn't support multi-GPU")
    @prepare_and_restore_caches()
    def test_LRU_cache11(self):
        # test if collectively deleting a multi-GPU plan works
        with device.Device(0):
            cache0 = config.get_plan_cache()
        with device.Device(1):
            cache1 = config.get_plan_cache()

        # ensure a fresh state
        assert cache0.get_curr_size() == 0 <= cache0.get_size()
        assert cache1.get_curr_size() == 0 <= cache1.get_size()

        # do a multi-GPU FFT
        c = testing.shaped_random((128,), cupy, cupy.complex64)
        cupy.fft.fft(c)
        assert cache0.get_curr_size() == 1 <= cache0.get_size()
        assert cache1.get_curr_size() == 1 <= cache1.get_size()

        node0 = next(iter(cache0))[1]
        node1 = next(iter(cache1))[1]
        assert node0.key == node1.key
        assert node0.plan is node1.plan
        assert cache0.get_curr_memsize() == node0.memsize > 0
        assert cache1.get_curr_memsize() == node1.memsize > 0

        # delete
        del cache0[node0.key]
        assert cache0.get_curr_size() == 0 <= cache0.get_size()
        assert cache1.get_curr_size() == 0 <= cache1.get_size()
        assert cache0.get_curr_memsize() == 0
        assert cache1.get_curr_memsize() == 0

    @multi_gpu_config(gpu_configs=[[0, 1], [1, 0]])
    @testing.multi_gpu(2)
    @pytest.mark.skipif(runtime.is_hip,
                        reason="hipFFT doesn't support multi-GPU")
    @prepare_and_restore_caches()
    def test_LRU_cache12(self):
        # test if an error is raise when one of the caches is unable
        # to fit it a multi-GPU plan
        with device.Device(0):
            cache0 = config.get_plan_cache()
        with device.Device(1):
            cache1 = config.get_plan_cache()

        # ensure a fresh state
        assert cache0.get_curr_size() == 0 <= cache0.get_size()
        assert cache1.get_curr_size() == 0 <= cache1.get_size()

        # make it impossible to cache
        cache1.set_memsize(1)

        # do a multi-GPU FFT
        with pytest.raises(RuntimeError) as e:
            c = testing.shaped_random((128,), cupy, cupy.complex64)
            cupy.fft.fft(c)
        assert 'plan memsize is too large for device 1' in str(e.value)
        assert cache0.get_curr_size() == 0 <= cache0.get_size()
        assert cache1.get_curr_size() == 0 <= cache1.get_size()

    @unittest.skipIf(runtime.is_hip, "rocFFT has different plan sizes")
    @unittest.skipIf(
        is_cuda_python, "nvmath plans release and do not account workspaces")
    @unittest.skipIf(runtime.runtimeGetVersion() >= 11080,
                     "CUDA 11.8 has different plan size")
    @prepare_and_restore_caches()
    def test_LRU_cache13(self):
        # test if plan insertion respect the memory size limit
        cache = config.get_plan_cache()
        cache.set_memsize(1024)

        # ensure a fresh state
        assert cache.get_curr_size() == 0 <= cache.get_size()

        # On CUDA 10.0 + sm75, this generates a plan of size 1024 bytes
        a = testing.shaped_random((128,), cupy, cupy.complex64)
        cupy.fft.ifft(a)
        assert cache.get_curr_size() == 1 <= cache.get_size()
        assert cache.get_curr_memsize() == 1024 == cache.get_memsize()

        # a second plan (of same size) is generated, but the cache is full,
        # so the first plan is evicted
        a = testing.shaped_random((64,), cupy, cupy.complex128)
        cupy.fft.ifft(a)
        assert cache.get_curr_size() == 1 <= cache.get_size()
        assert cache.get_curr_memsize() == 1024 == cache.get_memsize()
        plan = next(iter(cache))[1].plan

        # this plan is twice as large, so won't fit in
        a = testing.shaped_random((128,), cupy, cupy.complex128)
        with pytest.raises(RuntimeError) as e:
            cupy.fft.ifft(a)
        assert 'memsize is too large' in str(e.value)
        # the cache remains intact
        assert cache.get_curr_size() == 1 <= cache.get_size()
        assert cache.get_curr_memsize() == 1024 == cache.get_memsize()
        plan1 = next(iter(cache))[1].plan
        assert plan1 is plan

        # double the cache size would make the plan just fit (and evict
        # the existing one)
        cache.set_memsize(2048)
        cupy.fft.ifft(a)
        assert cache.get_curr_size() == 1 <= cache.get_size()
        assert cache.get_curr_memsize() == 2048 == cache.get_memsize()
        plan2 = next(iter(cache))[1].plan
        assert plan2 is not plan


@pytest.mark.skipif(
    not is_cuda_python, reason='requires a CUDA-Python build')
class TestNvmathFFTCache:

    @pytest.fixture(autouse=True)
    def prepare_cache(self):
        cache = config.get_plan_cache()
        old_size = cache.get_size()
        old_memsize = cache.get_memsize()
        cache.clear()
        cache.set_size(16)
        cache.set_memsize(-1)
        try:
            yield
        finally:
            cache.clear()
            cache.set_size(old_size)
            cache.set_memsize(old_memsize)

    def test_cached_lifecycle(self, monkeypatch):
        import nvmath.fft

        execute_release_workspace = []
        reset_count = 0
        release_count = 0
        free_count = 0

        original_execute = nvmath.fft.FFT.execute
        original_free = nvmath.fft.FFT.free
        original_release = nvmath.fft.FFT.release_operand
        original_reset = nvmath.fft.FFT.reset_operand_unchecked

        def execute(self, *args, **kwargs):
            execute_release_workspace.append(kwargs.get('release_workspace'))
            return original_execute(self, *args, **kwargs)

        def release(self):
            nonlocal release_count
            release_count += 1
            return original_release(self)

        def free(self):
            nonlocal free_count
            free_count += 1
            return original_free(self)

        def reset(self, *args, **kwargs):
            nonlocal reset_count
            reset_count += 1
            return original_reset(self, *args, **kwargs)

        monkeypatch.setattr(nvmath.fft.FFT, 'execute', execute)
        monkeypatch.setattr(nvmath.fft.FFT, 'free', free)
        monkeypatch.setattr(nvmath.fft.FFT, 'release_operand', release)
        monkeypatch.setattr(
            nvmath.fft.FFT, 'reset_operand_unchecked', reset)

        a = testing.shaped_random((3, 16), cupy, cupy.complex64)
        cupy.fft.fft(a, axis=-1)

        cache = config.get_plan_cache()
        assert cache.get_curr_size() == 1
        assert cache.get_curr_memsize() == 0
        _, node = next(iter(cache))
        plan = node.plan

        b = testing.shaped_random((3, 16), cupy, cupy.complex64)
        cupy.fft.fft(b, axis=-1)
        assert next(iter(cache))[1].plan is plan
        assert reset_count == 1
        assert release_count == 2
        assert execute_release_workspace == [True, True]

        cache.clear()
        assert free_count == 1

    def test_cache_is_separated_by_stream(self):
        stream1 = cupy.cuda.Stream(non_blocking=True)
        stream2 = cupy.cuda.Stream(non_blocking=True)

        cache = config.get_plan_cache()
        assert cache.get_curr_size() == 0

        with stream1:
            a1 = testing.shaped_random((16,), cupy, cupy.complex64)
            cupy.fft.fft(a1)
        with stream2:
            a2 = testing.shaped_random((16,), cupy, cupy.complex64)
            cupy.fft.fft(a2)

        assert cache.get_curr_size() == 2

    def test_disabled_cache_frees_ephemeral_plan(self, monkeypatch):
        import nvmath.fft

        free_count = 0
        original_free = nvmath.fft.FFT.free

        def free(self):
            nonlocal free_count
            free_count += 1
            return original_free(self)

        monkeypatch.setattr(nvmath.fft.FFT, 'free', free)
        cache = config.get_plan_cache()
        cache.set_size(0)

        a = testing.shaped_random((16,), cupy, cupy.complex64)
        cupy.fft.fft(a)

        assert free_count == 1
        assert cache.get_curr_size() == 0
