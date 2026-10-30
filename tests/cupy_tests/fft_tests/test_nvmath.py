from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import cupy
from cupy import testing
from cupy.cuda import driver
from cupy.fft import config

is_cuda_python = driver._is_cuda_python()


@pytest.fixture
def adapter():
    pytest.importorskip('nvmath.fft')
    from cupy.fft import _fft_nvmath
    return _fft_nvmath


@pytest.mark.parametrize('ndim,axes,expected', [
    (1, None, (0,)),
    (2, (0, 1), (0, 1)),
    (3, (-3, -2, -1), (0, 1, 2)),
    (2, [-2, -1], (0, 1)),
    (3, -1, (2,)),
    (3, (2, 0), (2, 0)),
    (3, (), ()),
    (3, (0, -3), None),
    (2, (2,), None),
    (2, (-3,), None),
    (2, 2, None),
    (4, None, None),
    (4, (0, 1, 2, 3), None),
    (0, None, ()),
    (0, 0, None),
])
def test_normalize_axes(adapter, ndim, axes, expected):
    assert adapter._normalize_axes(ndim, axes) == expected


@pytest.mark.parametrize('shape,axes,kind,policy,expected', [
    ((4, 8, 16), (1, 2), 'C2C', 'N', 128),
    ((4, 8, 16), (0, 2), 'R2C', 'N', 64),
    ((4, 8, 9), (1, 2), 'C2R', 'N', 128),
    ((4, 8, 9), (2, 0), 'C2R', 'N', 54),
    ((4, 8, 9), (1, 2), 'C2R', 'sqrtN', np.sqrt(128)),
    ((4, 8, 9), (1, 2), 'C2R', None, 1),
])
def test_normalization_divisor(adapter, shape, axes, kind, policy, expected):
    operand = SimpleNamespace(shape=shape)
    assert adapter._normalization_divisor(
        operand, axes, kind, policy) == expected


@pytest.mark.parametrize('version,expected', [
    (11299, False), (11300, True), (12040, True),
])
def test_lto_support_follows_loaded_cufft_version(
        adapter, monkeypatch, version, expected):
    scaling = adapter._nvmath_scaling
    monkeypatch.setattr(scaling.cufft, 'get_version',
                        Mock(return_value=version))
    scaling.supports_lto_callbacks.cache_clear()
    try:
        assert scaling.supports_lto_callbacks() is expected
    finally:
        scaling.supports_lto_callbacks.cache_clear()


# (function, input shape, input dtype, keyword arguments)
_NORMALIZED_FFTS = [
    ('ifft', (4, 16), cupy.complex64, {}),
    ('fft', (4, 16), cupy.complex128, {}),
    ('ifftn', (4, 6, 8), cupy.complex64, {'axes': [0, 2]}),
    ('rfft', (4, 16), cupy.float32, {}),
    ('irfft', (4, 9), cupy.complex64, {}),
    ('irfftn', (4, 6, 9), cupy.complex128, {'axes': (1, 2)}),
]


@pytest.mark.skipif(
    not is_cuda_python, reason='requires a CUDA-Python build')
class TestNvmathNormalization:

    @pytest.fixture(autouse=True)
    def prepare_cache(self, adapter):
        self.adapter = adapter
        self.scaling = adapter._nvmath_scaling
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

    def _cached_plans(self):
        return [node.plan for _, node in config.get_plan_cache()]

    def _check(self, name, shape, dtype, kwargs, norm):
        a = testing.shaped_random(shape, cupy, dtype)
        out = getattr(cupy.fft, name)(a, norm=norm, **kwargs)
        expected = getattr(np.fft, name)(a.get(), norm=norm, **kwargs)
        tol = 1e-4 if dtype in (cupy.complex64, cupy.float32) else 1e-10
        testing.assert_allclose(out, expected, rtol=tol, atol=tol)

    @pytest.mark.parametrize('norm', [None, 'ortho', 'forward'])
    @pytest.mark.parametrize('name,shape,dtype,kwargs', _NORMALIZED_FFTS)
    def test_fused_normalization(self, name, shape, dtype, kwargs, norm):
        if not self.scaling.supports_lto_callbacks():
            pytest.skip('cuFFT does not support LTO callbacks')
        self._check(name, shape, dtype, kwargs, norm)
        plans = self._cached_plans()
        assert len(plans) == 1
        assert isinstance(plans[0], self.adapter.CachedFFT)
        scaled = (norm == 'ortho'
                  or (norm == 'forward') == name.startswith(('fft', 'rfft')))
        assert (plans[0]._normalization_epilog is not None) == scaled

    @pytest.mark.parametrize('norm', [None, 'ortho', 'forward'])
    @pytest.mark.parametrize('name,shape,dtype,kwargs', _NORMALIZED_FFTS)
    def test_unfused_normalization_on_older_cufft(
            self, monkeypatch, name, shape, dtype, kwargs, norm):
        monkeypatch.setattr(self.scaling, 'supports_lto_callbacks',
                            lambda: False)
        self._check(name, shape, dtype, kwargs, norm)
        plans = self._cached_plans()
        assert len(plans) == 1
        assert isinstance(plans[0], self.adapter.CachedFFT)
        assert plans[0]._normalization_epilog is None

    def test_unfused_normalization_shares_unscaled_plan(self, monkeypatch):
        monkeypatch.setattr(self.scaling, 'supports_lto_callbacks',
                            lambda: False)
        for norm in (None, 'ortho', 'forward'):
            self._check('ifft', (4, 16), cupy.complex64, {}, norm)
        assert len(self._cached_plans()) == 1

    def test_epilog_failure_falls_back_to_native(self, monkeypatch):
        error = RuntimeError('compilation failed')
        monkeypatch.setattr(self.scaling, 'supports_lto_callbacks',
                            lambda: True)
        monkeypatch.setattr(self.scaling, 'get_scaling_epilog',
                            Mock(side_effect=error))
        with pytest.warns(RuntimeWarning, match='compilation failed'):
            self._check('ifft', (4, 16), cupy.complex64, {}, None)
        plans = self._cached_plans()
        assert not any(
            isinstance(plan, self.adapter.CachedFFT) for plan in plans)

    @pytest.mark.parametrize('dtype,real_dtype', [
        ('float32', np.float32), ('complex64', np.float32),
        ('float64', np.float64), ('complex128', np.float64),
    ])
    def test_epilogs_share_ltoir_and_own_scalars(self, dtype, real_dtype):
        device_id = cupy.cuda.Device().id
        first = self.scaling.get_scaling_epilog(dtype, 32, device_id)
        second = self.scaling.get_scaling_epilog(dtype, 128, device_id)
        assert first.ltoir is second.ltoir
        assert first.data != second.data
        for epilog, value in ((first, 32), (second, 128)):
            scalar = epilog._cupy_scalar
            assert epilog.data == scalar.data.ptr
            assert scalar.dtype == real_dtype
            assert scalar.get().tolist() == [value]

    @testing.multi_gpu(2)
    def test_epilog_scalar_lives_on_requested_device(self):
        with cupy.cuda.Device(0):
            epilog = self.scaling.get_scaling_epilog('complex64', 32, 1)
            assert cupy.cuda.Device().id == 0
        assert epilog._cupy_scalar.device.id == 1
