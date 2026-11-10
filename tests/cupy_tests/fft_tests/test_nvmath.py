from __future__ import annotations

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


@pytest.mark.parametrize('shape,axes,kind,scaling,expected', [
    ((4, 8, 16), (1, 2), 'C2C', 'N', 128),
    ((4, 8, 16), (0, 2), 'R2C', 'N', 64),
    ((4, 8, 9), (1, 2), 'C2R', 'N', 128),
    ((4, 8, 9), (2, 0), 'C2R', 'N', 54),
    ((4, 8, 9), (1, 2), 'C2R', 'SQRT_N', np.sqrt(128)),
    ((4, 8, 9), (1, 2), 'C2R', 'NONE', 1),
])
def test_normalization_divisor(adapter, shape, axes, kind, scaling, expected):
    divisor = adapter._Scaling[scaling].get_normalization_divisor(
        kind, shape, axes)
    assert divisor == expected


@pytest.mark.parametrize('norm,direction,expected', [
    ('backward', 'forward', 'NONE'),
    ('backward', 'inverse', 'N'),
    ('forward', 'forward', 'N'),
    ('forward', 'inverse', 'NONE'),
    ('ortho', 'forward', 'SQRT_N'),
    ('ortho', 'inverse', 'SQRT_N'),
])
def test_scaling_from_norm(adapter, norm, direction, expected):
    scaling = adapter._Scaling.from_norm(norm, direction)
    assert scaling is adapter._Scaling[expected]


@pytest.mark.parametrize('version,expected', [
    (11299, False), (11300, True), (12040, True),
])
def test_lto_support_follows_loaded_cufft_version(
        adapter, monkeypatch, version, expected):
    monkeypatch.setattr(adapter.nvmath_cufft, 'get_version',
                        Mock(return_value=version))
    adapter._supports_lto_callbacks.cache_clear()
    try:
        assert adapter._supports_lto_callbacks() is expected
    finally:
        adapter._supports_lto_callbacks.cache_clear()


@testing.multi_gpu(2)
def test_keys_distinguish_operand_devices(adapter):
    keys = []
    for device_id in (0, 1):
        with cupy.cuda.Device(device_id):
            operand = cupy.empty((4, 16), dtype=cupy.complex64)
        with cupy.cuda.Device(0):
            keys.append(
                adapter._create_key_from_operand(operand, (1,), 'C2C')[0])
    assert keys[0] != keys[1]


@pytest.mark.parametrize('shape,order,dtype,axes,fft_type,expected', [
    # Accepted layouts are used as is.
    ((4, 16), 'C', np.complex64, (1,), 'C2C', (None, (1,), False)),
    ((16, 4), 'F', np.complex64, (0,), 'C2C', (None, (0,), False)),
    # Promotion copies into C order, permuted only if nvmath rejects that.
    ((4, 16), 'C', np.float32, (1,), 'C2C', (None, (1,), True)),
    ((16, 4), 'F', np.float32, (0,), 'C2C', (None, (0,), True)),
    ((8, 6, 4), 'C', np.float16, (2, 0), 'R2C', ((1, 2, 0), (1, 2), False)),
    # Rejected layouts are copied as nvmath recommends.
    ((8, 12, 16, 20), 'C', np.complex64, (0, 2), 'C2C',
     ((1, 3, 0, 2), (2, 3), True)),
])
def test_create_key_layouts(
        adapter, shape, order, dtype, axes, fft_type, expected):
    operand = cupy.empty(shape, dtype=dtype, order=order)
    _, _, permutation, new_axes, options = adapter._create_key_from_operand(
        operand, axes, fft_type)
    assert (permutation, new_axes, options.inplace) == expected


# (function, input shape, input dtype, keyword arguments)
_NORMALIZED_FFTS = [
    ('ifft', (4, 16), cupy.complex64, {}),
    ('fft', (4, 16), cupy.complex128, {}),
    ('ifftn', (4, 6, 8), cupy.complex64, {'axes': [0, 2]}),
    ('rfft', (4, 16), cupy.float32, {}),
    ('irfft', (4, 9), cupy.complex64, {}),
    ('irfftn', (4, 6, 9), cupy.complex128, {'axes': (1, 2)}),
]

# Layouts nvmath rejects, so they run on a permuted copy: in place for C2C,
# out of place for R2C and C2R.
_PERMUTED_FFTS = [
    ('fftn', (8, 12, 16, 20), cupy.complex64, {'axes': (0, 2)}),
    ('ifft', (16, 24, 32), cupy.complex128, {'axis': 1}),
    ('rfftn', (8, 12, 16, 20), cupy.float32, {'axes': (0, 2)}),
    ('irfftn', (8, 12, 16, 11), cupy.complex64, {'axes': (0, 2)}),
]

# Inputs CuPy promotes before the transform, onto a C-contiguous copy that is
# permuted when nvmath rejects it.
_PROMOTED_FFTS = [
    ('fft', (4, 16), cupy.float32, {}),
    ('fft', (16, 4), cupy.float64, {'axis': 0}),
    ('ifftn', (6, 4, 8), cupy.float16, {'axes': (0, 2)}),
    ('rfftn', (8, 6, 4), cupy.float16, {'axes': (2, 0)}),
    ('irfft', (4, 9), cupy.float32, {}),
    ('irfftn', (6, 4, 5), cupy.float64, {'axes': (0, 2)}),
]


@pytest.mark.skipif(
    not is_cuda_python, reason='requires a CUDA-Python build')
class TestNvmathNormalization:

    @pytest.fixture(autouse=True)
    def prepare_cache(self, adapter):
        self.adapter = adapter
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
        self._check_operand(name, a, kwargs, norm)

    def _check_operand(self, name, a, kwargs, norm):
        before = a.copy()
        out = getattr(cupy.fft, name)(a, norm=norm, **kwargs)
        host = a.get()
        if host.dtype == np.float16:
            # NumPy applies 'ortho' scaling to float16 in half precision.
            host = host.astype(np.float32)
        expected = getattr(np.fft, name)(host, norm=norm, **kwargs)
        tol = 1e-4 if out.dtype in (cupy.complex64, cupy.float32) else 1e-10
        testing.assert_allclose(out, expected, rtol=tol, atol=tol)
        testing.assert_array_equal(a, before)

    @pytest.mark.parametrize('norm', [None, 'ortho', 'forward'])
    @pytest.mark.parametrize('name,shape,dtype,kwargs', _NORMALIZED_FFTS)
    def test_fused_normalization(self, name, shape, dtype, kwargs, norm):
        if not self.adapter._supports_lto_callbacks():
            pytest.skip('cuFFT does not support LTO callbacks')
        self._check(name, shape, dtype, kwargs, norm)
        plans = self._cached_plans()
        assert len(plans) == 1
        assert isinstance(plans[0], self.adapter.CachedFFT)
        scaled = (norm == 'ortho'
                  or (norm == 'forward') == name.startswith(('fft', 'rfft')))
        assert (plans[0]._cupy_normalization_scalar is not None) == scaled

    @pytest.mark.parametrize('norm', [None, 'ortho', 'forward'])
    @pytest.mark.parametrize('name,shape,dtype,kwargs', _NORMALIZED_FFTS)
    def test_unfused_normalization_on_older_cufft(
            self, monkeypatch, name, shape, dtype, kwargs, norm):
        monkeypatch.setattr(self.adapter, '_supports_lto_callbacks',
                            lambda: False)
        self._check(name, shape, dtype, kwargs, norm)
        plans = self._cached_plans()
        assert len(plans) == 1
        assert isinstance(plans[0], self.adapter.CachedFFT)
        assert plans[0]._cupy_normalization_scalar is None

    def test_unfused_normalization_shares_unscaled_plan(self, monkeypatch):
        monkeypatch.setattr(self.adapter, '_supports_lto_callbacks',
                            lambda: False)
        for norm in (None, 'ortho', 'forward'):
            self._check('ifft', (4, 16), cupy.complex64, {}, norm)
        assert len(self._cached_plans()) == 1

    def test_cached_layout_keeps_each_norms_divisor(self):
        for norm in (None, 'ortho', 'forward', None, 'ortho'):
            self._check('ifft', (4, 16), cupy.complex64, {}, norm)

    @pytest.mark.parametrize('name,shape,dtype,kwargs', _PERMUTED_FFTS)
    def test_permuted_layouts(self, name, shape, dtype, kwargs):
        for norm in (None, 'ortho', None):
            self._check(name, shape, dtype, kwargs, norm)

    @pytest.mark.parametrize('layout', ['C', 'F', 'reversed'])
    @pytest.mark.parametrize('name,shape,dtype,kwargs', _PROMOTED_FFTS)
    def test_promoted_dtypes(self, name, shape, dtype, kwargs, layout):
        a = testing.shaped_random(shape, cupy, dtype)
        if layout == 'F':
            a = cupy.asfortranarray(a)
        elif layout == 'reversed':
            a = a[..., ::-1]
        for norm in (None, 'ortho'):
            self._check_operand(name, a, kwargs, norm)
        plans = self._cached_plans()
        assert plans
        assert all(isinstance(plan, self.adapter.CachedFFT) for plan in plans)

    def test_promoted_layouts_share_plan(self):
        a = testing.shaped_random((8, 16), cupy, cupy.float32)
        for operand in (a, cupy.asfortranarray(a), a[:, ::-1]):
            self._check_operand('fft', operand, {}, None)
        assert len(self._cached_plans()) == 1

    def test_direct_call_on_permuted_copys_layout_keeps_its_input(self):
        a = testing.shaped_random((8, 12, 16, 20), cupy, cupy.complex64)
        _, _, permutation, axes, options = (
            self.adapter._create_key_from_operand(a, (0, 2), 'C2C'))
        assert options.inplace
        self._check_operand('fftn', a, {'axes': (0, 2)}, None)
        # Laid out like the private copy the call above transformed in place.
        b = a.transpose(permutation).copy()
        self._check_operand('fftn', b, {'axes': axes}, None)

    def test_epilog_failure_falls_back_to_native(self, monkeypatch):
        error = RuntimeError('compilation failed')
        monkeypatch.setattr(self.adapter, '_supports_lto_callbacks',
                            lambda: True)
        monkeypatch.setattr(self.adapter, '_get_scaling_epilog',
                            Mock(side_effect=error))
        with pytest.warns(RuntimeWarning, match='compilation failed'):
            self._check('ifft', (4, 16), cupy.complex64, {}, None)
        plans = self._cached_plans()
        assert not any(
            isinstance(plan, self.adapter.CachedFFT) for plan in plans)

    @pytest.mark.parametrize('dtype,real_dtype', [
        (np.float32, np.float32), (np.complex64, np.float32),
        (np.float64, np.float64), (np.complex128, np.float64),
    ])
    def test_epilogs_share_ltoir_and_own_scalars(self, dtype, real_dtype):
        dtype = np.dtype(dtype)
        device_id = cupy.cuda.Device().id
        first = self.adapter._get_scaling_epilog(dtype, 32, device_id)
        second = self.adapter._get_scaling_epilog(dtype, 128, device_id)
        assert first[0].ltoir is second[0].ltoir
        assert first[0].data != second[0].data
        for (epilog, scalar), value in ((first, 32), (second, 128)):
            assert epilog.data == scalar.data.ptr
            assert scalar.dtype == real_dtype
            assert scalar.get().tolist() == [value]

    @testing.multi_gpu(2)
    def test_epilog_scalar_lives_on_requested_device(self):
        with cupy.cuda.Device(0):
            epilog, scalar = self.adapter._get_scaling_epilog(
                np.dtype(np.complex64), 32, 1)
            assert cupy.cuda.Device().id == 0
        assert scalar.device.id == 1
        assert epilog.data == scalar.data.ptr
