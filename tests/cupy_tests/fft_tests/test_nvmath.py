from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


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
    (2, np.array([-2, -1]), (0, 1)),
    (3, np.int64(-1), (2,)),
    (3, (np.int64(-3), np.int64(-1)), (0, 2)),
    (3, (2, 0), (2, 0)),
    (3, (0, -3), None),
    (2, (2,), None),
    (2, (-3,), None),
    (4, None, None),
    (4, (0, 1, 2, 3), None),
    (0, None, ()),
    (0, 0, None),
])
def test_normalize_axes(adapter, ndim, axes, expected):
    assert adapter._normalize_axes(ndim, axes) == expected


def test_axes_cache_preserves_uncacheable_inputs(adapter):
    cached = adapter._normalize_axes_cached
    cached.cache_clear()
    first = adapter._normalize_axes(2, (1,))
    assert adapter._normalize_axes(2, (1,)) is first
    hits = cached.cache_info().hits
    assert type(adapter._normalize_axes(2, (1.0,))[0]) is float
    axes = [0]
    assert adapter._normalize_axes(2, axes) == (0,)
    axes[0] = 1
    assert adapter._normalize_axes(2, axes) == (1,)
    assert cached.cache_info().hits == hits


@pytest.mark.parametrize('shape,axes,kind,policy,expected', [
    ((4, 8, 16), (1, 2), 'C2C', 'N', 128),
    ((4, 8, 16), (0, 2), 'R2C', 'N', 64),
    ((4, 8, 9), (1, 2), 'C2R', 'N', 128),
    ((4, 8, 9), (2, 0), 'C2R', 'N', 54),
    ((4, 8, 9), (1, 2), 'C2R', 'sqrtN', np.sqrt(128)),
    ((4, 8, 9), (1, 2), 'C2R', 'none', 1),
])
def test_normalization_divisor(adapter, shape, axes, kind, policy, expected):
    operand = SimpleNamespace(shape=shape)
    assert adapter._normalization_divisor(
        operand, axes, kind, policy) == expected


class _Array:
    ndim = 1
    size = 32
    shape = (32,)
    dtype = np.dtype('complex64')
    data = SimpleNamespace(device_id=0)


class _Cache(dict):
    is_enabled = True


@pytest.fixture
def dispatch(adapter, monkeypatch):
    from cupy.cuda import cufft

    cache, plans = _Cache(), []
    stream = SimpleNamespace(ptr=7, synchronize=Mock())
    output = object()

    def create(*args, **kwargs):
        plan = SimpleNamespace(
            plan=Mock(), execute=Mock(return_value=output),
            reset_operand_unchecked=Mock(), release_operand=Mock(),
            free=Mock())
        plans.append(plan)
        return plan

    prepare = Mock(return_value=object())
    monkeypatch.setattr(adapter.cupy, 'ndarray', _Array)
    monkeypatch.setattr(adapter, 'config', SimpleNamespace(
        use_nvmath=True, devices=None,
        get_current_callback_manager=lambda: None))
    monkeypatch.setattr(cufft, 'get_current_plan', lambda: None)
    monkeypatch.setattr(adapter.cupy.cuda.runtime, 'getDevice', lambda: 0)
    stream_lookup = Mock(return_value=stream)
    monkeypatch.setattr(adapter.cupy.cuda, 'get_current_stream', stream_lookup)
    monkeypatch.setattr(adapter, 'get_plan_cache', lambda: cache)
    monkeypatch.setattr(adapter, '_create_key_from_operand',
                        Mock(return_value=('metadata',)))
    monkeypatch.setattr(adapter, 'CachedFFT', create)
    monkeypatch.setattr(adapter, '_normalization_epilog', prepare)

    def invoke(norm=None, direction='forward'):
        return adapter._try_use_nvmath(
            _Array(), None, (0,), norm,
            fft_type='C2C', fft_direction=direction)

    return SimpleNamespace(invoke=invoke, plans=plans, cache=cache,
                           prepare=prepare, output=output, stream=stream,
                           stream_lookup=stream_lookup)


@pytest.mark.parametrize('norm,direction,policy', [
    (None, 'forward', 'none'),
    ('backward', 'forward', 'none'),
    ('forward', 'inverse', 'none'),
    (None, 'inverse', 'N'),
    ('backward', 'inverse', 'N'),
    ('forward', 'forward', 'N'),
    ('ortho', 'forward', 'sqrtN'),
    ('ortho', 'inverse', 'sqrtN'),
])
def test_normalization_policy(dispatch, norm, direction, policy):
    dispatch.invoke(norm, direction)
    key = ('nvmath', dispatch.stream.ptr, ('metadata',))
    if policy != 'none':
        key += (('normalization', policy),)
    assert list(dispatch.cache) == [key]
    assert dispatch.prepare.call_count == (policy != 'none')


def test_warm_hit_skips_callback_preparation(dispatch):
    assert dispatch.invoke(direction='inverse') is dispatch.output
    assert dispatch.invoke(direction='inverse') is dispatch.output
    dispatch.prepare.assert_called_once()
    assert len(dispatch.plans) == 1
    plan = dispatch.plans[0]
    plan.reset_operand_unchecked.assert_called_once()
    assert plan.release_operand.call_count == 2
    assert plan._normalization_epilog is dispatch.prepare.return_value
    assert plan._normalization_stream is dispatch.stream
    dispatch.stream_lookup.assert_called_with(_Array.data.device_id)
    assert plan.execute.call_args.kwargs['release_workspace'] is True


def test_equivalent_normalizations_share_semantic_plan(dispatch):
    for norm, direction in ((None, 'inverse'), ('backward', 'inverse'),
                            ('forward', 'forward')):
        dispatch.invoke(norm, direction)
    dispatch.invoke('ortho', 'forward')
    dispatch.invoke('ortho', 'inverse')
    assert len(dispatch.plans) == 2
    assert dispatch.prepare.call_count == 2
    assert {key[-1] for key in dispatch.cache} == {
        ('normalization', 'N'), ('normalization', 'sqrtN')}


def test_unscaled_dispatch_never_prepares_callback(dispatch):
    dispatch.invoke()
    dispatch.invoke('forward', 'inverse')
    dispatch.prepare.assert_not_called()
    assert len(dispatch.plans) == 1
    assert next(iter(dispatch.cache)) == ('nvmath', 7, ('metadata',))
    assert dispatch.plans[0]._normalization_epilog is None


def test_failed_preparation_falls_back_without_caching_plan(
        adapter, dispatch, monkeypatch):
    error = RuntimeError('compilation failed')
    dispatch.prepare.side_effect = error
    warn = Mock()
    monkeypatch.setattr(adapter, '_warn_fallback', warn)
    assert dispatch.invoke(direction='inverse') is None
    assert not dispatch.cache
    assert not dispatch.plans
    warn.assert_called_once_with(error)


def test_failed_layout_retry_falls_back_without_preparing_callback(
        adapter, dispatch, monkeypatch):
    class UnsupportedLayoutError(Exception):
        permutation = (0,)
        axes = (0,)

    monkeypatch.setattr(adapter.nvmath_fft, 'UnsupportedLayoutError',
                        UnsupportedLayoutError)
    error = RuntimeError('recommended layout rejected')
    key = adapter._create_key_from_operand
    key.side_effect = [UnsupportedLayoutError(), error]
    warn = Mock()
    monkeypatch.setattr(adapter, '_warn_fallback', warn)
    assert dispatch.invoke(direction='inverse') is None
    assert key.call_count == 2
    dispatch.prepare.assert_not_called()
    assert not dispatch.plans
    warn.assert_called_once_with(error)


@pytest.mark.parametrize('kind,input_type,output_type', [
    ('C2C', 'complex64', 'complex64'),
    ('R2C', 'float64', 'complex128'),
    ('C2R', 'complex64', 'float32'),
])
def test_epilog_uses_result_dtype(adapter, monkeypatch, kind, input_type,
                                  output_type):
    prepare = Mock(return_value=object())
    monkeypatch.setattr(adapter._nvmath_scaling, 'get_scaling_epilog', prepare)
    operand = SimpleNamespace(shape=(8,), dtype=np.dtype(input_type))
    adapter._normalization_epilog(operand, (0,), kind, 'N')
    prepare.assert_called_once_with(output_type, 14 if kind == 'C2R' else 8)


@pytest.mark.parametrize('dtype,real_dtype', [
    ('float32', np.float32), ('complex64', np.float32),
    ('float64', np.float64), ('complex128', np.float64),
])
def test_epilogs_share_code_and_own_independent_scalars(
        adapter, monkeypatch, dtype, real_dtype):
    from cupy.cuda import compiler

    scaling = adapter._nvmath_scaling
    compile_code = Mock(return_value=b'IR')
    uploads = []

    def upload(host, *, blocking):
        assert blocking
        scalar = SimpleNamespace(host=host.copy(),
                                 data=SimpleNamespace(ptr=len(uploads) + 1))
        uploads.append(scalar)
        return scalar

    monkeypatch.setattr(compiler, '_compile_module_with_cache', compile_code)
    monkeypatch.setattr(compiler, '_get_arch', lambda: '90')
    monkeypatch.setattr(scaling.cupy.cuda, 'Device',
                        lambda: SimpleNamespace(compute_capability='100'))
    monkeypatch.setattr(scaling.cufft, 'get_version', lambda: 12040)
    monkeypatch.setattr(scaling.cupy, 'asarray', upload)
    scaling._compile_code.cache_clear()
    try:
        first = scaling.get_scaling_epilog(dtype, 32)
        second = scaling.get_scaling_epilog(dtype, 128)
        compile_code.assert_called_once()
        assert compile_code.call_args.kwargs['to_ltoir'] is True
        assert compile_code.call_args.kwargs['arch'] == '90'
        assert first.ltoir is second.ltoir
        assert first._cupy_scalar is not second._cupy_scalar
        assert first.data != second.data
        assert first._cupy_scalar.host.dtype == real_dtype
        assert second._cupy_scalar.host.dtype == real_dtype
        assert first._cupy_scalar.host[0] == 32
        assert second._cupy_scalar.host[0] == 128
    finally:
        scaling._compile_code.cache_clear()


def test_unit_divisor_does_not_compile_or_upload(adapter, monkeypatch):
    scaling = adapter._nvmath_scaling
    compile_code, upload = Mock(), Mock()
    monkeypatch.setattr(scaling, '_compile_code', compile_code)
    monkeypatch.setattr(scaling.cupy, 'asarray', upload)
    assert scaling.get_scaling_epilog('complex64', 1) is None
    compile_code.assert_not_called()
    upload.assert_not_called()


def test_cleanup_completes_callback_reads_before_dropping_owner(
        adapter, monkeypatch):
    from contextlib import nullcontext

    plan = object.__new__(adapter.CachedFFT)
    plan.device_id = 0
    plan._normalization_epilog = object()
    events = []
    plan._normalization_stream = SimpleNamespace(
        synchronize=lambda: events.append('synchronize'))

    def release(self):
        assert self._normalization_epilog is not None
        events.append('free')
        self._normalization_epilog = None

    monkeypatch.setattr(adapter.cupy.cuda, 'Device',
                        lambda device: nullcontext())
    monkeypatch.setattr(adapter.nvmath_fft.FFT, 'free', release)
    plan.free()
    assert events == ['synchronize', 'free']
