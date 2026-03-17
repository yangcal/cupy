from __future__ import annotations

import numpy
import pytest

import cupy
from cupy import testing
from cupyx.scipy import sparse


# Index value that exceeds INT32_MAX (= 2**31 - 1), forcing int64.
_LARGE = 2**31 + 1


class TestInt64Construction:
    """Index dtype is chosen by get_index_dtype, not forced to int32.

    The key change: constructors call get_index_dtype(check_contents=True)
    instead of unconditionally casting to int32.  Small values still produce
    int32 (no regression); large values now produce int64.
    """

    def test_csr_large_col_index_uses_int64(self):
        # Column index > INT32_MAX → both indices and indptr use int64.
        # CSR invariant: indices and indptr always share the same dtype.
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, _LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2], dtype=cupy.int64)
        m = sparse.csr_matrix((data, indices, indptr), shape=(2, _LARGE + 1))
        assert m.indices.dtype == cupy.int64
        assert m.indptr.dtype == cupy.int64
        assert int(m.indices[1]) == _LARGE

    def test_csc_large_row_index_uses_int64(self):
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, _LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2], dtype=cupy.int64)
        m = sparse.csc_matrix((data, indices, indptr), shape=(_LARGE + 1, 2))
        assert m.indices.dtype == cupy.int64
        assert m.indptr.dtype == cupy.int64
        assert int(m.indices[1]) == _LARGE

    def test_coo_large_col_index_uses_int64(self):
        data = cupy.array([1.0, 2.0])
        row = cupy.array([0, 1], dtype=cupy.int64)
        col = cupy.array([0, _LARGE], dtype=cupy.int64)
        m = sparse.coo_matrix((data, (row, col)), shape=(2, _LARGE + 1))
        assert m.row.dtype == cupy.int64
        assert m.col.dtype == cupy.int64
        assert int(m.col[1]) == _LARGE

    def test_csr_int64_with_small_values_stays_int32(self):
        # get_index_dtype(check_contents=True) downcasts int64 arrays when
        # all values fit in int32.  This is the scipy-compatible behavior.
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, 5], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2], dtype=cupy.int64)
        m = sparse.csr_matrix((data, indices, indptr), shape=(2, 10))
        assert m.indices.dtype == cupy.int32
        assert m.indptr.dtype == cupy.int32

    def test_coo_int64_with_small_values_stays_int32(self):
        data = cupy.array([1.0, 2.0])
        row = cupy.array([0, 1], dtype=cupy.int64)
        col = cupy.array([0, 5], dtype=cupy.int64)
        m = sparse.coo_matrix((data, (row, col)))
        assert m.row.dtype == cupy.int32
        assert m.col.dtype == cupy.int32

    def test_empty_csr_large_shape_uses_int64(self):
        # Shape-only construction: max(shape) > INT32_MAX → int64 index arrays,
        # even when there are no stored elements.
        m = sparse.csr_matrix((2, _LARGE + 1))
        assert m.indices.dtype == cupy.int64
        assert m.indptr.dtype == cupy.int64

    def test_empty_coo_large_shape_uses_int64(self):
        m = sparse.coo_matrix((2, _LARGE + 1))
        assert m.row.dtype == cupy.int64
        assert m.col.dtype == cupy.int64

    @testing.with_requires('scipy')
    def test_from_scipy_csr_preserves_int64(self):
        # scipy uses its own get_index_dtype; CuPy must trust and preserve it.
        import scipy.sparse
        data = numpy.array([1.0, 2.0])
        indices = numpy.array([0, _LARGE], dtype=numpy.int64)
        indptr = numpy.array([0, 1, 2], dtype=numpy.int64)
        sp = scipy.sparse.csr_matrix(
            (data, indices, indptr), shape=(2, _LARGE + 1))
        m = sparse.csr_matrix(sp)
        assert m.indices.dtype == cupy.int64
        assert m.indptr.dtype == cupy.int64
        assert int(m.indices[1]) == _LARGE


class TestInt64FormatConversion:
    """Pure-CuPy fallbacks for tocoo / tocsr / tocsc with int64.

    cuSPARSE's xcsr2coo / xcoo2csr / csr2cscEx2 only accept int32 pointers.
    The fallbacks use searchsorted (for indptr expansion) and unique+scatter
    (for indptr construction), avoiding the 2×large-allocation OOM of bincount.
    """

    def _make_int64_csr(self):
        """2-row CSR: row 0 → col 0, row 1 → col _LARGE."""
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, _LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2], dtype=cupy.int64)
        return sparse.csr_matrix(
            (data, indices, indptr), shape=(2, _LARGE + 1))

    def _make_int64_coo(self):
        """2-entry COO: (row=0, col=0, val=1.0), (row=1, col=_LARGE, val=2.0)."""
        data = cupy.array([1.0, 2.0])
        row = cupy.array([0, 1], dtype=cupy.int64)
        col = cupy.array([0, _LARGE], dtype=cupy.int64)
        return sparse.coo_matrix(
            (data, (row, col)), shape=(2, _LARGE + 1))

    def test_csr_tocoo_int64(self):
        # Earlier in development, xcsr2coo read int64 indptr as int32, producing
        # silently wrong row indices (e.g. row=[1,0] instead of [0,1]).
        coo = self._make_int64_csr().tocoo()
        assert coo.row.dtype == cupy.int64
        assert coo.col.dtype == cupy.int64
        assert int(coo.row[0]) == 0
        assert int(coo.row[1]) == 1
        assert int(coo.col[0]) == 0
        assert int(coo.col[1]) == _LARGE

    def test_coo_tocsr_int64(self):
        csr = self._make_int64_coo().tocsr()
        assert csr.indices.dtype == cupy.int64
        assert csr.indptr.dtype == cupy.int64
        assert int(csr.indptr[0]) == 0
        assert int(csr.indptr[1]) == 1
        assert int(csr.indptr[2]) == 2
        assert int(csr.indices[1]) == _LARGE

    def test_csc_tocoo_int64(self):
        # csc2coo has its own searchsorted fallback, separate from csr2coo.
        # CSC: col 0 → row 0, col 1 → row _LARGE.
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, _LARGE], dtype=cupy.int64)  # row indices
        indptr = cupy.array([0, 1, 2], dtype=cupy.int64)     # 1 nnz per column
        csc = sparse.csc_matrix((data, indices, indptr), shape=(_LARGE + 1, 2))
        coo = csc.tocoo()
        assert coo.row.dtype == cupy.int64
        assert coo.col.dtype == cupy.int64
        assert int(coo.row[0]) == 0
        assert int(coo.row[1]) == _LARGE
        assert int(coo.col[0]) == 0
        assert int(coo.col[1]) == 1

    def test_csr_tocsc_int32_regression(self):
        # The int32 tocsc path must remain correct as we make changes.
        data = cupy.array([1.0, 2.0, 3.0])
        indices = cupy.array([0, 2, 1], dtype=cupy.int32)
        indptr = cupy.array([0, 2, 3], dtype=cupy.int32)
        csr = sparse.csr_matrix((data, indices, indptr), shape=(2, 3))
        csc = csr.tocsc()
        assert csc.indices.dtype == cupy.int32
        assert csc.indptr.dtype == cupy.int32
        testing.assert_array_equal(csc.toarray(), csr.toarray())

    @testing.slow
    def test_csr_tocsc_int64(self):
        # _cupy_csr2csc_int64 uses unique+scatter for the CSC indptr.
        # This test requires ~17 GB for the CSC indptr; skipped if OOM.
        n = numpy.iinfo(numpy.int32).max + 3  # INT32_MAX + 3 ≈ 2.15 B
        try:
            data = cupy.array([1.0, 2.0])
            indices = cupy.array([0, n - 1], dtype=cupy.int64)
            indptr = cupy.array([0, 1, 2], dtype=cupy.int64)
            csr = sparse.csr_matrix((data, indices, indptr), shape=(2, n))
            csc = csr.tocsc()
        except cupy.cuda.memory.OutOfMemoryError:
            pytest.skip('not enough GPU memory')
        assert csc.indices.dtype == cupy.int64
        assert csc.indptr.dtype == cupy.int64
        # Row indices in CSC: col 0 holds row 0, col n-1 holds row 1.
        assert int(csc.indices[0]) == 0
        assert int(csc.indices[1]) == 1


class TestInt64Sort:
    """Lexsort-based fallbacks for csrsort, cscsort, sum_duplicates.

    cuSPARSE xcsrsort / xcscsort / xcoosort accept only int32 index pointers.
    """

    def test_csr_sort_indices_int64(self):
        # csrsort int64 path: expand indptr via searchsorted, then lexsort.
        # Row 0 has columns [_LARGE+1, _LARGE] (deliberately unsorted).
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([_LARGE + 1, _LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 2], dtype=cupy.int64)
        m = sparse.csr_matrix((data, indices, indptr), shape=(1, _LARGE + 2))
        assert not m.has_sorted_indices

        m.sort_indices()

        assert m.has_sorted_indices
        assert m.indices.dtype == cupy.int64
        assert int(m.indices[0]) == _LARGE       # smaller col first
        assert int(m.indices[1]) == _LARGE + 1
        assert float(m.data[0]) == 2.0           # data reordered with indices
        assert float(m.data[1]) == 1.0

    def test_csc_sort_indices_int64(self):
        # cscsort int64 path: column 0 has rows [_LARGE+1, _LARGE] (unsorted).
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([_LARGE + 1, _LARGE], dtype=cupy.int64)
        indptr = cupy.array([0, 2, 2], dtype=cupy.int64)  # 2 nnz in col 0
        m = sparse.csc_matrix(
            (data, indices, indptr), shape=(_LARGE + 2, 2))
        assert not m.has_sorted_indices

        m.sort_indices()

        assert m.has_sorted_indices
        assert m.indices.dtype == cupy.int64
        assert int(m.indices[0]) == _LARGE       # smaller row first
        assert int(m.indices[1]) == _LARGE + 1
        assert float(m.data[0]) == 2.0
        assert float(m.data[1]) == 1.0

    def test_coo_sum_duplicates_int64(self):
        # Previously, the ElementwiseKernel declared 'int32 src_col',
        # silently truncating int64 col values > INT32_MAX to their low 32 bits.
        # Two duplicate entries: both at (row=0, col=_LARGE).
        data = cupy.array([2.0, 3.0])
        row = cupy.array([0, 0], dtype=cupy.int64)
        col = cupy.array([_LARGE, _LARGE], dtype=cupy.int64)
        m = sparse.coo_matrix((data, (row, col)), shape=(1, _LARGE + 1))

        m.sum_duplicates()

        assert m.nnz == 1
        assert m.col.dtype == cupy.int64
        assert int(m.col[0]) == _LARGE  # must not be truncated to int32
        assert float(m.data[0]) == pytest.approx(5.0)

    def test_csr_sort_indices_int32_regression(self):
        # The int32 csrsort path must remain correct.
        data = cupy.array([1.0, 2.0, 3.0])
        indices = cupy.array([2, 0, 1], dtype=cupy.int32)
        indptr = cupy.array([0, 3], dtype=cupy.int32)
        m = sparse.csr_matrix((data, indices, indptr), shape=(1, 3))
        m.sort_indices()
        assert m.has_sorted_indices
        testing.assert_array_equal(m.indices, cupy.array([0, 1, 2]))
        testing.assert_array_equal(m.data, cupy.array([2.0, 3.0, 1.0]))
