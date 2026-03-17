from __future__ import annotations

import numpy
import pytest

import cupy
from cupy import testing
from cupyx import cusparse
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


class TestInt64ArithmeticFallback:
    """Sparse addition with int64 indices — pure-CuPy fallback.

    csrgeam2 routes int64 inputs to _cupy_csrgeam_int64 *before* checking
    cuSPARSE availability, so the path works on any CUDA version.

    The fallback: expand indptr→row via searchsorted, concatenate COO entries
    from both matrices, call sum_duplicates() to merge overlapping positions,
    then convert back to CSR.  Index dtype is
    numpy.result_type(a.indices.dtype, b.indices.dtype) throughout.

    Note: cusparseSpGEAM (the Generic API path) is absent from all public
    cuSPARSE releases through 12.7.9, so the pure-CuPy fallback is always
    active on current installations.
    """

    # Shape has _LARGE+2 columns so a column index of _LARGE is valid and
    # forces int64.  Only 2 rows, so indptr has 3 elements (cheap).
    _shape = (2, _LARGE + 2)

    def _make_int64_csr(self, col, value=1.0):
        """Single-entry CSR: row 0 has one nonzero at (0, col)."""
        data = cupy.array([value])
        indices = cupy.array([col], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 1], dtype=cupy.int64)
        return sparse.csr_matrix((data, indices, indptr), shape=self._shape)

    def test_add_int64_preserves_dtype(self):
        # Both operands have int64 indices; the result must too.
        # (If the fallback accidentally truncated to int32, int(indices[1])
        # would silently wrap and give the wrong column.)
        a = self._make_int64_csr(0)
        b = self._make_int64_csr(_LARGE)
        c = a + b
        assert c.indices.dtype == cupy.int64
        assert c.indptr.dtype == cupy.int64
        assert c.nnz == 2

    def test_add_int64_values_correct(self):
        # Values at column positions 0 and _LARGE are preserved after addition.
        # After sort_indices(), col 0 is always at position 0 and col _LARGE
        # at position 1, so direct array access is safe.
        a = self._make_int64_csr(0, value=3.0)
        b = self._make_int64_csr(_LARGE, value=7.0)
        c = (a + b)
        c.sort_indices()
        assert c.nnz == 2
        assert int(c.indices[0]) == 0
        assert int(c.indices[1]) == _LARGE
        assert float(c.data[0]) == pytest.approx(3.0)
        assert float(c.data[1]) == pytest.approx(7.0)

    def test_add_int64_overlapping_entries_summed(self):
        # When A and B share a (row, col) position, the fallback concatenates
        # both entries into a COO and relies on sum_duplicates() to merge them.
        a = self._make_int64_csr(_LARGE, value=2.0)
        b = self._make_int64_csr(_LARGE, value=5.0)
        c = a + b
        assert c.nnz == 1
        assert c.indices.dtype == cupy.int64
        assert int(c.indices[0]) == _LARGE
        assert float(c.data[0]) == pytest.approx(7.0)

    def test_add_int64_alpha_beta_scaling(self):
        # _cupy_csrgeam_int64 scales a.data by alpha and b.data by beta before
        # concatenation.  Verify through the direct cusparse.csrgeam2 interface
        # since the __add__ operator always uses alpha=1, beta=1.
        # (This test caught a bug where _numpy.array(alpha, ...) returned a
        # 0-d ndarray that CuPy's __mul__ rejected with TypeError.)
        a = self._make_int64_csr(0, value=1.0)
        b = self._make_int64_csr(_LARGE, value=1.0)
        c = cusparse.csrgeam2(a, b, alpha=3.0, beta=4.0)
        c.sort_indices()
        assert c.nnz == 2
        # col 0 → alpha*1.0 = 3.0;  col _LARGE → beta*1.0 = 4.0.
        assert float(c.data[0]) == pytest.approx(3.0)
        assert float(c.data[1]) == pytest.approx(4.0)

    def test_spgeam_int64_fallback(self):
        # cusparse.spgeam() routes int64 directly to _cupy_csrgeam_int64 when
        # cusparseSpGEAM is unavailable (absent from all public releases ≤12.7.9).
        a = self._make_int64_csr(0, value=1.0)
        b = self._make_int64_csr(_LARGE, value=2.0)
        c = cusparse.spgeam(a, b)
        assert c.indices.dtype == cupy.int64
        assert c.nnz == 2

    def test_add_int64_multirow(self):
        # The searchsorted(indptr[1:], arange(nnz)) expansion must assign
        # each nonzero to the correct row.  a has entries in both rows;
        # b has an entry only in row 0 that overlaps a's row-0 entry.
        a = sparse.csr_matrix(
            (cupy.array([1.0, 2.0]),
             cupy.array([0, _LARGE], dtype=cupy.int64),
             cupy.array([0, 1, 2], dtype=cupy.int64)),   # 1 nnz per row
            shape=self._shape)
        b = self._make_int64_csr(_LARGE, value=3.0)   # row 0 → col _LARGE
        c = a + b
        c.sort_indices()
        assert c.nnz == 3
        # Row 0: cols 0 and _LARGE (2 entries).  Row 1: col _LARGE (1 entry).
        assert int(c.indptr[1]) == 2
        assert int(c.indptr[2]) == 3
        # Row 1's entry must retain the exact int64 column value.
        assert int(c.indices[2]) == _LARGE
        assert float(c.data[2]) == pytest.approx(2.0)

    def test_add_mixed_dtype_int32_plus_int64_promotes(self):
        # idx_dtype = numpy.result_type(int32, int64) == int64.
        # The int32 matrix has small column values; the int64 matrix has _LARGE.
        # The result must use int64 to represent _LARGE.
        data = cupy.array([1.0])
        a = sparse.csr_matrix(
            (data, cupy.array([5], dtype=cupy.int32),
             cupy.array([0, 1, 1], dtype=cupy.int32)),
            shape=self._shape)
        b = self._make_int64_csr(_LARGE, value=2.0)
        c = a + b
        assert c.indices.dtype == cupy.int64
        assert c.nnz == 2

    def test_add_int64_empty_operand(self):
        # When one matrix has nnz=0, _cupy_csrgeam_int64 enters the
        # `a_rows = cupy.empty(0, idx_dtype)` branch.  The result equals
        # the non-empty matrix.
        a = self._make_int64_csr(_LARGE)
        b = sparse.csr_matrix(
            (cupy.empty(0, cupy.float64),
             cupy.empty(0, cupy.int64),
             cupy.zeros(3, cupy.int64)),
            shape=self._shape)
        c = a + b
        assert c.indices.dtype == cupy.int64
        assert c.nnz == 1
        assert int(c.indices[0]) == _LARGE

    def test_add_int32_regression(self):
        # int32 + int32 must continue to use the cuSPARSE (csrgeam2) path and
        # return int32 indices with correct values.
        data = cupy.array([1.0, 2.0])
        a = sparse.csr_matrix(
            (data[:1], cupy.array([0], dtype=cupy.int32),
             cupy.array([0, 1, 1], dtype=cupy.int32)),
            shape=(2, 4))
        b = sparse.csr_matrix(
            (data[1:], cupy.array([3], dtype=cupy.int32),
             cupy.array([0, 0, 1], dtype=cupy.int32)),
            shape=(2, 4))
        c = a + b
        assert c.indices.dtype == cupy.int32
        assert c.nnz == 2
        testing.assert_array_equal(c.toarray(), a.toarray() + b.toarray())


class TestInt64SpGEMM:
    """int64-related fixes to spgemm.

    spgemm previously hardcoded 'i' (int32) for the c_indices allocation.
    We updated this to numpy.result_type(a.indices, b.indices).

    Note: cuSPARSE spgemm does not support int64 inputs on current releases
    (all int64 tests xfail), so only the int32 regression is verified here.
    """

    def test_spgemm_int32_result_dtype_preserved(self):
        # result_type(int32, int32) == int32: int64 work must not
        # regress the int32 path.
        if not cusparse.check_availability('spgemm'):
            pytest.skip('spgemm is not available')
        a = sparse.csr_matrix(cupy.eye(3, dtype=cupy.float64))
        b = sparse.csr_matrix(cupy.eye(3, dtype=cupy.float64))
        c = a @ b
        assert c.indices.dtype == cupy.int32
        testing.assert_array_almost_equal(c.toarray(), cupy.eye(3))
