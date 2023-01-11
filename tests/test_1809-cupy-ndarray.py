# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import sys

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak  # noqa: F401

numba = pytest.importorskip("numba")

ak.numba_cuda.register_and_check()

np = pytest.importorskip("numpy")
cupy = pytest.importorskip("cupy")

try:
    import numba as nb
    import numba.cuda as nb_cuda
except ImportError:
    nb = nb_cuda = None

try:
    import pyarrow as pa
    import pyarrow.cuda as pa_cuda
except ImportError:
    pa = pa_cuda = None

pyarrowtest = pytest.mark.skipif(
    pa is None or pa_cuda is None,
    reason="requires the pyarrow and pyarrow.cuda packages")
numbatest = pytest.mark.skipif(
    nb is None or nb_cuda is None,
    reason="requires the numba and numba.cuda packages")


@pyarrowtest
def test_pyarrow_cuda_buffer():
    cp_arr = cupy_ndarray_as_random(5)
    pa_cbuf = cupy_ndarray_as.pyarrow_cuda_buffer(cp_arr)

    arr1 = cupy_ndarray_as.numpy_ndarray(cp_arr)
    arr2 = pyarrow_cuda_buffer_as.numpy_ndarray(pa_cbuf)
    np.testing.assert_array_equal(arr1, arr2)

    cp_arr[1] = 99
    arr1 = cupy_ndarray_as.numpy_ndarray(cp_arr)
    arr2 = pyarrow_cuda_buffer_as.numpy_ndarray(pa_cbuf)
    np.testing.assert_array_equal(arr1, arr2)
    assert arr1[1] == 99


@numbatest
def test_numba_cuda_DeviceNDArray():
    cp_arr = cupy_ndarray_as.random(5)
    nb_arr = cupy_ndarray_as.numba_cuda_DeviceNDArray(cp_arr)

    arr1 = cupy_ndarray_as.numpy_ndarray(cp_arr)
    arr2 = numba_cuda_DeviceNDArray_as.numpy_ndarray(nb_arr)
    np.testing.assert_array_equal(arr1, arr2)

    cp_arr[1] = 99
    arr1 = cupy_ndarray_as.numpy_ndarray(cp_arr)
    arr2 = numba_cuda_DeviceNDArray_as.numpy_ndarray(nb_arr)
    np.testing.assert_array_equal(arr1, arr2)
    assert arr1[1] == 99
