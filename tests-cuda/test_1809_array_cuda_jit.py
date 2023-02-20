# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import cupy as cp  # noqa: F401
import numpy as np
import pytest

import awkward as ak

try:
    import numba as nb
    import numba.cuda as nb_cuda
    from numba import config

    ak.numba.register_and_check()

except ImportError:
    nb = nb_cuda = None

numbatest = pytest.mark.skipif(
    nb is None or nb_cuda is None, reason="requires the numba and numba.cuda packages"
)

config.CUDA_LOW_OCCUPANCY_WARNINGS = False
config.CUDA_WARN_ON_IMPLICIT_COPY = False


@nb_cuda.jit(extensions=[ak.numba.array_view_arg_handler])
def multiply(array, n, out):
    tid = nb_cuda.grid(1)
    out[tid] = array[tid] * n


@nb_cuda.jit(extensions=[ak.numba.array_view_arg_handler])
def pass_through(array, out):
    x = nb_cuda.grid(1)
    out[x] = array[x]


@nb_cuda.jit(extensions=[ak.numba.array_view_arg_handler])
def pass_through_2d(array, out):
    x, y = nb_cuda.grid(2)
    if x < len(array) and y < len(array[x]):
        out[x][y] = array[x][y]
    else:
        out[x][y] = np.nan


@nb_cuda.jit(extensions=[ak.numba.array_view_arg_handler])
def pass_regular_through_2d(array, out):
    x, y = nb_cuda.grid(2)
    out[x][y] = array[x][y]


@nb_cuda.jit(extensions=[ak.numba.array_view_arg_handler])
def pass_record_through(array, out):
    tid = nb_cuda.grid(1)
    out[tid] = array.x[tid]


@numbatest
def test_array_multiply():
    # create an ak.Array with a cuda backend:
    akarray = ak.Array([0, 1, 2, 3], backend="cuda")

    # allocate the result:
    results = nb_cuda.to_device(np.empty(4, dtype=np.int32))

    multiply[1, 4](akarray, 3, results)

    nb_cuda.synchronize()
    host_results = results.copy_to_host()

    assert ak.Array(host_results).tolist() == [0, 3, 6, 9]


@numbatest
def test_array_on_cpu_multiply():
    # create an ak.Array with a cpu backend:
    array = ak.Array([0, 1, 2, 3])

    # allocate the result:
    results = nb_cuda.to_device(np.empty(4, dtype=np.int32))

    with pytest.raises(TypeError):
        multiply[1, 4](array, 3, results)

    multiply[1, 4](ak.to_backend(array, backend="cuda"), 3, results)

    nb_cuda.synchronize()
    host_results = results.copy_to_host()

    assert ak.Array(host_results).tolist() == [0, 3, 6, 9]


@numbatest
def test_ListOffsetArray():
    array = ak.Array([[0, 1], [2], [3, 4, 5]], backend="cuda")

    results = nb_cuda.to_device(np.empty(9, dtype=np.float32).reshape((3, 3)))

    pass_through_2d[(3, 1), (1, 3)](array, results)

    nb_cuda.synchronize()
    host_results = results.copy_to_host()

    assert (
        str(ak.to_list(host_results))
        == "[[0.0, 1.0, nan], [2.0, nan, nan], [3.0, 4.0, 5.0]]"
    )


@numbatest
def test_RecordArray():
    array = ak.Array(
        [{"x": 0}, {"x": 1}, {"x": 2}, {"x": 3}, {"x": 4}, {"x": 5}], backend="cuda"
    )

    results = nb_cuda.to_device(np.empty(6, dtype=np.int32))

    pass_record_through[1, 6](array, results)

    nb_cuda.synchronize()
    host_results = results.copy_to_host()

    assert ak.Array(host_results).tolist() == [0, 1, 2, 3, 4, 5]


@numbatest
def test_EmptyArray():
    array = ak.Array(ak.contents.EmptyArray(), backend="cuda")

    results = nb_cuda.to_device(np.empty(len(array), dtype=np.int32))
    pass_through[1, 1](array, results)

    nb_cuda.synchronize()
    host_results = results.copy_to_host()

    assert ak.Array(host_results).tolist() == []


@numbatest
def test_NumpyArray():
    array = ak.Array(ak.contents.NumpyArray([0, 1, 2, 3]), backend="cuda")

    results = nb_cuda.to_device(np.empty(len(array), dtype=np.int32))
    pass_through[1, 4](array, results)

    nb_cuda.synchronize()
    host_results = results.copy_to_host()

    assert ak.Array(host_results).tolist() == [0, 1, 2, 3]


@numbatest
def test_RegularArray_NumpyArray():
    array = ak.Array(
        ak.contents.RegularArray(
            ak.contents.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
            3,
        ),
        backend="cuda",
    )

    results = nb_cuda.to_device(np.empty(6, dtype=np.float64).reshape((2, 3)))
    pass_through_2d[(2, 1), (1, 3)](array, results)

    nb_cuda.synchronize()
    host_results = results.copy_to_host()

    assert ak.Array(host_results).tolist() == [[0.0, 1.1, 2.2], [3.3, 4.4, 5.5]]


@numbatest
def test_RegularArray_EmptyArray():
    array = ak.Array(
        ak.contents.RegularArray(ak.contents.EmptyArray(), 0, zeros_length=10),
        backend="cuda",
    )

    results = nb_cuda.to_device(np.empty(10, dtype=np.float64).reshape((10, 1)))
    pass_through_2d[(10, 1), (1, 1)](array, results)

    nb_cuda.synchronize()
    host_results = results.copy_to_host()

    assert (
        str(ak.Array(host_results).tolist())
        == "[[nan], [nan], [nan], [nan], [nan], [nan], [nan], [nan], [nan], [nan]]"
    )
