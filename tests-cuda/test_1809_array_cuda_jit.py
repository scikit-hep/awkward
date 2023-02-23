# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

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


@nb_cuda.jit(extensions=[ak.numba.cuda])
def multiply(array, n, out):
    tid = nb_cuda.grid(1)
    out[tid] = array[tid] * n


@nb_cuda.jit(extensions=[ak.numba.cuda])
def pass_through(array, out):
    x = nb_cuda.grid(1)
    out[x] = array[x] if array[x] is not None else np.nan


@nb_cuda.jit(extensions=[ak.numba.cuda])
def pass_through_2d(array, out):
    x, y = nb_cuda.grid(2)
    if x < len(array) and y < len(array[x]):
        out[x][y] = array[x][y]
    else:
        out[x][y] = np.nan


@nb_cuda.jit(extensions=[ak.numba.cuda])
def pass_regular_through_2d(array, out):
    x, y = nb_cuda.grid(2)
    out[x][y] = array[x][y]


@nb_cuda.jit(extensions=[ak.numba.cuda])
def pass_record_through(array, out):
    tid = nb_cuda.grid(1)
    out[tid] = array.x[tid]


@nb_cuda.jit(extensions=[ak.numba.cuda])
def count_records(array, out):
    tid = nb_cuda.grid(1)
    record = array[tid]
    out[tid] = np.nan if record is None else tid


@nb_cuda.jit(extensions=[ak.numba.cuda])
def pass_two_records_through(array, out):
    tid = nb_cuda.grid(1)
    out[tid][0] = array.x[tid]
    out[tid][1] = array.y[tid]


@nb_cuda.jit(extensions=[ak.numba.cuda])
def pass_two_tuple_through(array, out):
    tid = nb_cuda.grid(1)
    out[tid][0] = array["0"][tid]
    out[tid][1] = array["1"][tid]


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
        ak.drop_none(ak.nan_to_none(ak.Array(host_results))).tolist() == array.tolist()
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

    assert ak.Array(host_results).tolist() == array.x.to_list()


@numbatest
def test_EmptyArray():
    array = ak.Array(ak.contents.EmptyArray(), backend="cuda")

    results = nb_cuda.to_device(np.empty(len(array), dtype=np.int32))
    pass_through[1, 1](array, results)

    nb_cuda.synchronize()
    host_results = results.copy_to_host()

    assert ak.Array(host_results).tolist() == array.to_list()


@numbatest
def test_NumpyArray():
    array = ak.Array(ak.contents.NumpyArray([0, 1, 2, 3]), backend="cuda")

    results = nb_cuda.to_device(np.empty(len(array), dtype=np.int32))
    pass_through[1, 4](array, results)

    nb_cuda.synchronize()
    host_results = results.copy_to_host()

    assert ak.Array(host_results).tolist() == array.to_list()


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

    assert ak.Array(host_results).tolist() == array.to_list()


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
        ak.drop_none(ak.nan_to_none(ak.Array(host_results))).tolist() == array.to_list()
    )


@numbatest
def test_ListArray_NumpyArray():
    array = ak.Array(
        ak.contents.ListArray(
            ak.index.Index(np.array([4, 100, 1], np.int64)),
            ak.index.Index(np.array([7, 100, 3, 200], np.int64)),
            ak.contents.NumpyArray(np.array([6.6, 4.4, 5.5, 7.7, 1.1, 2.2, 3.3, 8.8])),
        ),
        backend="cuda",
    )
    results = nb_cuda.to_device(np.empty(9, dtype=np.float64).reshape((3, 3)))

    pass_through_2d[(3, 1), (1, 3)](array, results)

    nb_cuda.synchronize()
    host_results = results.copy_to_host()

    assert ak.drop_none(ak.nan_to_none(ak.Array(host_results))).tolist() == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
    ]


@numbatest
def test_ListOffsetArray_NumpyArray():
    array = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index(np.array([1, 4, 4, 6, 7], np.int64)),
            ak.contents.NumpyArray(np.array([6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7])),
        ),
        backend="cuda",
    )
    results = nb_cuda.to_device(np.empty(12, dtype=np.float64).reshape((4, 3)))

    pass_through_2d[(4, 1), (1, 3)](array, results)

    nb_cuda.synchronize()
    host_results = results.copy_to_host()

    assert (
        ak.drop_none(ak.nan_to_none(ak.Array(host_results))).tolist() == array.to_list()
    )


@numbatest
def test_RecordArray_NumpyArray():
    array = ak.Array(
        ak.contents.RecordArray(
            [
                ak.contents.NumpyArray(np.array([0, 1, 2, 3, 4], np.int64)),
                ak.contents.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
            ],
            ["x", "y"],
        ),
        backend="cuda",
    )
    assert array.to_list() == [
        {"x": 0, "y": 0.0},
        {"x": 1, "y": 1.1},
        {"x": 2, "y": 2.2},
        {"x": 3, "y": 3.3},
        {"x": 4, "y": 4.4},
    ]
    results = nb_cuda.to_device(np.empty(10, dtype=np.float64).reshape((5, 2)))

    pass_two_records_through[5, 1](array, results)

    nb_cuda.synchronize()
    host_results = results.copy_to_host()

    assert ak.to_list(host_results) == [
        [0.0, 0.0],
        [1.0, 1.1],
        [2.0, 2.2],
        [3.0, 3.3],
        [4.0, 4.4],
    ]

    array = ak.Array(
        ak.contents.RecordArray(
            [
                ak.contents.NumpyArray(np.array([0, 1, 2, 3, 4], np.int64)),
                ak.contents.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
            ],
            None,
        ),
        backend="cuda",
    )
    assert array.to_list() == [(0, 0.0), (1, 1.1), (2, 2.2), (3, 3.3), (4, 4.4)]
    results = nb_cuda.to_device(np.empty(10, dtype=np.float64).reshape((5, 2)))

    pass_two_tuple_through[5, 1](array, results)

    nb_cuda.synchronize()
    host_results = results.copy_to_host()

    assert ak.Array(host_results).to_list() == [
        [0.0, 0.0],
        [1.0, 1.1],
        [2.0, 2.2],
        [3.0, 3.3],
        [4.0, 4.4],
    ]

    array = ak.Array(
        ak.contents.RecordArray([], [], 10),
        backend="cuda",
    )
    assert array.to_list() == [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
    results = nb_cuda.to_device(np.empty(10, dtype=np.int32))
    count_records[1, 10](array, results)

    nb_cuda.synchronize()
    host_results = results.copy_to_host()

    assert ak.Array(host_results).tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    array = ak.Array(
        ak.contents.RecordArray([], None, 10),
        backend="cuda",
    )
    assert array.to_list() == [(), (), (), (), (), (), (), (), (), ()]
    results = nb_cuda.to_device(np.empty(10, dtype=np.int32))
    count_records[1, 10](array, results)

    nb_cuda.synchronize()
    host_results = results.copy_to_host()

    assert ak.Array(host_results).tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


@numbatest
def test_IndexedArray_NumpyArray():
    array = ak.Array(
        ak.contents.IndexedArray(
            ak.index.Index(np.array([2, 2, 0, 1, 4, 5, 4], np.int64)),
            ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        ),
        backend="cuda",
    )
    results = nb_cuda.to_device(np.empty(7, dtype=np.float64))

    pass_through[1, 7](array, results)

    nb_cuda.synchronize()
    host_results = results.copy_to_host()

    assert ak.Array(host_results).tolist() == array.to_list()


@numbatest
def test_IndexedOptionArray_NumpyArray():
    array = ak.Array(
        ak.contents.IndexedOptionArray(
            ak.index.Index(np.array([2, 2, -1, 1, -1, 5, 4], np.int64)),
            ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        ),
        backend="cuda",
    )
    results = nb_cuda.to_device(np.empty(7, dtype=np.float64))

    pass_through[1, 7](array, results)

    nb_cuda.synchronize()
    host_results = results.copy_to_host()

    assert ak.nan_to_none(ak.Array(host_results)).tolist() == [
        3.3,
        3.3,
        None,
        2.2,
        None,
        6.6,
        5.5,
    ]


@numbatest
def test_ByteMaskedArray_NumpyArray():
    array = ak.Array(
        ak.contents.ByteMaskedArray(
            ak.index.Index(np.array([1, 0, 1, 0, 1], np.int8)),
            ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
            valid_when=True,
        ),
        backend="cuda",
    )
    results = nb_cuda.to_device(np.empty(5, dtype=np.float64))

    pass_through[1, 5](array, results)

    nb_cuda.synchronize()
    host_results = results.copy_to_host()

    assert ak.nan_to_none(ak.Array(host_results)).tolist() == array.to_list()

    array = ak.Array(
        ak.contents.ByteMaskedArray(
            ak.index.Index(np.array([0, 1, 0, 1, 0], np.int8)),
            ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
            valid_when=False,
        ),
        backend="cuda",
    )
    results = nb_cuda.to_device(np.empty(5, dtype=np.float64))

    pass_through[1, 5](array, results)

    nb_cuda.synchronize()
    host_results = results.copy_to_host()

    assert ak.nan_to_none(ak.Array(host_results)).tolist() == array.to_list()
