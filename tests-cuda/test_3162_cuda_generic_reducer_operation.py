from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

import awkward as ak

to_list = ak.operations.to_list


@pytest.fixture(scope="function", autouse=True)
def cleanup_cuda():
    yield
    cp._default_memory_pool.free_all_blocks()
    cp.cuda.Device().synchronize()


to_list = ak.operations.to_list

primes = [x for x in range(2, 1000) if all(x % n != 0 for n in range(2, x))]


def test_0115_generic_reducer_operation_ListOffsetArray_to_RegularArray():
    content = ak.contents.NumpyArray(np.array(primes[: 2 * 3 * 5], dtype=np.int64))
    offsets1 = ak.index.Index64(np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.int64))
    listoffsetarray = ak.contents.ListOffsetArray(offsets1, content)
    regulararray = listoffsetarray.to_RegularArray()
    cuda_listoffsetarray = ak.to_backend(listoffsetarray, "cuda")
    cuda_regulararray = ak.to_backend(regulararray, "cuda")

    assert to_list(cuda_listoffsetarray) == to_list(cuda_regulararray)
    del cuda_listoffsetarray, cuda_regulararray


def test_0115_generic_reducer_operation_dimension_optiontype_1():
    content = ak.contents.NumpyArray(np.array(primes[: 2 * 3 * 5], dtype=np.int64))
    offsets1 = ak.index.Index64(np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.int64))
    listoffsetarray = ak.contents.ListOffsetArray(offsets1, content)
    index = ak.index.Index64(np.array([5, -1, 3, 2, -1, 0], dtype=np.int64))
    indexedarray = ak.contents.IndexedOptionArray(index, listoffsetarray)
    depth2 = ak.contents.RegularArray(indexedarray, 3)
    depth2 = ak.to_backend(depth2, "cuda")

    assert to_list(depth2) == [
        [[101, 103, 107, 109, 113], None, [53, 59, 61, 67, 71]],
        [[31, 37, 41, 43, 47], None, [2, 3, 5, 7, 11]],
    ]
    assert to_list(ak.prod(depth2, axis=-1, keepdims=False, highlevel=False)) == [
        [101 * 103 * 107 * 109 * 113, None, 53 * 59 * 61 * 67 * 71],
        [31 * 37 * 41 * 43 * 47, None, 2 * 3 * 5 * 7 * 11],
    ]
    assert to_list(ak.prod(depth2, axis=-1, keepdims=True, highlevel=False)) == [
        [[101 * 103 * 107 * 109 * 113], None, [53 * 59 * 61 * 67 * 71]],
        [[31 * 37 * 41 * 43 * 47], None, [2 * 3 * 5 * 7 * 11]],
    ]
    del depth2


def test_0115_generic_reducer_operation_dimension_optiontype_2():
    content = ak.contents.NumpyArray(np.array(primes[: 2 * 3 * 5], dtype=np.int64))
    offsets1 = ak.index.Index64(np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.int64))
    listoffsetarray = ak.contents.ListOffsetArray(offsets1, content)
    index = ak.index.Index64(np.array([5, 4, 3, 2, 1, 0], dtype=np.int64))
    indexedarray = ak.contents.IndexedArray(index, listoffsetarray)
    depth2 = ak.contents.RegularArray(indexedarray, 3)
    depth2 = ak.to_backend(depth2, "cuda")

    assert to_list(depth2) == [
        [[101, 103, 107, 109, 113], [73, 79, 83, 89, 97], [53, 59, 61, 67, 71]],
        [[31, 37, 41, 43, 47], [13, 17, 19, 23, 29], [2, 3, 5, 7, 11]],
    ]
    assert to_list(ak.prod(depth2, axis=-1, highlevel=False)) == [
        [101 * 103 * 107 * 109 * 113, 73 * 79 * 83 * 89 * 97, 53 * 59 * 61 * 67 * 71],
        [31 * 37 * 41 * 43 * 47, 13 * 17 * 19 * 23 * 29, 2 * 3 * 5 * 7 * 11],
    ]
    assert to_list(ak.prod(depth2, axis=-1, keepdims=True, highlevel=False)) == [
        [
            [101 * 103 * 107 * 109 * 113],
            [73 * 79 * 83 * 89 * 97],
            [53 * 59 * 61 * 67 * 71],
        ],
        [[31 * 37 * 41 * 43 * 47], [13 * 17 * 19 * 23 * 29], [2 * 3 * 5 * 7 * 11]],
    ]
    del depth2


def test_0115_generic_reducer_operation_reproduce_numpy_1():
    content1 = ak.contents.NumpyArray(np.array(primes[: 2 * 3 * 5], dtype=np.int64))
    offsets1 = ak.index.Index64(np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.int64))
    offsets2 = ak.index.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(
        offsets2, ak.contents.ListOffsetArray(offsets1, content1)
    )
    depth2 = ak.to_backend(depth2, "cuda", highlevel=False)

    assert to_list(depth2) == [
        [[2, 3, 5, 7, 11], [13, 17, 19, 23, 29], [31, 37, 41, 43, 47]],
        [[53, 59, 61, 67, 71], [73, 79, 83, 89, 97], [101, 103, 107, 109, 113]],
    ]

    assert to_list(ak.prod(depth2, axis=-1, highlevel=False)) == [
        [2 * 3 * 5 * 7 * 11, 13 * 17 * 19 * 23 * 29, 31 * 37 * 41 * 43 * 47],
        [53 * 59 * 61 * 67 * 71, 73 * 79 * 83 * 89 * 97, 101 * 103 * 107 * 109 * 113],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), axis=-1, highlevel=False).form
        == ak.prod(depth2, axis=-1, highlevel=False).form
    )
    assert to_list(ak.prod(depth2, axis=2, highlevel=False)) == [
        [2 * 3 * 5 * 7 * 11, 13 * 17 * 19 * 23 * 29, 31 * 37 * 41 * 43 * 47],
        [53 * 59 * 61 * 67 * 71, 73 * 79 * 83 * 89 * 97, 101 * 103 * 107 * 109 * 113],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), axis=2, highlevel=False).form
        == ak.prod(depth2, axis=2, highlevel=False).form
    )

    assert to_list(ak.prod(depth2, axis=-2, highlevel=False)) == [
        [2 * 13 * 31, 3 * 17 * 37, 5 * 19 * 41, 7 * 23 * 43, 11 * 29 * 47],
        [53 * 73 * 101, 59 * 79 * 103, 61 * 83 * 107, 67 * 89 * 109, 71 * 97 * 113],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), axis=-2, highlevel=False).form
        == ak.prod(depth2, axis=-2, highlevel=False).form
    )
    assert to_list(ak.prod(depth2, axis=1, highlevel=False)) == [
        [2 * 13 * 31, 3 * 17 * 37, 5 * 19 * 41, 7 * 23 * 43, 11 * 29 * 47],
        [53 * 73 * 101, 59 * 79 * 103, 61 * 83 * 107, 67 * 89 * 109, 71 * 97 * 113],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), axis=1, highlevel=False).form
        == ak.prod(depth2, axis=1, highlevel=False).form
    )

    assert to_list(ak.prod(depth2, axis=-3, highlevel=False)) == [
        [2 * 53, 3 * 59, 5 * 61, 7 * 67, 11 * 71],
        [13 * 73, 17 * 79, 19 * 83, 23 * 89, 29 * 97],
        [31 * 101, 37 * 103, 41 * 107, 43 * 109, 47 * 113],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), axis=-3, highlevel=False).form
        == ak.prod(depth2, axis=-3, highlevel=False).form
    )
    assert to_list(ak.prod(depth2, axis=0, highlevel=False)) == [
        [2 * 53, 3 * 59, 5 * 61, 7 * 67, 11 * 71],
        [13 * 73, 17 * 79, 19 * 83, 23 * 89, 29 * 97],
        [31 * 101, 37 * 103, 41 * 107, 43 * 109, 47 * 113],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), axis=0, highlevel=False).form
        == ak.prod(depth2, axis=0, highlevel=False).form
    )
    del depth2


def test_0115_generic_reducer_operation_reproduce_numpy_2():
    content2 = ak.contents.NumpyArray(np.array(primes[:12], dtype=np.int64))
    offsets3 = ak.index.Index64(np.array([0, 4, 8, 12], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)
    depth1 = ak.to_backend(depth1, "cuda", highlevel=False)

    assert to_list(ak.prod(depth1, -1, highlevel=False)) == [
        2 * 3 * 5 * 7,
        11 * 13 * 17 * 19,
        23 * 29 * 31 * 37,
    ]
    assert (
        ak.prod(depth1.to_typetracer(), -1, highlevel=False).form
        == ak.prod(depth1, -1, highlevel=False).form
    )
    assert to_list(ak.prod(depth1, 1, highlevel=False)) == [
        2 * 3 * 5 * 7,
        11 * 13 * 17 * 19,
        23 * 29 * 31 * 37,
    ]
    assert (
        ak.prod(depth1.to_typetracer(), 1, highlevel=False).form
        == ak.prod(depth1, 1, highlevel=False).form
    )

    assert to_list(ak.prod(depth1, -2, highlevel=False)) == [
        2 * 11 * 23,
        3 * 13 * 29,
        5 * 17 * 31,
        7 * 19 * 37,
    ]
    assert (
        ak.prod(depth1.to_typetracer(), -2, highlevel=False).form
        == ak.prod(depth1, -2, highlevel=False).form
    )
    assert to_list(ak.prod(depth1, 0, highlevel=False)) == [
        2 * 11 * 23,
        3 * 13 * 29,
        5 * 17 * 31,
        7 * 19 * 37,
    ]
    assert (
        ak.prod(depth1.to_typetracer(), 0, highlevel=False).form
        == ak.prod(depth1, 0, highlevel=False).form
    )
    del depth1


def test_0115_generic_reducer_operation_gaps_1():
    content1 = ak.contents.NumpyArray(
        np.array([123] + primes[: 2 * 3 * 5], dtype=np.int64)
    )
    offsets1 = ak.index.Index64(np.array([0, 1, 6, 11, 16, 21, 26, 31], dtype=np.int64))
    offsets2 = ak.index.Index64(np.array([1, 4, 7], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(
        offsets2, ak.contents.ListOffsetArray(offsets1, content1)
    )
    depth2 = ak.to_backend(depth2, "cuda", highlevel=False)

    assert to_list(depth2) == [
        [[2, 3, 5, 7, 11], [13, 17, 19, 23, 29], [31, 37, 41, 43, 47]],
        [[53, 59, 61, 67, 71], [73, 79, 83, 89, 97], [101, 103, 107, 109, 113]],
    ]

    assert to_list(ak.prod(depth2, -3, highlevel=False)) == [
        [106, 177, 305, 469, 781],
        [949, 1343, 1577, 2047, 2813],
        [3131, 3811, 4387, 4687, 5311],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), -3, highlevel=False).form
        == ak.prod(depth2, -3, highlevel=False).form
    )
    del depth2


def test_0115_generic_reducer_operation_gaps_2():
    content1 = ak.contents.NumpyArray(np.array(primes[: 2 * 3 * 5 - 1], dtype=np.int64))
    offsets1 = ak.index.Index64(np.array([0, 5, 10, 15, 20, 25, 29], dtype=np.int64))
    offsets2 = ak.index.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(
        offsets2, ak.contents.ListOffsetArray(offsets1, content1)
    )
    depth2 = ak.to_backend(depth2, "cuda", highlevel=False)

    assert to_list(depth2) == [
        [[2, 3, 5, 7, 11], [13, 17, 19, 23, 29], [31, 37, 41, 43, 47]],
        [
            [53, 59, 61, 67, 71],
            [73, 79, 83, 89, 97],
            [
                101,
                103,
                107,
                109,
            ],
        ],
    ]

    assert to_list(ak.prod(depth2, -3, highlevel=False)) == [
        [106, 177, 305, 469, 781],
        [949, 1343, 1577, 2047, 2813],
        [3131, 3811, 4387, 4687, 47],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), -3, highlevel=False).form
        == ak.prod(depth2, -3, highlevel=False).form
    )
    del depth2


def test_0115_generic_reducer_operation_gaps_3():
    content1 = ak.contents.NumpyArray(np.array(primes[: 2 * 3 * 5 - 2], dtype=np.int64))
    offsets1 = ak.index.Index64(np.array([0, 5, 10, 15, 20, 25, 28], dtype=np.int64))
    offsets2 = ak.index.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(
        offsets2, ak.contents.ListOffsetArray(offsets1, content1)
    )
    depth2 = ak.to_backend(depth2, "cuda", highlevel=False)

    assert to_list(depth2) == [
        [[2, 3, 5, 7, 11], [13, 17, 19, 23, 29], [31, 37, 41, 43, 47]],
        [
            [53, 59, 61, 67, 71],
            [73, 79, 83, 89, 97],
            [
                101,
                103,
                107,
            ],
        ],
    ]

    assert to_list(ak.prod(depth2, -3, highlevel=False)) == [
        [106, 177, 305, 469, 781],
        [949, 1343, 1577, 2047, 2813],
        [3131, 3811, 4387, 43, 47],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), -3, highlevel=False).form
        == ak.prod(depth2, -3, highlevel=False).form
    )
    del depth2


def test_0115_generic_reducer_operation_gaps_4():
    content1 = ak.contents.NumpyArray(
        np.array(
            [
                2,
                3,
                5,
                7,
                11,
                13,
                17,
                19,
                23,
                29,
                31,
                37,
                41,
                43,
                47,
                53,
                59,
                61,
                67,
                71,
                73,
                79,
                83,
                89,
                101,
                103,
                107,
                109,
            ],
            dtype=np.int64,
        )
    )
    offsets1 = ak.index.Index64(np.array([0, 5, 10, 15, 20, 24, 28], dtype=np.int64))
    offsets2 = ak.index.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(
        offsets2, ak.contents.ListOffsetArray(offsets1, content1)
    )
    depth2 = ak.to_backend(depth2, "cuda", highlevel=False)

    assert to_list(depth2) == [
        [[2, 3, 5, 7, 11], [13, 17, 19, 23, 29], [31, 37, 41, 43, 47]],
        [
            [53, 59, 61, 67, 71],
            [
                73,
                79,
                83,
                89,
            ],
            [101, 103, 107, 109],
        ],
    ]

    assert to_list(ak.prod(depth2, -3, highlevel=False)) == [
        [106, 177, 305, 469, 781],
        [949, 1343, 1577, 2047, 29],
        [3131, 3811, 4387, 4687, 47],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), -3, highlevel=False).form
        == ak.prod(depth2, -3, highlevel=False).form
    )
    del depth2


def test_0115_generic_reducer_operation_gaps_5():
    content1 = ak.contents.NumpyArray(np.array(primes[1 : 2 * 3 * 5], dtype=np.int64))
    offsets1 = ak.index.Index64(np.array([0, 4, 9, 14, 19, 24, 29], dtype=np.int64))
    offsets2 = ak.index.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(
        offsets2, ak.contents.ListOffsetArray(offsets1, content1)
    )
    depth2 = ak.to_backend(depth2, "cuda", highlevel=False)

    assert to_list(depth2) == [
        [[3, 5, 7, 11], [13, 17, 19, 23, 29], [31, 37, 41, 43, 47]],
        [[53, 59, 61, 67, 71], [73, 79, 83, 89, 97], [101, 103, 107, 109, 113]],
    ]

    assert to_list(ak.prod(depth2, -3, highlevel=False)) == [
        [159, 295, 427, 737, 71],
        [949, 1343, 1577, 2047, 2813],
        [3131, 3811, 4387, 4687, 5311],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), -3, highlevel=False).form
        == ak.prod(depth2, -3, highlevel=False).form
    )
    del depth2


def test_0115_generic_reducer_operation_gaps_6():
    content1 = ak.contents.NumpyArray(np.array(primes[2 : 2 * 3 * 5], dtype=np.int64))
    offsets1 = ak.index.Index64(np.array([0, 3, 8, 13, 18, 23, 28], dtype=np.int64))
    offsets2 = ak.index.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(
        offsets2, ak.contents.ListOffsetArray(offsets1, content1)
    )
    depth2 = ak.to_backend(depth2, "cuda", highlevel=False)

    assert to_list(depth2) == [
        [[5, 7, 11], [13, 17, 19, 23, 29], [31, 37, 41, 43, 47]],
        [[53, 59, 61, 67, 71], [73, 79, 83, 89, 97], [101, 103, 107, 109, 113]],
    ]

    assert to_list(ak.prod(depth2, -3, highlevel=False)) == [
        [265, 413, 671, 67, 71],
        [949, 1343, 1577, 2047, 2813],
        [3131, 3811, 4387, 4687, 5311],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), -3, highlevel=False).form
        == ak.prod(depth2, -3, highlevel=False).form
    )
    del depth2


def test_0115_generic_reducer_operation_gaps_7():
    content1 = ak.contents.NumpyArray(
        np.array(
            [
                3,
                5,
                7,
                13,
                17,
                19,
                23,
                29,
                31,
                37,
                41,
                43,
                47,
                53,
                59,
                61,
                67,
                71,
                73,
                79,
                83,
                89,
                97,
                101,
                103,
                107,
                109,
                113,
            ],
            dtype=np.int64,
        )
    )
    offsets1 = ak.index.Index64(np.array([0, 3, 8, 13, 18, 23, 28], dtype=np.int64))
    offsets2 = ak.index.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(
        offsets2, ak.contents.ListOffsetArray(offsets1, content1)
    )
    depth2 = ak.to_backend(depth2, "cuda", highlevel=False)

    assert to_list(depth2) == [
        [
            [
                3,
                5,
                7,
            ],
            [13, 17, 19, 23, 29],
            [31, 37, 41, 43, 47],
        ],
        [[53, 59, 61, 67, 71], [73, 79, 83, 89, 97], [101, 103, 107, 109, 113]],
    ]

    assert to_list(ak.prod(depth2, -3, highlevel=False)) == [
        [159, 295, 427, 67, 71],
        [949, 1343, 1577, 2047, 2813],
        [3131, 3811, 4387, 4687, 5311],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), -3, highlevel=False).form
        == ak.prod(depth2, -3, highlevel=False).form
    )
    del depth2


def test_0115_generic_reducer_operation_gaps_8():
    content1 = ak.contents.NumpyArray(
        np.array(
            [
                3,
                5,
                7,
                11,
                13,
                17,
                19,
                23,
                31,
                37,
                41,
                43,
                47,
                53,
                59,
                61,
                67,
                71,
                73,
                79,
                83,
                89,
                97,
                101,
                103,
                107,
                109,
                113,
            ],
            dtype=np.int64,
        )
    )
    offsets1 = ak.index.Index64(np.array([0, 4, 8, 13, 18, 23, 28], dtype=np.int64))
    offsets2 = ak.index.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(
        offsets2, ak.contents.ListOffsetArray(offsets1, content1)
    )
    depth2 = ak.to_backend(depth2, "cuda", highlevel=False)

    assert to_list(depth2) == [
        [[3, 5, 7, 11], [13, 17, 19, 23], [31, 37, 41, 43, 47]],
        [[53, 59, 61, 67, 71], [73, 79, 83, 89, 97], [101, 103, 107, 109, 113]],
    ]

    assert to_list(ak.prod(depth2, -3, highlevel=False)) == [
        [159, 295, 427, 737, 71],
        [949, 1343, 1577, 2047, 97],
        [3131, 3811, 4387, 4687, 5311],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), -3, highlevel=False).form
        == ak.prod(depth2, -3, highlevel=False).form
    )
    del depth2


def test_0115_generic_reducer_operation_gaps_9():
    content1 = ak.contents.NumpyArray(
        np.array(
            [
                2,
                3,
                5,
                7,
                11,
                13,
                17,
                19,
                23,
                29,
                31,
                37,
                41,
                43,
                53,
                59,
                61,
                67,
                71,
                73,
                79,
                83,
                89,
                97,
                101,
                103,
                107,
                109,
            ],
            dtype=np.int64,
        )
    )
    offsets1 = ak.index.Index64(np.array([0, 5, 10, 14, 19, 24, 28], dtype=np.int64))
    offsets2 = ak.index.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(
        offsets2, ak.contents.ListOffsetArray(offsets1, content1)
    )
    depth2 = ak.to_backend(depth2, "cuda", highlevel=False)

    assert to_list(depth2) == [
        [[2, 3, 5, 7, 11], [13, 17, 19, 23, 29], [31, 37, 41, 43]],
        [[53, 59, 61, 67, 71], [73, 79, 83, 89, 97], [101, 103, 107, 109]],
    ]

    assert to_list(ak.prod(depth2, -3, highlevel=False)) == [
        [106, 177, 305, 469, 781],
        [949, 1343, 1577, 2047, 2813],
        [3131, 3811, 4387, 4687],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), -3, highlevel=False).form
        == ak.prod(depth2, -3, highlevel=False).form
    )
    del depth2


def test_0115_generic_reducer_operation_gaps_10():
    content1 = ak.contents.NumpyArray(
        np.array(
            [
                2,
                3,
                5,
                7,
                11,
                13,
                17,
                19,
                23,
                31,
                37,
                41,
                43,
                47,
                53,
                59,
                61,
                67,
                71,
                73,
                79,
                83,
                89,
                101,
                103,
                107,
                109,
                113,
            ],
            dtype=np.int64,
        )
    )
    offsets1 = ak.index.Index64(np.array([0, 5, 9, 14, 19, 23, 28], dtype=np.int64))
    offsets2 = ak.index.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(
        offsets2, ak.contents.ListOffsetArray(offsets1, content1)
    )
    depth2 = ak.to_backend(depth2, "cuda", highlevel=False)

    assert to_list(depth2) == [
        [[2, 3, 5, 7, 11], [13, 17, 19, 23], [31, 37, 41, 43, 47]],
        [[53, 59, 61, 67, 71], [73, 79, 83, 89], [101, 103, 107, 109, 113]],
    ]

    assert to_list(ak.prod(depth2, -3, highlevel=False)) == [
        [106, 177, 305, 469, 781],
        [949, 1343, 1577, 2047],
        [3131, 3811, 4387, 4687, 5311],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), -3, highlevel=False).form
        == ak.prod(depth2, -3, highlevel=False).form
    )
    del depth2


def test_0115_generic_reducer_operation_gaps_11():
    content1 = ak.contents.NumpyArray(np.array(primes[:9], dtype=np.int64))
    offsets1 = ak.index.Index64(np.array([0, 3, 4, 6, 6, 7, 9], dtype=np.int64))
    offsets2 = ak.index.Index64(np.array([0, 2, 4, 6], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(
        offsets2, ak.contents.ListOffsetArray(offsets1, content1)
    )
    depth2 = ak.to_backend(depth2, "cuda", highlevel=False)

    assert to_list(depth2) == [[[2, 3, 5], [7]], [[11, 13], []], [[17], [19, 23]]]

    assert to_list(ak.prod(depth2, -3, highlevel=False)) == [
        [2 * 11 * 17, 3 * 13, 5],
        [7 * 19, 23],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), -3, highlevel=False).form
        == ak.prod(depth2, -3, highlevel=False).form
    )
    del depth2


def test_0115_generic_reducer_operation_gaps_12():
    content1 = ak.contents.NumpyArray(np.array(primes[:9], dtype=np.int64))
    offsets1 = ak.index.Index64(np.array([0, 3, 4, 6, 7, 9], dtype=np.int64))
    offsets2 = ak.index.Index64(np.array([0, 2, 3, 5], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(
        offsets2, ak.contents.ListOffsetArray(offsets1, content1)
    )
    depth2 = ak.to_backend(depth2, "cuda", highlevel=False)

    assert to_list(depth2) == [[[2, 3, 5], [7]], [[11, 13]], [[17], [19, 23]]]

    assert to_list(ak.prod(depth2, -3, highlevel=False)) == [
        [2 * 11 * 17, 3 * 13, 5],
        [7 * 19, 23],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), -3, highlevel=False).form
        == ak.prod(depth2, -3, highlevel=False).form
    )
    del depth2


def test_0115_generic_reducer_operation_gaps_13():
    content1 = ak.contents.NumpyArray(np.array(primes[:10], dtype=np.int64))
    offsets1 = ak.index.Index64(np.array([0, 3, 5, 6, 8, 9, 10], dtype=np.int64))
    offsets2 = ak.index.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(
        offsets2, ak.contents.ListOffsetArray(offsets1, content1)
    )
    depth2 = ak.to_backend(depth2, "cuda", highlevel=False)

    assert to_list(depth2) == [[[2, 3, 5], [7, 11], [13]], [[17, 19], [23], [29]]]

    assert to_list(ak.prod(depth2, -3, highlevel=False)) == [
        [34, 57, 5],
        [161, 11],
        [377],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), -3, highlevel=False).form
        == ak.prod(depth2, -3, highlevel=False).form
    )
    del depth2


def test_0115_generic_reducer_operation_gaps_14():
    content1 = ak.contents.NumpyArray(np.array(primes[:9], dtype=np.int64))
    offsets1 = ak.index.Index64(np.array([0, 3, 3, 5, 6, 8, 9], dtype=np.int64))
    offsets2 = ak.index.Index64(np.array([0, 4, 6], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(
        offsets2, ak.contents.ListOffsetArray(offsets1, content1)
    )
    depth2 = ak.to_backend(depth2, "cuda", highlevel=False)

    assert to_list(depth2) == [[[2, 3, 5], [], [7, 11], [13]], [[17, 19], [23]]]

    assert to_list(ak.prod(depth2, -3, highlevel=False)) == [
        [34, 57, 5],
        [23],
        [7, 11],
        [13],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), -3, highlevel=False).form
        == ak.prod(depth2, -3, highlevel=False).form
    )
    del depth2


def test_0115_generic_reducer_operation_gaps_15():
    content1 = ak.contents.NumpyArray(np.array(primes[:9], dtype=np.int64))
    offsets1 = ak.index.Index64(np.array([0, 3, 3, 5, 6, 8, 9], dtype=np.int64))
    offsets2 = ak.index.Index64(np.array([0, 4, 4, 6], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(
        offsets2, ak.contents.ListOffsetArray(offsets1, content1)
    )
    depth2 = ak.to_backend(depth2, "cuda", highlevel=False)

    assert to_list(depth2) == [[[2, 3, 5], [], [7, 11], [13]], [], [[17, 19], [23]]]

    assert to_list(ak.prod(depth2, -3, highlevel=False)) == [
        [34, 57, 5],
        [23],
        [7, 11],
        [13],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), -3, highlevel=False).form
        == ak.prod(depth2, -3, highlevel=False).form
    )
    del depth2


def test_0115_generic_reducer_operation_gaps_16():
    content1 = ak.contents.NumpyArray(np.array(primes[: 2 * 3 * 5], dtype=np.int64))
    offsets1 = ak.index.Index64(np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.int64))
    offsets2 = ak.index.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(
        offsets2, ak.contents.ListOffsetArray(offsets1, content1)
    )
    depth2 = ak.to_backend(depth2, "cuda", highlevel=False)

    assert to_list(depth2) == [
        [[2, 3, 5, 7, 11], [13, 17, 19, 23, 29], [31, 37, 41, 43, 47]],
        [[53, 59, 61, 67, 71], [73, 79, 83, 89, 97], [101, 103, 107, 109, 113]],
    ]

    assert to_list(ak.prod(depth2, -1, highlevel=False)) == [
        [2 * 3 * 5 * 7 * 11, 13 * 17 * 19 * 23 * 29, 31 * 37 * 41 * 43 * 47],
        [53 * 59 * 61 * 67 * 71, 73 * 79 * 83 * 89 * 97, 101 * 103 * 107 * 109 * 113],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), -1, highlevel=False).form
        == ak.prod(depth2, -1, highlevel=False).form
    )

    assert to_list(ak.prod(depth2, -2, highlevel=False)) == [
        [2 * 13 * 31, 3 * 17 * 37, 5 * 19 * 41, 7 * 23 * 43, 11 * 29 * 47],
        [53 * 73 * 101, 59 * 79 * 103, 61 * 83 * 107, 67 * 89 * 109, 71 * 97 * 113],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), -2, highlevel=False).form
        == ak.prod(depth2, -2, highlevel=False).form
    )
    del depth2


def test_0115_generic_reducer_operation_gaps_17():
    content1 = ak.contents.NumpyArray(np.array(primes[:9], dtype=np.int64))
    offsets1 = ak.index.Index64(np.array([0, 3, 3, 5, 6, 8, 9], dtype=np.int64))
    offsets2 = ak.index.Index64(np.array([0, 4, 4, 6], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(
        offsets2, ak.contents.ListOffsetArray(offsets1, content1)
    )
    depth2 = ak.to_backend(depth2, "cuda", highlevel=False)

    assert to_list(depth2) == [
        [[2, 3, 5], [], [7, 11], [13]],
        [],
        [[17, 19], [23]],
    ]

    assert to_list(ak.prod(depth2, -1, highlevel=False)) == [
        [2 * 3 * 5, 1, 7 * 11, 13],
        [],
        [17 * 19, 23],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), -1, highlevel=False).form
        == ak.prod(depth2, -1, highlevel=False).form
    )

    assert to_list(ak.prod(depth2, -2, highlevel=False)) == [
        [2 * 7 * 13, 3 * 11, 5],
        [],
        [17 * 23, 19],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), -2, highlevel=False).form
        == ak.prod(depth2, -2, highlevel=False).form
    )

    assert to_list(ak.prod(depth2, -3, highlevel=False)) == [
        [2 * 17, 3 * 19, 5],
        [23],
        [7, 11],
        [13],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), -3, highlevel=False).form
        == ak.prod(depth2, -3, highlevel=False).form
    )
    del depth2
