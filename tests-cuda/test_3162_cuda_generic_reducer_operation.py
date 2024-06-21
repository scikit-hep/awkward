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


def test_0115_generic_reducer_operation_complicated():
    offsets1 = ak.index.Index64(np.array([0, 3, 3, 5], dtype=np.int64))
    content1 = ak.contents.ListOffsetArray(
        offsets1, ak.contents.NumpyArray(np.array(primes[:5], dtype=np.int64))
    )
    offsets2 = ak.index.Index64(np.array([0, 3, 3, 5, 6, 8, 9], dtype=np.int64))
    offsets3 = ak.index.Index64(np.array([0, 4, 4, 6], dtype=np.int64))
    content2 = ak.contents.ListOffsetArray(
        offsets3,
        ak.contents.ListOffsetArray(
            offsets2, ak.contents.NumpyArray(np.array(primes[:9], dtype=np.int64))
        ),
    )
    offsets4 = ak.index.Index64(np.array([0, 1, 1, 3], dtype=np.int64))
    complicated = ak.contents.ListOffsetArray(
        offsets4, ak.contents.RecordArray([content1, content2], ["x", "y"])
    )
    complicated = ak.to_backend(complicated, "cuda", highlevel=False)

    assert to_list(complicated) == [
        [{"x": [2, 3, 5], "y": [[2, 3, 5], [], [7, 11], [13]]}],
        [],
        [{"x": [], "y": []}, {"x": [7, 11], "y": [[17, 19], [23]]}],
    ]

    assert to_list(complicated["x"]) == [[[2, 3, 5]], [], [[], [7, 11]]]
    assert complicated.to_typetracer()["x"].form == complicated["x"].form
    assert to_list(complicated["y"]) == [
        [[[2, 3, 5], [], [7, 11], [13]]],
        [],
        [[], [[17, 19], [23]]],
    ]
    assert complicated.to_typetracer()["y"].form == complicated["y"].form

    with pytest.raises(TypeError):
        to_list(ak.prod(complicated, -1, highlevel=False))

    with pytest.raises(TypeError):
        assert (
            ak.prod(complicated.to_typetracer(), -1, highlevel=False).form
            == ak.prod(complicated, -1, highlevel=False).form
        )

    assert to_list(ak.prod(complicated["x"], -1, highlevel=False)) == [
        [30],
        [],
        [1, 77],
    ]
    assert (
        ak.prod(complicated.to_typetracer()["x"], -1, highlevel=False).form
        == ak.prod(complicated["x"], -1, highlevel=False).form
    )
    assert to_list(ak.prod(complicated["y"], -1, highlevel=False)) == [
        [[30, 1, 77, 13]],
        [],
        [[], [323, 23]],
    ]
    assert (
        ak.prod(complicated.to_typetracer()["y"], -1, highlevel=False).form
        == ak.prod(complicated["y"], -1, highlevel=False).form
    )

    with pytest.raises(TypeError):
        to_list(ak.prod(complicated, -2, highlevel=False))

    with pytest.raises(TypeError):
        assert (
            ak.prod(complicated.to_typetracer(), -2, highlevel=False).form
            == ak.prod(complicated, -2, highlevel=False).form
        )
    assert to_list(ak.prod(complicated["x"], -2, highlevel=False)) == [
        [2, 3, 5],
        [],
        [7, 11],
    ]
    assert (
        ak.prod(complicated.to_typetracer()["x"], -2, highlevel=False).form
        == ak.prod(complicated["x"], -2, highlevel=False).form
    )
    assert to_list(ak.prod(complicated["y"], -2, highlevel=False)) == [
        [[182, 33, 5]],
        [],
        [[], [391, 19]],
    ]
    assert (
        ak.prod(complicated.to_typetracer()["y"], -2, highlevel=False).form
        == ak.prod(complicated["y"], -2, highlevel=False).form
    )

    assert to_list(complicated[0]) == [
        {"x": [2, 3, 5], "y": [[2, 3, 5], [], [7, 11], [13]]}
    ]
    assert complicated.to_typetracer()[0].form == complicated[0].form

    with pytest.raises(TypeError):
        to_list(ak.prod(complicated[0], -1, highlevel=False))

    with pytest.raises(TypeError):
        to_list(ak.prod(complicated.to_typetracer()[0], -1, highlevel=False))
    del complicated


def test_0115_generic_reducer_operation_EmptyArray():
    offsets = ak.index.Index64(np.array([0, 0, 0, 0], dtype=np.int64))
    array = ak.contents.ListOffsetArray(offsets, ak.contents.EmptyArray())
    array = ak.to_backend(array, "cuda")

    assert to_list(array) == [[], [], []]

    assert to_list(ak.prod(array, -1, highlevel=False)) == [1, 1, 1]
    assert (
        ak.prod(array.to_typetracer(), -1, highlevel=False).form
        == ak.prod(array, -1, highlevel=False).form
    )

    offsets = ak.index.Index64(np.array([0, 0, 0, 0], dtype=np.int64))
    array = ak.contents.ListOffsetArray(
        offsets, ak.contents.NumpyArray(np.array([], dtype=np.int64))
    )
    array = ak.to_backend(array, "cuda")

    assert to_list(array) == [[], [], []]

    assert to_list(ak.prod(array, -1, highlevel=False)) == [1, 1, 1]
    assert (
        ak.prod(array.to_typetracer(), -1, highlevel=False).form
        == ak.prod(array, -1, highlevel=False).form
    )
    del array


def test_0115_generic_reducer_operation_IndexedOptionArray_1():
    content = ak.contents.NumpyArray(np.array(primes[: 2 * 3 * 5], dtype=np.int64))
    offsets1 = ak.index.Index64(np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.int64))
    listoffsetarray = ak.contents.ListOffsetArray(offsets1, content)
    index = ak.index.Index64(np.array([5, 4, 3, 2, 1, 0], dtype=np.int64))
    indexedarray = ak.contents.IndexedArray(index, listoffsetarray)
    offsets2 = ak.index.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(offsets2, indexedarray)
    depth2 = ak.to_backend(depth2, "cuda")

    assert to_list(depth2) == [
        [[101, 103, 107, 109, 113], [73, 79, 83, 89, 97], [53, 59, 61, 67, 71]],
        [[31, 37, 41, 43, 47], [13, 17, 19, 23, 29], [2, 3, 5, 7, 11]],
    ]

    assert to_list(ak.prod(depth2, -1, highlevel=False)) == [
        [101 * 103 * 107 * 109 * 113, 73 * 79 * 83 * 89 * 97, 53 * 59 * 61 * 67 * 71],
        [31 * 37 * 41 * 43 * 47, 13 * 17 * 19 * 23 * 29, 2 * 3 * 5 * 7 * 11],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), -1, highlevel=False).form
        == ak.prod(depth2, -1, highlevel=False).form
    )

    assert to_list(ak.prod(depth2, -2, highlevel=False)) == [
        [101 * 73 * 53, 103 * 79 * 59, 107 * 83 * 61, 109 * 89 * 67, 113 * 97 * 71],
        [31 * 13 * 2, 37 * 17 * 3, 41 * 19 * 5, 43 * 23 * 7, 47 * 29 * 11],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), -2, highlevel=False).form
        == ak.prod(depth2, -2, highlevel=False).form
    )

    assert to_list(ak.prod(depth2, -3, highlevel=False)) == [
        [101 * 31, 103 * 37, 107 * 41, 109 * 43, 113 * 47],
        [73 * 13, 79 * 17, 83 * 19, 89 * 23, 97 * 29],
        [53 * 2, 59 * 3, 61 * 5, 67 * 7, 71 * 11],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), -3, highlevel=False).form
        == ak.prod(depth2, -3, highlevel=False).form
    )
    del depth2


def test_0115_generic_reducer_operation_IndexedOptionArray_2():
    content = ak.contents.NumpyArray(
        np.array(
            [
                2,
                3,
                5,
                7,
                11,
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
                101,
                103,
                107,
                109,
                113,
            ],
            dtype=np.int64,
        )
    )
    offsets1 = ak.index.Index64(np.array([0, 5, 10, 15, 20], dtype=np.int64))
    listoffsetarray = ak.contents.ListOffsetArray(offsets1, content)
    index = ak.index.Index64(np.array([3, -1, 2, 1, -1, 0], dtype=np.int64))
    indexedoptionarray = ak.contents.IndexedOptionArray(index, listoffsetarray)
    offsets2 = ak.index.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(offsets2, indexedoptionarray)
    depth2 = ak.to_backend(depth2, "cuda")

    assert to_list(depth2) == [
        [[101, 103, 107, 109, 113], None, [53, 59, 61, 67, 71]],
        [[31, 37, 41, 43, 47], None, [2, 3, 5, 7, 11]],
    ]

    assert to_list(ak.prod(depth2, -1, highlevel=False)) == [
        [101 * 103 * 107 * 109 * 113, None, 53 * 59 * 61 * 67 * 71],
        [31 * 37 * 41 * 43 * 47, None, 2 * 3 * 5 * 7 * 11],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), -1, highlevel=False).form
        == ak.prod(depth2, -1, highlevel=False).form
    )

    assert to_list(ak.prod(depth2, -2, highlevel=False)) == [
        [101 * 53, 103 * 59, 107 * 61, 109 * 67, 113 * 71],
        [31 * 2, 37 * 3, 41 * 5, 43 * 7, 47 * 11],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), -2, highlevel=False).form
        == ak.prod(depth2, -2, highlevel=False).form
    )

    assert to_list(ak.prod(depth2, -3, highlevel=False)) == [
        [101 * 31, 103 * 37, 107 * 41, 109 * 43, 113 * 47],
        [],
        [53 * 2, 59 * 3, 61 * 5, 67 * 7, 71 * 11],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), -3, highlevel=False).form
        == ak.prod(depth2, -3, highlevel=False).form
    )
    del depth2


def test_0115_generic_reducer_operation_IndexedOptionArray_3():
    content = ak.contents.NumpyArray(
        np.array(
            [
                2,
                3,
                5,
                7,
                11,
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
                101,
                103,
                107,
                109,
                113,
            ],
            dtype=np.int64,
        )
    )
    index = ak.index.Index64(
        np.array(
            [
                15,
                16,
                17,
                18,
                19,
                -1,
                -1,
                -1,
                -1,
                -1,
                10,
                11,
                12,
                13,
                14,
                5,
                6,
                7,
                8,
                9,
                -1,
                -1,
                -1,
                -1,
                -1,
                0,
                1,
                2,
                3,
                4,
            ],
            dtype=np.int64,
        )
    )
    indexedoptionarray = ak.contents.IndexedOptionArray(index, content)
    offsets1 = ak.index.Index64(np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.int64))
    listoffsetarray = ak.contents.ListOffsetArray(offsets1, indexedoptionarray)
    offsets2 = ak.index.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(offsets2, listoffsetarray)
    depth2 = ak.to_backend(depth2, "cuda")

    assert to_list(depth2) == [
        [
            [101, 103, 107, 109, 113],
            [None, None, None, None, None],
            [53, 59, 61, 67, 71],
        ],
        [[31, 37, 41, 43, 47], [None, None, None, None, None], [2, 3, 5, 7, 11]],
    ]

    assert to_list(ak.prod(depth2, -1, highlevel=False)) == [
        [101 * 103 * 107 * 109 * 113, 1 * 1 * 1 * 1 * 1, 53 * 59 * 61 * 67 * 71],
        [31 * 37 * 41 * 43 * 47, 1 * 1 * 1 * 1 * 1, 2 * 3 * 5 * 7 * 11],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), -1, highlevel=False).form
        == ak.prod(depth2, -1, highlevel=False).form
    )

    assert to_list(ak.prod(depth2, -2, highlevel=False)) == [
        [101 * 53, 103 * 59, 107 * 61, 109 * 67, 113 * 71],
        [31 * 2, 37 * 3, 41 * 5, 43 * 7, 47 * 11],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), -2, highlevel=False).form
        == ak.prod(depth2, -2, highlevel=False).form
    )

    assert to_list(ak.prod(depth2, -3, highlevel=False)) == [
        [101 * 31, 103 * 37, 107 * 41, 109 * 43, 113 * 47],
        [1, 1, 1, 1, 1],
        [53 * 2, 59 * 3, 61 * 5, 67 * 7, 71 * 11],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), -3, highlevel=False).form
        == ak.prod(depth2, -3, highlevel=False).form
    )
    del depth2


def test_0115_generic_reducer_operation_IndexedOptionArray_4():
    content = ak.contents.NumpyArray(
        np.array(
            [
                2,
                3,
                5,
                7,
                11,
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
                101,
                103,
                107,
                109,
                113,
            ],
            dtype=np.int64,
        )
    )
    index = ak.index.Index64(
        np.array(
            [
                15,
                16,
                17,
                18,
                19,
                -1,
                10,
                11,
                12,
                13,
                14,
                5,
                6,
                7,
                8,
                9,
                -1,
                0,
                1,
                2,
                3,
                4,
            ],
            dtype=np.int64,
        )
    )
    indexedoptionarray = ak.contents.IndexedOptionArray(index, content)
    offsets1 = ak.index.Index64(np.array([0, 5, 6, 11, 16, 17, 22], dtype=np.int64))
    listoffsetarray = ak.contents.ListOffsetArray(offsets1, indexedoptionarray)
    offsets2 = ak.index.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(offsets2, listoffsetarray)
    depth2 = ak.to_backend(depth2, "cuda")

    assert to_list(depth2) == [
        [[101, 103, 107, 109, 113], [None], [53, 59, 61, 67, 71]],
        [[31, 37, 41, 43, 47], [None], [2, 3, 5, 7, 11]],
    ]

    assert to_list(ak.prod(depth2, -1, highlevel=False)) == [
        [101 * 103 * 107 * 109 * 113, 1, 53 * 59 * 61 * 67 * 71],
        [31 * 37 * 41 * 43 * 47, 1, 2 * 3 * 5 * 7 * 11],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), -1, highlevel=False).form
        == ak.prod(depth2, -1, highlevel=False).form
    )

    assert to_list(ak.prod(depth2, -2, highlevel=False)) == [
        [101 * 53, 103 * 59, 107 * 61, 109 * 67, 113 * 71],
        [31 * 2, 37 * 3, 41 * 5, 43 * 7, 47 * 11],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), -2, highlevel=False).form
        == ak.prod(depth2, -2, highlevel=False).form
    )

    assert to_list(ak.prod(depth2, -3, highlevel=False)) == [
        [101 * 31, 103 * 37, 107 * 41, 109 * 43, 113 * 47],
        [1],
        [53 * 2, 59 * 3, 61 * 5, 67 * 7, 71 * 11],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), -3, highlevel=False).form
        == ak.prod(depth2, -3, highlevel=False).form
    )
    del depth2


def test_0115_generic_reducer_operation_sum():
    content2 = ak.contents.NumpyArray(
        np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], dtype=np.int64)
    )
    offsets3 = ak.index.Index64(np.array([0, 4, 8, 12], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)
    depth1 = ak.to_backend(depth1, "cuda", highlevel=False)

    assert to_list(ak.sum(depth1, -1, highlevel=False)) == [
        1 + 2 + 4 + 8,
        16 + 32 + 64 + 128,
        256 + 512 + 1024 + 2048,
    ]
    assert (
        ak.sum(depth1.to_typetracer(), -1, highlevel=False).form
        == ak.sum(depth1, -1, highlevel=False).form
    )
    assert to_list(ak.sum(depth1, 1, highlevel=False)) == [
        1 + 2 + 4 + 8,
        16 + 32 + 64 + 128,
        256 + 512 + 1024 + 2048,
    ]
    assert (
        ak.sum(depth1.to_typetracer(), 1, highlevel=False).form
        == ak.sum(depth1, 1, highlevel=False).form
    )

    assert to_list(ak.sum(depth1, -2, highlevel=False)) == [
        1 + 16 + 256,
        2 + 32 + 512,
        4 + 64 + 1024,
        8 + 128 + 2048,
    ]
    assert (
        ak.sum(depth1.to_typetracer(), -2, highlevel=False).form
        == ak.sum(depth1, -2, highlevel=False).form
    )
    assert to_list(ak.sum(depth1, 0, highlevel=False)) == [
        1 + 16 + 256,
        2 + 32 + 512,
        4 + 64 + 1024,
        8 + 128 + 2048,
    ]
    assert (
        ak.sum(depth1.to_typetracer(), 0, highlevel=False).form
        == ak.sum(depth1, 0, highlevel=False).form
    )
    del depth1


def test_0115_generic_reducer_operation_any():
    content2 = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 0.0, 2.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    offsets3 = ak.index.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)
    depth1 = ak.to_backend(depth1, "cuda", highlevel=False)

    assert to_list(depth1) == [
        [1.1, 2.2, 3.3],
        [0.0, 2.2, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]

    assert to_list(ak.any(depth1, -1, highlevel=False)) == [True, True, False]
    assert (
        ak.any(depth1.to_typetracer(), -1, highlevel=False).form
        == ak.any(depth1, -1, highlevel=False).form
    )
    assert to_list(ak.any(depth1, 1, highlevel=False)) == [True, True, False]
    assert (
        ak.any(depth1.to_typetracer(), 1, highlevel=False).form
        == ak.any(depth1, 1, highlevel=False).form
    )

    assert to_list(ak.any(depth1, -2, highlevel=False)) == [True, True, True, False]
    assert (
        ak.any(depth1.to_typetracer(), -2, highlevel=False).form
        == ak.any(depth1, -2, highlevel=False).form
    )
    assert to_list(ak.any(depth1, 0, highlevel=False)) == [True, True, True, False]
    assert (
        ak.any(depth1.to_typetracer(), 0, highlevel=False).form
        == ak.any(depth1, 0, highlevel=False).form
    )
    del depth1


def test_0115_generic_reducer_operation_all():
    content2 = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 0.0, 2.2, 0.0, 0.0, 2.2, 0.0, 4.4])
    )
    offsets3 = ak.index.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)
    depth1 = ak.to_backend(depth1, "cuda", highlevel=False)

    assert to_list(depth1) == [
        [1.1, 2.2, 3.3],
        [0.0, 2.2, 0.0],
        [0.0, 2.2, 0.0, 4.4],
    ]

    assert to_list(ak.all(depth1, -1, highlevel=False)) == [True, False, False]
    assert (
        ak.all(depth1.to_typetracer(), -1, highlevel=False).form
        == ak.all(depth1, -1, highlevel=False).form
    )
    assert to_list(ak.all(depth1, 1, highlevel=False)) == [True, False, False]
    assert (
        ak.all(depth1.to_typetracer(), 1, highlevel=False).form
        == ak.all(depth1, 1, highlevel=False).form
    )

    assert to_list(ak.all(depth1, -2, highlevel=False)) == [False, True, False, True]
    assert (
        ak.all(depth1.to_typetracer(), -2, highlevel=False).form
        == ak.all(depth1, -2, highlevel=False).form
    )
    assert to_list(ak.all(depth1, 0, highlevel=False)) == [False, True, False, True]
    assert (
        ak.all(depth1.to_typetracer(), 0, highlevel=False).form
        == ak.all(depth1, 0, highlevel=False).form
    )
    del depth1


def test_0115_generic_reducer_operation_count():
    content2 = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 0.0, 2.2, 0.0, 0.0, 2.2, 0.0, 4.4])
    )
    offsets3 = ak.index.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)
    depth1 = ak.to_backend(depth1, "cuda", highlevel=False)

    assert to_list(depth1) == [
        [1.1, 2.2, 3.3],
        [0.0, 2.2, 0.0],
        [0.0, 2.2, 0.0, 4.4],
    ]

    assert to_list(ak.count(depth1, -1, highlevel=False)) == [3, 3, 4]
    assert (
        ak.count(depth1.to_typetracer(), -1, highlevel=False).form
        == ak.count(depth1, -1, highlevel=False).form
    )
    assert to_list(ak.count(depth1, 1, highlevel=False)) == [3, 3, 4]
    assert (
        ak.count(depth1.to_typetracer(), 1, highlevel=False).form
        == ak.count(depth1, 1, highlevel=False).form
    )

    assert to_list(ak.count(depth1, -2, highlevel=False)) == [3, 3, 3, 1]
    assert (
        ak.count(depth1.to_typetracer(), -2, highlevel=False).form
        == ak.count(depth1, -2, highlevel=False).form
    )
    assert to_list(ak.count(depth1, 0, highlevel=False)) == [3, 3, 3, 1]
    assert (
        ak.count(depth1.to_typetracer(), 0, highlevel=False).form
        == ak.count(depth1, 0, highlevel=False).form
    )
    del depth1


def test_0115_generic_reducer_operation_count_nonzero():
    content2 = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 0.0, 2.2, 0.0, 0.0, 2.2, 0.0, 4.4])
    )
    offsets3 = ak.index.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)
    depth1 = ak.to_backend(depth1, "cuda", highlevel=False)

    assert to_list(depth1) == [
        [1.1, 2.2, 3.3],
        [0.0, 2.2, 0.0],
        [0.0, 2.2, 0.0, 4.4],
    ]

    assert to_list(ak.count_nonzero(depth1, -1, highlevel=False)) == [3, 1, 2]
    assert (
        ak.count_nonzero(depth1.to_typetracer(), -1, highlevel=False).form
        == ak.count_nonzero(depth1, -1, highlevel=False).form
    )
    assert to_list(ak.count_nonzero(depth1, 1, highlevel=False)) == [3, 1, 2]
    assert (
        ak.count_nonzero(depth1.to_typetracer(), 1, highlevel=False).form
        == ak.count_nonzero(depth1, 1, highlevel=False).form
    )

    assert to_list(ak.count_nonzero(depth1, -2, highlevel=False)) == [1, 3, 1, 1]
    assert (
        ak.count_nonzero(depth1.to_typetracer(), -2, highlevel=False).form
        == ak.count_nonzero(depth1, -2, highlevel=False).form
    )
    assert to_list(ak.count_nonzero(depth1, 0, highlevel=False)) == [1, 3, 1, 1]
    assert (
        ak.count_nonzero(depth1.to_typetracer(), 0, highlevel=False).form
        == ak.count_nonzero(depth1, 0, highlevel=False).form
    )
    del depth1


def test_0115_generic_reducer_operation_count_min_1():
    content2 = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 0.0, 2.2, 0.0, 0.0, 2.2, 0.0, 4.4])
    )
    offsets3 = ak.index.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)
    depth1 = ak.to_backend(depth1, "cuda", highlevel=False)

    assert to_list(depth1) == [
        [1.1, 2.2, 3.3],
        [0.0, 2.2, 0.0],
        [0.0, 2.2, 0.0, 4.4],
    ]

    assert to_list(ak.min(depth1, -1, highlevel=False)) == [1.1, 0.0, 0.0]
    assert (
        ak.min(depth1.to_typetracer(), -1, highlevel=False).form
        == ak.min(depth1, -1, highlevel=False).form
    )
    assert to_list(ak.min(depth1, 1, highlevel=False)) == [1.1, 0.0, 0.0]
    assert (
        ak.min(depth1.to_typetracer(), 1, highlevel=False).form
        == ak.min(depth1, 1, highlevel=False).form
    )

    assert to_list(ak.min(depth1, -2, highlevel=False)) == [0.0, 2.2, 0.0, 4.4]
    assert (
        ak.min(depth1.to_typetracer(), -2, highlevel=False).form
        == ak.min(depth1, -2, highlevel=False).form
    )
    assert to_list(ak.min(depth1, 0, highlevel=False)) == [0.0, 2.2, 0.0, 4.4]
    assert (
        ak.min(depth1.to_typetracer(), 0, highlevel=False).form
        == ak.min(depth1, 0, highlevel=False).form
    )
    del depth1


def test_0115_generic_reducer_operation_count_min_2():
    content2 = ak.contents.NumpyArray(
        np.array([True, True, True, False, True, False, False, True, False, True])
    )
    offsets3 = ak.index.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)
    depth1 = ak.to_backend(depth1, "cuda", highlevel=False)

    assert to_list(depth1) == [
        [True, True, True],
        [False, True, False],
        [False, True, False, True],
    ]

    assert to_list(ak.min(depth1, -1, highlevel=False)) == [True, False, False]
    assert (
        ak.min(depth1.to_typetracer(), -1, highlevel=False).form
        == ak.min(depth1, -1, highlevel=False).form
    )
    assert to_list(ak.min(depth1, 1, highlevel=False)) == [True, False, False]
    assert (
        ak.min(depth1.to_typetracer(), 1, highlevel=False).form
        == ak.min(depth1, 1, highlevel=False).form
    )

    assert to_list(ak.min(depth1, -2, highlevel=False)) == [False, True, False, True]
    assert (
        ak.min(depth1.to_typetracer(), -2, highlevel=False).form
        == ak.min(depth1, -2, highlevel=False).form
    )
    assert to_list(ak.min(depth1, 0, highlevel=False)) == [False, True, False, True]
    assert (
        ak.min(depth1.to_typetracer(), 0, highlevel=False).form
        == ak.min(depth1, 0, highlevel=False).form
    )
    del depth1


def test_0115_generic_reducer_operation_count_max_1():
    content2 = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 0.0, 2.2, 0.0, 0.0, 2.2, 0.0, 4.4])
    )
    offsets3 = ak.index.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)
    depth1 = ak.to_backend(depth1, "cuda", highlevel=False)

    assert to_list(depth1) == [
        [1.1, 2.2, 3.3],
        [0.0, 2.2, 0.0],
        [0.0, 2.2, 0.0, 4.4],
    ]

    assert to_list(ak.max(depth1, -1, highlevel=False)) == [3.3, 2.2, 4.4]
    assert (
        ak.max(depth1.to_typetracer(), -1, highlevel=False).form
        == ak.max(depth1, -1, highlevel=False).form
    )
    assert to_list(ak.max(depth1, 1, highlevel=False)) == [3.3, 2.2, 4.4]
    assert (
        ak.max(depth1.to_typetracer(), 1, highlevel=False).form
        == ak.max(depth1, 1, highlevel=False).form
    )

    assert to_list(ak.max(depth1, -2, highlevel=False)) == [1.1, 2.2, 3.3, 4.4]
    assert (
        ak.max(depth1.to_typetracer(), -2, highlevel=False).form
        == ak.max(depth1, -2, highlevel=False).form
    )
    assert to_list(ak.max(depth1, 0, highlevel=False)) == [1.1, 2.2, 3.3, 4.4]
    assert (
        ak.max(depth1.to_typetracer(), 0, highlevel=False).form
        == ak.max(depth1, 0, highlevel=False).form
    )
    del depth1


def test_0115_generic_reducer_operation_count_max_2():
    content2 = ak.contents.NumpyArray(
        np.array([False, True, True, False, True, False, False, False, False, False])
    )
    offsets3 = ak.index.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)
    depth1 = ak.to_backend(depth1, "cuda", highlevel=False)

    assert to_list(depth1) == [
        [False, True, True],
        [False, True, False],
        [False, False, False, False],
    ]

    assert to_list(ak.max(depth1, -1, highlevel=False)) == [True, True, False]
    assert (
        ak.max(depth1.to_typetracer(), -1, highlevel=False).form
        == ak.max(depth1, -1, highlevel=False).form
    )
    assert to_list(ak.max(depth1, 1, highlevel=False)) == [True, True, False]
    assert (
        ak.max(depth1.to_typetracer(), 1, highlevel=False).form
        == ak.max(depth1, 1, highlevel=False).form
    )

    assert to_list(ak.max(depth1, -2, highlevel=False)) == [False, True, True, False]
    assert (
        ak.max(depth1.to_typetracer(), -2, highlevel=False).form
        == ak.max(depth1, -2, highlevel=False).form
    )
    assert to_list(ak.max(depth1, 0, highlevel=False)) == [False, True, True, False]
    assert (
        ak.max(depth1.to_typetracer(), 0, highlevel=False).form
        == ak.max(depth1, 0, highlevel=False).form
    )
    del depth1


def test_0115_generic_reducer_operation_mask():
    content = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 6, 6, 9], dtype=np.int64))
    array = ak.contents.ListOffsetArray(offsets, content)
    array = ak.to_backend(array, "cuda", highlevel=False)

    assert to_list(ak.min(array, axis=-1, mask_identity=False, highlevel=False)) == [
        1.1,
        np.inf,
        4.4,
        6.6,
        np.inf,
        np.inf,
        7.7,
    ]
    assert (
        ak.min(
            array.to_typetracer(), axis=-1, mask_identity=False, highlevel=False
        ).form
        == ak.min(array, axis=-1, mask_identity=False, highlevel=False).form
    )
    assert to_list(ak.min(array, axis=-1, mask_identity=True, highlevel=False)) == [
        1.1,
        None,
        4.4,
        6.6,
        None,
        None,
        7.7,
    ]
    assert (
        ak.min(array.to_typetracer(), axis=-1, mask_identity=True, highlevel=False).form
        == ak.min(array, axis=-1, mask_identity=True, highlevel=False).form
    )
    del array


def test_0115_generic_reducer_operation_ByteMaskedArray():
    content = ak.operations.from_iter(
        [
            [[1.1, 0.0, 2.2], [], [3.3, 4.4]],
            [],
            [[5.5]],
            [[6.6, 9.9, 8.8, 7.7]],
            [[], [12.2, 11.1, 10.0]],
        ],
        highlevel=False,
    )
    mask = ak.index.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    v2_array = ak.contents.ByteMaskedArray(mask, content, valid_when=False)
    v2_array = ak.to_backend(v2_array, "cuda", highlevel=False)

    assert to_list(v2_array) == [
        [[1.1, 0.0, 2.2], [], [3.3, 4.4]],
        [],
        None,
        None,
        [[], [12.2, 11.1, 10.0]],
    ]
    assert to_list(ak.argmin(v2_array, axis=-1, highlevel=False)) == [
        [1, None, 0],
        [],
        None,
        None,
        [None, 2],
    ]
    assert (
        ak.argmin(v2_array.to_typetracer(), axis=-1, highlevel=False).form
        == ak.argmin(v2_array, axis=-1, highlevel=False).form
    )
    del v2_array


def test_0115_generic_reducer_operation_keepdims():
    nparray = np.array(primes[: 2 * 3 * 5], dtype=np.int64).reshape(2, 3, 5)
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

    assert to_list(
        ak.prod(depth2, axis=-1, keepdims=False, highlevel=False)
    ) == to_list(ak.prod(nparray, axis=-1, keepdims=False, highlevel=False))
    assert (
        ak.prod(depth2.to_typetracer(), axis=-1, keepdims=False, highlevel=False).form
        == ak.prod(depth2, axis=-1, keepdims=False, highlevel=False).form
    )
    assert to_list(
        ak.prod(depth2, axis=-2, keepdims=False, highlevel=False)
    ) == to_list(ak.prod(nparray, axis=-2, keepdims=False, highlevel=False))
    assert (
        ak.prod(depth2.to_typetracer(), axis=-2, keepdims=False, highlevel=False).form
        == ak.prod(depth2, axis=-2, keepdims=False, highlevel=False).form
    )
    assert to_list(
        ak.prod(depth2, axis=-3, keepdims=False, highlevel=False)
    ) == to_list(ak.prod(nparray, axis=-3, keepdims=False, highlevel=False))
    assert (
        ak.prod(depth2.to_typetracer(), axis=-3, keepdims=False, highlevel=False).form
        == ak.prod(depth2, axis=-3, keepdims=False, highlevel=False).form
    )

    assert to_list(ak.prod(depth2, axis=-1, keepdims=True, highlevel=False)) == to_list(
        ak.prod(nparray, axis=-1, keepdims=True, highlevel=False)
    )
    assert (
        ak.prod(depth2.to_typetracer(), axis=-1, keepdims=True, highlevel=False).form
        == ak.prod(depth2, axis=-1, keepdims=True, highlevel=False).form
    )
    assert to_list(ak.prod(depth2, axis=-2, keepdims=True, highlevel=False)) == to_list(
        ak.prod(nparray, axis=-2, keepdims=True, highlevel=False)
    )
    assert (
        ak.prod(depth2.to_typetracer(), axis=-2, keepdims=True, highlevel=False).form
        == ak.prod(depth2, axis=-2, keepdims=True, highlevel=False).form
    )
    assert to_list(ak.prod(depth2, axis=-3, keepdims=True, highlevel=False)) == to_list(
        ak.prod(nparray, axis=-3, keepdims=True, highlevel=False)
    )
    assert (
        ak.prod(depth2.to_typetracer(), axis=-3, keepdims=True, highlevel=False).form
        == ak.prod(depth2, axis=-3, keepdims=True, highlevel=False).form
    )
    del depth2


def test_0115_generic_reducer_operation_highlevel_1():
    array = ak.highlevel.Array(
        [[[2, 3, 5], [], [7, 11], [13]], [], [[17, 19], [23]]], check_valid=True
    )
    array = ak.to_backend(array, "cuda", highlevel=False)

    assert ak.operations.count(array) == 9
    assert to_list(ak.operations.count(array, axis=-1)) == [
        [3, 0, 2, 1],
        [],
        [2, 1],
    ]
    assert to_list(ak.operations.count(array, axis=2)) == [
        [3, 0, 2, 1],
        [],
        [2, 1],
    ]
    assert to_list(ak.operations.count(array, axis=-1, keepdims=True)) == [
        [[3], [0], [2], [1]],
        [],
        [[2], [1]],
    ]
    assert to_list(ak.operations.count(array, axis=-2)) == [
        [3, 2, 1],
        [],
        [2, 1],
    ]
    assert to_list(ak.operations.count(array, axis=1)) == [
        [3, 2, 1],
        [],
        [2, 1],
    ]
    assert to_list(ak.operations.count(array, axis=-2, keepdims=True)) == [
        [[3, 2, 1]],
        [[]],
        [[2, 1]],
    ]

    assert ak.operations.count_nonzero(array) == 9
    assert to_list(ak.operations.count_nonzero(array, axis=-1)) == [
        [3, 0, 2, 1],
        [],
        [2, 1],
    ]
    assert to_list(ak.operations.count_nonzero(array, axis=-2)) == [
        [3, 2, 1],
        [],
        [2, 1],
    ]

    assert ak.operations.sum(array) == 2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23
    assert to_list(ak.operations.sum(array, axis=-1)) == [
        [2 + 3 + 5, 0, 7 + 11, 13],
        [],
        [17 + 19, 23],
    ]
    assert to_list(ak.operations.sum(array, axis=-2)) == [
        [2 + 7 + 13, 3 + 11, 5],
        [],
        [17 + 23, 19],
    ]

    assert ak.operations.prod(array) == 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23
    assert to_list(ak.operations.prod(array, axis=-1)) == [
        [2 * 3 * 5, 1, 7 * 11, 13],
        [],
        [17 * 19, 23],
    ]
    assert to_list(ak.operations.prod(array, axis=-2)) == [
        [2 * 7 * 13, 3 * 11, 5],
        [],
        [17 * 23, 19],
    ]

    assert ak.operations.min(array) == 2
    assert to_list(ak.operations.min(array, axis=-1)) == [
        [2, None, 7, 13],
        [],
        [17, 23],
    ]
    assert to_list(ak.operations.min(array, axis=-2)) == [
        [2, 3, 5],
        [],
        [17, 19],
    ]

    assert ak.operations.max(array) == 23
    assert to_list(ak.operations.max(array, axis=-1)) == [
        [5, None, 11, 13],
        [],
        [19, 23],
    ]
    assert to_list(ak.operations.max(array, axis=-2)) == [
        [13, 11, 5],
        [],
        [23, 19],
    ]
    del array


def test_0115_generic_reducer_operation_highlevel_2():
    array = ak.highlevel.Array(
        [
            [[True, False, True], [], [False, False], [True]],
            [],
            [[False, True], [True]],
        ],
        check_valid=True,
    )
    array = ak.to_backend(array, "cuda", highlevel=False)

    assert ak.operations.any(array) is np.bool_(True)
    assert to_list(ak.operations.any(array, axis=-1)) == [
        [True, False, False, True],
        [],
        [True, True],
    ]
    assert to_list(ak.operations.any(array, axis=-2)) == [
        [True, False, True],
        [],
        [True, True],
    ]

    assert ak.operations.all(array) is np.bool_(False)
    assert to_list(ak.operations.all(array, axis=-1)) == [
        [False, True, False, True],
        [],
        [False, True],
    ]
    assert to_list(ak.operations.all(array, axis=-2)) == [
        [False, False, True],
        [],
        [False, True],
    ]
    del array
