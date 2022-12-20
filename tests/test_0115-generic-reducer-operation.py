# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

to_list = ak.operations.to_list

primes = [x for x in range(2, 1000) if all(x % n != 0 for n in range(2, x))]


def test_ListOffsetArray_to_RegularArray():
    content = ak.contents.NumpyArray(np.array(primes[: 2 * 3 * 5], dtype=np.int64))
    offsets1 = ak.index.Index64(np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.int64))
    listoffsetarray = ak.contents.ListOffsetArray(offsets1, content)
    regulararray = listoffsetarray.to_RegularArray()
    assert to_list(listoffsetarray) == to_list(regulararray)


def test_dimension_optiontype():
    content = ak.contents.NumpyArray(np.array(primes[: 2 * 3 * 5], dtype=np.int64))
    offsets1 = ak.index.Index64(np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.int64))
    listoffsetarray = ak.contents.ListOffsetArray(offsets1, content)
    index = ak.index.Index64(np.array([5, -1, 3, 2, -1, 0], dtype=np.int64))
    indexedarray = ak.contents.IndexedOptionArray(index, listoffsetarray)
    depth2 = ak.contents.RegularArray(indexedarray, 3)
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

    content = ak.contents.NumpyArray(np.array(primes[: 2 * 3 * 5], dtype=np.int64))
    offsets1 = ak.index.Index64(np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.int64))
    listoffsetarray = ak.contents.ListOffsetArray(offsets1, content)
    index = ak.index.Index64(np.array([5, 4, 3, 2, 1, 0], dtype=np.int64))
    indexedarray = ak.contents.IndexedArray(index, listoffsetarray)
    depth2 = ak.contents.RegularArray(indexedarray, 3)
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


def test_reproduce_numpy():
    content1 = ak.contents.NumpyArray(np.array(primes[: 2 * 3 * 5], dtype=np.int64))
    offsets1 = ak.index.Index64(np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.int64))
    offsets2 = ak.index.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(
        offsets2, ak.contents.ListOffsetArray(offsets1, content1)
    )
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

    content2 = ak.contents.NumpyArray(np.array(primes[:12], dtype=np.int64))
    offsets3 = ak.index.Index64(np.array([0, 4, 8, 12], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)
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


def test_gaps():
    content1 = ak.contents.NumpyArray(
        np.array([123] + primes[: 2 * 3 * 5], dtype=np.int64)
    )
    offsets1 = ak.index.Index64(np.array([0, 1, 6, 11, 16, 21, 26, 31], dtype=np.int64))
    offsets2 = ak.index.Index64(np.array([1, 4, 7], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(
        offsets2, ak.contents.ListOffsetArray(offsets1, content1)
    )

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

    content1 = ak.contents.NumpyArray(np.array(primes[: 2 * 3 * 5 - 1], dtype=np.int64))
    offsets1 = ak.index.Index64(np.array([0, 5, 10, 15, 20, 25, 29], dtype=np.int64))
    offsets2 = ak.index.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(
        offsets2, ak.contents.ListOffsetArray(offsets1, content1)
    )

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

    content1 = ak.contents.NumpyArray(np.array(primes[: 2 * 3 * 5 - 2], dtype=np.int64))
    offsets1 = ak.index.Index64(np.array([0, 5, 10, 15, 20, 25, 28], dtype=np.int64))
    offsets2 = ak.index.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(
        offsets2, ak.contents.ListOffsetArray(offsets1, content1)
    )

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

    content1 = ak.contents.NumpyArray(np.array(primes[1 : 2 * 3 * 5], dtype=np.int64))
    offsets1 = ak.index.Index64(np.array([0, 4, 9, 14, 19, 24, 29], dtype=np.int64))
    offsets2 = ak.index.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(
        offsets2, ak.contents.ListOffsetArray(offsets1, content1)
    )

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

    content1 = ak.contents.NumpyArray(np.array(primes[2 : 2 * 3 * 5], dtype=np.int64))
    offsets1 = ak.index.Index64(np.array([0, 3, 8, 13, 18, 23, 28], dtype=np.int64))
    offsets2 = ak.index.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(
        offsets2, ak.contents.ListOffsetArray(offsets1, content1)
    )

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

    content1 = ak.contents.NumpyArray(np.array(primes[:9], dtype=np.int64))
    offsets1 = ak.index.Index64(np.array([0, 3, 4, 6, 6, 7, 9], dtype=np.int64))
    offsets2 = ak.index.Index64(np.array([0, 2, 4, 6], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(
        offsets2, ak.contents.ListOffsetArray(offsets1, content1)
    )

    assert to_list(depth2) == [[[2, 3, 5], [7]], [[11, 13], []], [[17], [19, 23]]]

    assert to_list(ak.prod(depth2, -3, highlevel=False)) == [
        [2 * 11 * 17, 3 * 13, 5],
        [7 * 19, 23],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), -3, highlevel=False).form
        == ak.prod(depth2, -3, highlevel=False).form
    )

    content1 = ak.contents.NumpyArray(np.array(primes[:9], dtype=np.int64))
    offsets1 = ak.index.Index64(np.array([0, 3, 4, 6, 7, 9], dtype=np.int64))
    offsets2 = ak.index.Index64(np.array([0, 2, 3, 5], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(
        offsets2, ak.contents.ListOffsetArray(offsets1, content1)
    )

    assert to_list(depth2) == [[[2, 3, 5], [7]], [[11, 13]], [[17], [19, 23]]]

    assert to_list(ak.prod(depth2, -3, highlevel=False)) == [
        [2 * 11 * 17, 3 * 13, 5],
        [7 * 19, 23],
    ]
    assert (
        ak.prod(depth2.to_typetracer(), -3, highlevel=False).form
        == ak.prod(depth2, -3, highlevel=False).form
    )

    content1 = ak.contents.NumpyArray(np.array(primes[:10], dtype=np.int64))
    offsets1 = ak.index.Index64(np.array([0, 3, 5, 6, 8, 9, 10], dtype=np.int64))
    offsets2 = ak.index.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(
        offsets2, ak.contents.ListOffsetArray(offsets1, content1)
    )

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

    content1 = ak.contents.NumpyArray(np.array(primes[:9], dtype=np.int64))
    offsets1 = ak.index.Index64(np.array([0, 3, 3, 5, 6, 8, 9], dtype=np.int64))
    offsets2 = ak.index.Index64(np.array([0, 4, 6], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(
        offsets2, ak.contents.ListOffsetArray(offsets1, content1)
    )

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

    content1 = ak.contents.NumpyArray(np.array(primes[:9], dtype=np.int64))
    offsets1 = ak.index.Index64(np.array([0, 3, 3, 5, 6, 8, 9], dtype=np.int64))
    offsets2 = ak.index.Index64(np.array([0, 4, 4, 6], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(
        offsets2, ak.contents.ListOffsetArray(offsets1, content1)
    )

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

    content1 = ak.contents.NumpyArray(np.array(primes[: 2 * 3 * 5], dtype=np.int64))
    offsets1 = ak.index.Index64(np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.int64))
    offsets2 = ak.index.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(
        offsets2, ak.contents.ListOffsetArray(offsets1, content1)
    )

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

    content1 = ak.contents.NumpyArray(np.array(primes[:9], dtype=np.int64))
    offsets1 = ak.index.Index64(np.array([0, 3, 3, 5, 6, 8, 9], dtype=np.int64))
    offsets2 = ak.index.Index64(np.array([0, 4, 4, 6], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(
        offsets2, ak.contents.ListOffsetArray(offsets1, content1)
    )

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


def test_complicated():
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
        ak.prod(complicated.to_typetracer(), -1, highlevel=False).form

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
        ak.prod(complicated.to_typetracer(), -2, highlevel=False).form
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


def test_EmptyArray():
    offsets = ak.index.Index64(np.array([0, 0, 0, 0], dtype=np.int64))
    array = ak.contents.ListOffsetArray(offsets, ak.contents.EmptyArray())

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

    assert to_list(array) == [[], [], []]

    assert to_list(ak.prod(array, -1, highlevel=False)) == [1, 1, 1]
    assert (
        ak.prod(array.to_typetracer(), -1, highlevel=False).form
        == ak.prod(array, -1, highlevel=False).form
    )


def test_IndexedOptionArray():
    content = ak.contents.NumpyArray(np.array(primes[: 2 * 3 * 5], dtype=np.int64))
    offsets1 = ak.index.Index64(np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.int64))
    listoffsetarray = ak.contents.ListOffsetArray(offsets1, content)
    index = ak.index.Index64(np.array([5, 4, 3, 2, 1, 0], dtype=np.int64))
    indexedarray = ak.contents.IndexedArray(index, listoffsetarray)
    offsets2 = ak.index.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(offsets2, indexedarray)

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


@pytest.mark.skip(
    reason="I can't think of a canonical UnionArray (non-mergeable contents) that can be used in a reducer"
)
def test_UnionArray():
    content1 = ak.operations.from_iter(
        [[[2, 3, 5, 7, 11], [13, 17, 19, 23, 29], [31, 37, 41, 43, 47]]],
        highlevel=False,
    )
    content2 = ak.operations.from_iter(
        [[[53, 59, 61, 67, 71], [73, 79, 83, 89, 97], [101, 103, 107, 109, 113]]],
        highlevel=False,
    )

    tags = ak.index.Index8(np.array([0, 1], dtype=np.int8))
    index = ak.index.Index64(np.array([0, 0], dtype=np.int64))
    depth2 = ak.contents.UnionArray(tags, index, [content1, content2])

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

    content1 = ak.contents.NumpyArray(np.array(primes[: 2 * 3 * 5], dtype=np.int64))
    offsets1a = ak.index.Index64(np.array([0, 5, 10, 15], dtype=np.int64))
    offsets1b = ak.index.Index64(np.array([15, 20, 25, 30], dtype=np.int64))
    tags = ak.index.Index8(np.array([0, 0, 0, 1, 1, 1], dtype=np.int8))
    index = ak.index.Index64(np.array([0, 1, 2, 0, 1, 2], dtype=np.int64))
    unionarray = ak.contents.UnionArray(
        tags,
        index,
        [
            ak.contents.ListOffsetArray(offsets1a, content1),
            ak.contents.ListOffsetArray(offsets1b, content1),
        ],
    )
    offsets2 = ak.index.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(offsets2, unionarray)

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


def test_sum():
    content2 = ak.contents.NumpyArray(
        np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], dtype=np.int64)
    )
    offsets3 = ak.index.Index64(np.array([0, 4, 8, 12], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)

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


def test_sumprod_types_FIXME():
    array = np.array([[True, False, False], [True, False, False]])
    content2 = ak.contents.NumpyArray(array.reshape(-1))
    offsets3 = ak.index.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)
    assert (
        np.sum(array, axis=-1).dtype
        == np.asarray(ak.sum(depth1, axis=-1, highlevel=False)).dtype
    )
    assert (
        np.prod(array, axis=-1).dtype
        == np.asarray(ak.prod(depth1, axis=-1, highlevel=False)).dtype
    )


def test_sumprod_types():
    def prod(xs):
        out = 1
        for x in xs:
            out *= x
        return out

    array = np.array([[True, False, False], [True, False, False]])
    content2 = ak.contents.NumpyArray(array.reshape(-1))
    offsets3 = ak.index.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)

    assert sum(to_list(np.sum(array, axis=-1))) == sum(
        to_list(ak.sum(depth1, axis=-1, highlevel=False))
    )
    assert prod(to_list(np.prod(array, axis=-1))) == prod(
        to_list(ak.prod(depth1, axis=-1, highlevel=False))
    )

    array = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int8)
    content2 = ak.contents.NumpyArray(array.reshape(-1))
    offsets3 = ak.index.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)

    assert (
        np.sum(array, axis=-1).dtype
        == np.asarray(ak.sum(depth1, axis=-1, highlevel=False)).dtype
    )
    assert (
        np.prod(array, axis=-1).dtype
        == np.asarray(ak.prod(depth1, axis=-1, highlevel=False)).dtype
    )
    assert sum(to_list(np.sum(array, axis=-1))) == sum(
        to_list(ak.sum(depth1, axis=-1, highlevel=False))
    )
    assert prod(to_list(np.prod(array, axis=-1))) == prod(
        to_list(ak.prod(depth1, axis=-1, highlevel=False))
    )

    array = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
    content2 = ak.contents.NumpyArray(array.reshape(-1))
    offsets3 = ak.index.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)

    assert (
        np.sum(array, axis=-1).dtype
        == np.asarray(ak.sum(depth1, axis=-1, highlevel=False)).dtype
    )
    assert (
        np.prod(array, axis=-1).dtype
        == np.asarray(ak.prod(depth1, axis=-1, highlevel=False)).dtype
    )
    assert sum(to_list(np.sum(array, axis=-1))) == sum(
        to_list(ak.sum(depth1, axis=-1, highlevel=False))
    )
    assert prod(to_list(np.prod(array, axis=-1))) == prod(
        to_list(ak.prod(depth1, axis=-1, highlevel=False))
    )

    array = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int16)
    content2 = ak.contents.NumpyArray(array.reshape(-1))
    offsets3 = ak.index.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)

    assert (
        np.sum(array, axis=-1).dtype
        == np.asarray(ak.sum(depth1, axis=-1, highlevel=False)).dtype
    )
    assert (
        np.prod(array, axis=-1).dtype
        == np.asarray(ak.prod(depth1, axis=-1, highlevel=False)).dtype
    )
    assert sum(to_list(np.sum(array, axis=-1))) == sum(
        to_list(ak.sum(depth1, axis=-1, highlevel=False))
    )
    assert prod(to_list(np.prod(array, axis=-1))) == prod(
        to_list(ak.prod(depth1, axis=-1, highlevel=False))
    )

    array = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint16)
    content2 = ak.contents.NumpyArray(array.reshape(-1))
    offsets3 = ak.index.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)

    assert (
        np.sum(array, axis=-1).dtype
        == np.asarray(ak.sum(depth1, axis=-1, highlevel=False)).dtype
    )
    assert (
        np.prod(array, axis=-1).dtype
        == np.asarray(ak.prod(depth1, axis=-1, highlevel=False)).dtype
    )
    assert sum(to_list(np.sum(array, axis=-1))) == sum(
        to_list(ak.sum(depth1, axis=-1, highlevel=False))
    )
    assert prod(to_list(np.prod(array, axis=-1))) == prod(
        to_list(ak.prod(depth1, axis=-1, highlevel=False))
    )

    array = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    content2 = ak.contents.NumpyArray(array.reshape(-1))
    offsets3 = ak.index.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)

    assert (
        np.sum(array, axis=-1).dtype
        == np.asarray(ak.sum(depth1, axis=-1, highlevel=False)).dtype
    )
    assert (
        np.prod(array, axis=-1).dtype
        == np.asarray(ak.prod(depth1, axis=-1, highlevel=False)).dtype
    )
    assert sum(to_list(np.sum(array, axis=-1))) == sum(
        to_list(ak.sum(depth1, axis=-1, highlevel=False))
    )
    assert prod(to_list(np.prod(array, axis=-1))) == prod(
        to_list(ak.prod(depth1, axis=-1, highlevel=False))
    )

    array = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)
    content2 = ak.contents.NumpyArray(array.reshape(-1))
    offsets3 = ak.index.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)

    assert (
        np.sum(array, axis=-1).dtype
        == np.asarray(ak.sum(depth1, axis=-1, highlevel=False)).dtype
    )
    assert (
        np.prod(array, axis=-1).dtype
        == np.asarray(ak.prod(depth1, axis=-1, highlevel=False)).dtype
    )
    assert sum(to_list(np.sum(array, axis=-1))) == sum(
        to_list(ak.sum(depth1, axis=-1, highlevel=False))
    )
    assert prod(to_list(np.prod(array, axis=-1))) == prod(
        to_list(ak.prod(depth1, axis=-1, highlevel=False))
    )

    array = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    content2 = ak.contents.NumpyArray(array.reshape(-1))
    offsets3 = ak.index.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)

    assert (
        np.sum(array, axis=-1).dtype
        == np.asarray(ak.sum(depth1, axis=-1, highlevel=False)).dtype
    )
    assert (
        np.prod(array, axis=-1).dtype
        == np.asarray(ak.prod(depth1, axis=-1, highlevel=False)).dtype
    )
    assert sum(to_list(np.sum(array, axis=-1))) == sum(
        to_list(ak.sum(depth1, axis=-1, highlevel=False))
    )
    assert prod(to_list(np.prod(array, axis=-1))) == prod(
        to_list(ak.prod(depth1, axis=-1, highlevel=False))
    )

    array = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint64)
    content2 = ak.contents.NumpyArray(array.reshape(-1))
    offsets3 = ak.index.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)

    assert (
        np.sum(array, axis=-1).dtype
        == np.asarray(ak.sum(depth1, axis=-1, highlevel=False)).dtype
    )
    assert (
        np.prod(array, axis=-1).dtype
        == np.asarray(ak.prod(depth1, axis=-1, highlevel=False)).dtype
    )
    assert sum(to_list(np.sum(array, axis=-1))) == sum(
        to_list(ak.sum(depth1, axis=-1, highlevel=False))
    )
    assert prod(to_list(np.prod(array, axis=-1))) == prod(
        to_list(ak.prod(depth1, axis=-1, highlevel=False))
    )


def test_any():
    content2 = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 0.0, 2.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    offsets3 = ak.index.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)

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


def test_all():
    content2 = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 0.0, 2.2, 0.0, 0.0, 2.2, 0.0, 4.4])
    )
    offsets3 = ak.index.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)

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


def test_count():
    content2 = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 0.0, 2.2, 0.0, 0.0, 2.2, 0.0, 4.4])
    )
    offsets3 = ak.index.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)

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


def test_count_nonzero():
    content2 = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 0.0, 2.2, 0.0, 0.0, 2.2, 0.0, 4.4])
    )
    offsets3 = ak.index.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)

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


def test_count_min():
    content2 = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 0.0, 2.2, 0.0, 0.0, 2.2, 0.0, 4.4])
    )
    offsets3 = ak.index.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)

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

    content2 = ak.contents.NumpyArray(
        np.array([True, True, True, False, True, False, False, True, False, True])
    )
    offsets3 = ak.index.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)

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


def test_count_max():
    content2 = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 0.0, 2.2, 0.0, 0.0, 2.2, 0.0, 4.4])
    )
    offsets3 = ak.index.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)

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

    content2 = ak.contents.NumpyArray(
        np.array([False, True, True, False, True, False, False, False, False, False])
    )
    offsets3 = ak.index.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)

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


def test_mask():
    content = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 6, 6, 9], dtype=np.int64))
    array = ak.contents.ListOffsetArray(offsets, content)

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


def test_ByteMaskedArray():
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


def test_keepdims():
    nparray = np.array(primes[: 2 * 3 * 5], dtype=np.int64).reshape(2, 3, 5)
    content1 = ak.contents.NumpyArray(np.array(primes[: 2 * 3 * 5], dtype=np.int64))
    offsets1 = ak.index.Index64(np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.int64))
    offsets2 = ak.index.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.contents.ListOffsetArray(
        offsets2, ak.contents.ListOffsetArray(offsets1, content1)
    )

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


def test_highlevel():
    array = ak.highlevel.Array(
        [[[2, 3, 5], [], [7, 11], [13]], [], [[17, 19], [23]]], check_valid=True
    )

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

    array = ak.highlevel.Array(
        [
            [[True, False, True], [], [False, False], [True]],
            [],
            [[False, True], [True]],
        ],
        check_valid=True,
    )

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


def test_nonreducers():
    x = ak.highlevel.Array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], check_valid=True)
    y = ak.highlevel.Array(
        [[1.1, 2.2, 2.9, 4.0, 5.1], [0.9, 2.1, 3.2, 4.1, 4.9]], check_valid=True
    )

    assert ak.operations.mean(y) == np.mean(ak.operations.to_numpy(y))
    assert ak.operations.var(y) == np.var(ak.operations.to_numpy(y))
    assert ak.operations.var(y, ddof=1) == np.var(ak.operations.to_numpy(y), ddof=1)
    assert ak.operations.std(y) == np.std(ak.operations.to_numpy(y))
    assert ak.operations.std(y, ddof=1) == np.std(ak.operations.to_numpy(y), ddof=1)

    assert ak.operations.moment(y, 1) == np.mean(ak.operations.to_numpy(y))
    assert ak.operations.moment(y - ak.operations.mean(y), 2) == np.var(
        ak.operations.to_numpy(y)
    )
    assert ak.operations.covar(y, y) == np.var(ak.operations.to_numpy(y))
    assert ak.operations.corr(y, y) == 1.0

    assert ak.operations.corr(x, y) == pytest.approx(0.9968772535047296)
    fit = ak.operations.linear_fit(x, y)
    assert to_list(fit) == pytest.approx(
        {
            "intercept": 0.07999999999999773,
            "slope": 0.99,
            "intercept_error": 0.7416198487095663,
            "slope_error": 0.22360679774997896,
        }
    )

    assert to_list(ak.operations.mean(y, axis=-1)) == pytest.approx(
        to_list(np.mean(ak.operations.to_numpy(y), axis=-1))
    )
    assert to_list(ak.operations.var(y, axis=-1)) == pytest.approx(
        to_list(np.var(ak.operations.to_numpy(y), axis=-1))
    )
    assert to_list(ak.operations.var(y, axis=-1, ddof=1)) == pytest.approx(
        to_list(np.var(ak.operations.to_numpy(y), axis=-1, ddof=1))
    )
    assert to_list(ak.operations.std(y, axis=-1)) == pytest.approx(
        to_list(np.std(ak.operations.to_numpy(y), axis=-1))
    )
    assert to_list(ak.operations.std(y, axis=-1, ddof=1)) == pytest.approx(
        to_list(np.std(ak.operations.to_numpy(y), axis=-1, ddof=1))
    )

    assert to_list(ak.operations.moment(y, 1, axis=-1)) == pytest.approx(
        to_list(np.mean(ak.operations.to_numpy(y), axis=-1))
    )
    assert to_list(
        ak.operations.moment(y - ak.operations.mean(y, axis=-1), 2, axis=-1)
    ) == pytest.approx(to_list(np.var(ak.operations.to_numpy(y), axis=-1)))
    assert to_list(ak.operations.covar(y, y, axis=-1)) == pytest.approx(
        to_list(np.var(ak.operations.to_numpy(y), axis=-1))
    )
    assert to_list(ak.operations.corr(y, y, axis=-1)) == pytest.approx([1.0, 1.0])

    assert to_list(ak.operations.corr(x, y, axis=-1)) == pytest.approx(
        [0.9975103695813371, 0.9964193240901015]
    )
    fit = ak.operations.linear_fit(x, y, axis=-1)
    assert to_list(fit[0]) == pytest.approx(
        {
            "intercept": 0.11999999999999772,
            "slope": 0.9800000000000005,
            "intercept_error": 1.0488088481701516,
            "slope_error": 0.31622776601683794,
        }
    )
    assert to_list(fit[1]) == pytest.approx(
        {
            "intercept": 0.04000000000000228,
            "slope": 0.9999999999999994,
            "intercept_error": 1.0488088481701516,
            "slope_error": 0.31622776601683794,
        }
    )


def test_softmax():
    array = ak.highlevel.Array(
        [[np.log(2), np.log(2), np.log(4)], [], [np.log(5), np.log(5)]],
        check_valid=True,
    )
    assert to_list(ak.operations.softmax(array, axis=-1)) == [
        pytest.approx([0.25, 0.25, 0.5]),
        [],
        pytest.approx([0.5, 0.5]),
    ]


def test_prod_bool():
    # this had been silently broken
    array = np.array([[True, False, False], [True, False, False]])
    content2 = ak.contents.NumpyArray(array.reshape(-1))
    offsets3 = ak.index.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)
    assert to_list(ak.prod(depth1, axis=-1, highlevel=False)) == [0, 1, 0, 0]
    assert to_list(ak.all(depth1, axis=-1, highlevel=False)) == [
        False,
        True,
        False,
        False,
    ]
    assert to_list(ak.min(depth1, axis=-1, highlevel=False)) == [
        False,
        None,
        False,
        False,
    ]

    array = np.array([[True, False, False], [True, False, False]]).view(np.uint8)
    content2 = ak.contents.NumpyArray(array.reshape(-1))
    offsets3 = ak.index.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)
    assert to_list(ak.prod(depth1, axis=-1, highlevel=False)) == [0, 1, 0, 0]
    assert to_list(ak.all(depth1, axis=-1, highlevel=False)) == [0, 1, 0, 0]
    assert to_list(ak.min(depth1, axis=-1, highlevel=False)) == [0, None, 0, 0]

    array = np.array([[True, False, False], [True, False, False]]).astype(np.int32)
    content2 = ak.contents.NumpyArray(array.reshape(-1))
    offsets3 = ak.index.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)
    assert to_list(ak.prod(depth1, axis=-1, highlevel=False)) == [0, 1, 0, 0]
    assert to_list(ak.all(depth1, axis=-1, highlevel=False)) == [0, 1, 0, 0]
    assert to_list(ak.min(depth1, axis=-1, highlevel=False)) == [0, None, 0, 0]
