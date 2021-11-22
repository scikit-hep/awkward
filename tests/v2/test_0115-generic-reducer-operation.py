# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)

primes = [x for x in range(2, 1000) if all(x % n != 0 for n in range(2, x))]


def test_ListOffsetArray_to_RegularArray():
    content = ak.layout.NumpyArray(np.array(primes[: 2 * 3 * 5], dtype=np.int64))
    offsets1 = ak.layout.Index64(np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.int64))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets1, content)
    listoffsetarray = v1_to_v2(listoffsetarray)
    regulararray = listoffsetarray.toRegularArray()
    assert ak.to_list(listoffsetarray) == ak.to_list(regulararray)


def test_dimension_optiontype():
    content = ak.layout.NumpyArray(np.array(primes[: 2 * 3 * 5], dtype=np.int64))
    offsets1 = ak.layout.Index64(np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.int64))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets1, content)
    index = ak.layout.Index64(np.array([5, -1, 3, 2, -1, 0], dtype=np.int64))
    indexedarray = ak.layout.IndexedOptionArray64(index, listoffsetarray)
    depth2 = ak.layout.RegularArray(indexedarray, 3)
    depth2 = v1_to_v2(depth2)
    assert ak.to_list(depth2) == [
        [[101, 103, 107, 109, 113], None, [53, 59, 61, 67, 71]],
        [[31, 37, 41, 43, 47], None, [2, 3, 5, 7, 11]],
    ]
    assert ak.to_list(depth2.prod(axis=-1, keepdims=False)) == [
        [101 * 103 * 107 * 109 * 113, None, 53 * 59 * 61 * 67 * 71],
        [31 * 37 * 41 * 43 * 47, None, 2 * 3 * 5 * 7 * 11],
    ]
    assert ak.to_list(depth2.prod(axis=-1, keepdims=True)) == [
        [[101 * 103 * 107 * 109 * 113], None, [53 * 59 * 61 * 67 * 71]],
        [[31 * 37 * 41 * 43 * 47], None, [2 * 3 * 5 * 7 * 11]],
    ]

    content = ak.layout.NumpyArray(np.array(primes[: 2 * 3 * 5], dtype=np.int64))
    offsets1 = ak.layout.Index64(np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.int64))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets1, content)
    index = ak.layout.Index64(np.array([5, 4, 3, 2, 1, 0], dtype=np.int64))
    indexedarray = ak.layout.IndexedArray64(index, listoffsetarray)
    depth2 = ak.layout.RegularArray(indexedarray, 3)
    depth2 = v1_to_v2(depth2)
    assert ak.to_list(depth2) == [
        [[101, 103, 107, 109, 113], [73, 79, 83, 89, 97], [53, 59, 61, 67, 71]],
        [[31, 37, 41, 43, 47], [13, 17, 19, 23, 29], [2, 3, 5, 7, 11]],
    ]
    assert ak.to_list(depth2.prod(axis=-1)) == [
        [101 * 103 * 107 * 109 * 113, 73 * 79 * 83 * 89 * 97, 53 * 59 * 61 * 67 * 71],
        [31 * 37 * 41 * 43 * 47, 13 * 17 * 19 * 23 * 29, 2 * 3 * 5 * 7 * 11],
    ]
    assert ak.to_list(depth2.prod(axis=-1, keepdims=True)) == [
        [
            [101 * 103 * 107 * 109 * 113],
            [73 * 79 * 83 * 89 * 97],
            [53 * 59 * 61 * 67 * 71],
        ],
        [[31 * 37 * 41 * 43 * 47], [13 * 17 * 19 * 23 * 29], [2 * 3 * 5 * 7 * 11]],
    ]


def test_reproduce_numpy():
    content1 = ak.layout.NumpyArray(np.array(primes[: 2 * 3 * 5], dtype=np.int64))
    offsets1 = ak.layout.Index64(np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.int64))
    offsets2 = ak.layout.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.layout.ListOffsetArray64(
        offsets2, ak.layout.ListOffsetArray64(offsets1, content1)
    )
    depth2 = v1_to_v2(depth2)
    assert ak.to_list(depth2) == [
        [[2, 3, 5, 7, 11], [13, 17, 19, 23, 29], [31, 37, 41, 43, 47]],
        [[53, 59, 61, 67, 71], [73, 79, 83, 89, 97], [101, 103, 107, 109, 113]],
    ]

    assert ak.to_list(depth2.prod(axis=-1)) == [
        [2 * 3 * 5 * 7 * 11, 13 * 17 * 19 * 23 * 29, 31 * 37 * 41 * 43 * 47],
        [53 * 59 * 61 * 67 * 71, 73 * 79 * 83 * 89 * 97, 101 * 103 * 107 * 109 * 113],
    ]
    assert depth2.typetracer.prod(axis=-1).form == depth2.prod(axis=-1).form
    assert ak.to_list(depth2.prod(axis=2)) == [
        [2 * 3 * 5 * 7 * 11, 13 * 17 * 19 * 23 * 29, 31 * 37 * 41 * 43 * 47],
        [53 * 59 * 61 * 67 * 71, 73 * 79 * 83 * 89 * 97, 101 * 103 * 107 * 109 * 113],
    ]
    assert depth2.typetracer.prod(axis=2).form == depth2.prod(axis=2).form

    assert ak.to_list(depth2.prod(axis=-2)) == [
        [2 * 13 * 31, 3 * 17 * 37, 5 * 19 * 41, 7 * 23 * 43, 11 * 29 * 47],
        [53 * 73 * 101, 59 * 79 * 103, 61 * 83 * 107, 67 * 89 * 109, 71 * 97 * 113],
    ]
    assert depth2.typetracer.prod(axis=-2).form == depth2.prod(axis=-2).form
    assert ak.to_list(depth2.prod(axis=1)) == [
        [2 * 13 * 31, 3 * 17 * 37, 5 * 19 * 41, 7 * 23 * 43, 11 * 29 * 47],
        [53 * 73 * 101, 59 * 79 * 103, 61 * 83 * 107, 67 * 89 * 109, 71 * 97 * 113],
    ]
    assert depth2.typetracer.prod(axis=1).form == depth2.prod(axis=1).form

    assert ak.to_list(depth2.prod(axis=-3)) == [
        [2 * 53, 3 * 59, 5 * 61, 7 * 67, 11 * 71],
        [13 * 73, 17 * 79, 19 * 83, 23 * 89, 29 * 97],
        [31 * 101, 37 * 103, 41 * 107, 43 * 109, 47 * 113],
    ]
    assert depth2.typetracer.prod(axis=-3).form == depth2.prod(axis=-3).form
    assert ak.to_list(depth2.prod(axis=0)) == [
        [2 * 53, 3 * 59, 5 * 61, 7 * 67, 11 * 71],
        [13 * 73, 17 * 79, 19 * 83, 23 * 89, 29 * 97],
        [31 * 101, 37 * 103, 41 * 107, 43 * 109, 47 * 113],
    ]
    assert depth2.typetracer.prod(axis=0).form == depth2.prod(axis=0).form

    content2 = ak.layout.NumpyArray(np.array(primes[:12], dtype=np.int64))
    offsets3 = ak.layout.Index64(np.array([0, 4, 8, 12], dtype=np.int64))
    depth1 = ak.layout.ListOffsetArray64(offsets3, content2)
    depth1 = v1_to_v2(depth1)
    assert ak.to_list(depth1.prod(-1)) == [
        2 * 3 * 5 * 7,
        11 * 13 * 17 * 19,
        23 * 29 * 31 * 37,
    ]
    assert depth1.typetracer.prod(-1).form == depth1.prod(-1).form
    assert ak.to_list(depth1.prod(1)) == [
        2 * 3 * 5 * 7,
        11 * 13 * 17 * 19,
        23 * 29 * 31 * 37,
    ]
    assert depth1.typetracer.prod(1).form == depth1.prod(1).form

    assert ak.to_list(depth1.prod(-2)) == [
        2 * 11 * 23,
        3 * 13 * 29,
        5 * 17 * 31,
        7 * 19 * 37,
    ]
    assert depth1.typetracer.prod(-2).form == depth1.prod(-2).form
    assert ak.to_list(depth1.prod(0)) == [
        2 * 11 * 23,
        3 * 13 * 29,
        5 * 17 * 31,
        7 * 19 * 37,
    ]
    assert depth1.typetracer.prod(0).form == depth1.prod(0).form


def test_gaps():
    content1 = ak.layout.NumpyArray(
        np.array([123] + primes[: 2 * 3 * 5], dtype=np.int64)
    )
    offsets1 = ak.layout.Index64(
        np.array([0, 1, 6, 11, 16, 21, 26, 31], dtype=np.int64)
    )
    offsets2 = ak.layout.Index64(np.array([1, 4, 7], dtype=np.int64))
    depth2 = ak.layout.ListOffsetArray64(
        offsets2, ak.layout.ListOffsetArray64(offsets1, content1)
    )
    depth2 = v1_to_v2(depth2)

    assert ak.to_list(depth2) == [
        [[2, 3, 5, 7, 11], [13, 17, 19, 23, 29], [31, 37, 41, 43, 47]],
        [[53, 59, 61, 67, 71], [73, 79, 83, 89, 97], [101, 103, 107, 109, 113]],
    ]

    assert ak.to_list(depth2.prod(-3)) == [
        [106, 177, 305, 469, 781],
        [949, 1343, 1577, 2047, 2813],
        [3131, 3811, 4387, 4687, 5311],
    ]
    assert depth2.typetracer.prod(-3).form == depth2.prod(-3).form

    content1 = ak.layout.NumpyArray(np.array(primes[: 2 * 3 * 5 - 1], dtype=np.int64))
    offsets1 = ak.layout.Index64(np.array([0, 5, 10, 15, 20, 25, 29], dtype=np.int64))
    offsets2 = ak.layout.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.layout.ListOffsetArray64(
        offsets2, ak.layout.ListOffsetArray64(offsets1, content1)
    )
    depth2 = v1_to_v2(depth2)

    assert ak.to_list(depth2) == [
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

    assert ak.to_list(depth2.prod(-3)) == [
        [106, 177, 305, 469, 781],
        [949, 1343, 1577, 2047, 2813],
        [3131, 3811, 4387, 4687, 47],
    ]
    assert depth2.typetracer.prod(-3).form == depth2.prod(-3).form

    content1 = ak.layout.NumpyArray(np.array(primes[: 2 * 3 * 5 - 2], dtype=np.int64))
    offsets1 = ak.layout.Index64(np.array([0, 5, 10, 15, 20, 25, 28], dtype=np.int64))
    offsets2 = ak.layout.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.layout.ListOffsetArray64(
        offsets2, ak.layout.ListOffsetArray64(offsets1, content1)
    )
    depth2 = v1_to_v2(depth2)

    assert ak.to_list(depth2) == [
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

    assert ak.to_list(depth2.prod(-3)) == [
        [106, 177, 305, 469, 781],
        [949, 1343, 1577, 2047, 2813],
        [3131, 3811, 4387, 43, 47],
    ]
    assert depth2.typetracer.prod(-3).form == depth2.prod(-3).form

    content1 = ak.layout.NumpyArray(
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
    offsets1 = ak.layout.Index64(np.array([0, 5, 10, 15, 20, 24, 28], dtype=np.int64))
    offsets2 = ak.layout.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.layout.ListOffsetArray64(
        offsets2, ak.layout.ListOffsetArray64(offsets1, content1)
    )
    depth2 = v1_to_v2(depth2)

    assert ak.to_list(depth2) == [
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

    assert ak.to_list(depth2.prod(-3)) == [
        [106, 177, 305, 469, 781],
        [949, 1343, 1577, 2047, 29],
        [3131, 3811, 4387, 4687, 47],
    ]
    assert depth2.typetracer.prod(-3).form == depth2.prod(-3).form

    content1 = ak.layout.NumpyArray(np.array(primes[1 : 2 * 3 * 5], dtype=np.int64))
    offsets1 = ak.layout.Index64(np.array([0, 4, 9, 14, 19, 24, 29], dtype=np.int64))
    offsets2 = ak.layout.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.layout.ListOffsetArray64(
        offsets2, ak.layout.ListOffsetArray64(offsets1, content1)
    )
    depth2 = v1_to_v2(depth2)

    assert ak.to_list(depth2) == [
        [[3, 5, 7, 11], [13, 17, 19, 23, 29], [31, 37, 41, 43, 47]],
        [[53, 59, 61, 67, 71], [73, 79, 83, 89, 97], [101, 103, 107, 109, 113]],
    ]

    assert ak.to_list(depth2.prod(-3)) == [
        [159, 295, 427, 737, 71],
        [949, 1343, 1577, 2047, 2813],
        [3131, 3811, 4387, 4687, 5311],
    ]
    assert depth2.typetracer.prod(-3).form == depth2.prod(-3).form

    content1 = ak.layout.NumpyArray(np.array(primes[2 : 2 * 3 * 5], dtype=np.int64))
    offsets1 = ak.layout.Index64(np.array([0, 3, 8, 13, 18, 23, 28], dtype=np.int64))
    offsets2 = ak.layout.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.layout.ListOffsetArray64(
        offsets2, ak.layout.ListOffsetArray64(offsets1, content1)
    )
    depth2 = v1_to_v2(depth2)

    assert ak.to_list(depth2) == [
        [[5, 7, 11], [13, 17, 19, 23, 29], [31, 37, 41, 43, 47]],
        [[53, 59, 61, 67, 71], [73, 79, 83, 89, 97], [101, 103, 107, 109, 113]],
    ]

    assert ak.to_list(depth2.prod(-3)) == [
        [265, 413, 671, 67, 71],
        [949, 1343, 1577, 2047, 2813],
        [3131, 3811, 4387, 4687, 5311],
    ]
    assert depth2.typetracer.prod(-3).form == depth2.prod(-3).form

    content1 = ak.layout.NumpyArray(
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
    offsets1 = ak.layout.Index64(np.array([0, 3, 8, 13, 18, 23, 28], dtype=np.int64))
    offsets2 = ak.layout.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.layout.ListOffsetArray64(
        offsets2, ak.layout.ListOffsetArray64(offsets1, content1)
    )
    depth2 = v1_to_v2(depth2)

    assert ak.to_list(depth2) == [
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

    assert ak.to_list(depth2.prod(-3)) == [
        [159, 295, 427, 67, 71],
        [949, 1343, 1577, 2047, 2813],
        [3131, 3811, 4387, 4687, 5311],
    ]
    assert depth2.typetracer.prod(-3).form == depth2.prod(-3).form

    content1 = ak.layout.NumpyArray(
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
    offsets1 = ak.layout.Index64(np.array([0, 4, 8, 13, 18, 23, 28], dtype=np.int64))
    offsets2 = ak.layout.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.layout.ListOffsetArray64(
        offsets2, ak.layout.ListOffsetArray64(offsets1, content1)
    )
    depth2 = v1_to_v2(depth2)

    assert ak.to_list(depth2) == [
        [[3, 5, 7, 11], [13, 17, 19, 23], [31, 37, 41, 43, 47]],
        [[53, 59, 61, 67, 71], [73, 79, 83, 89, 97], [101, 103, 107, 109, 113]],
    ]

    assert ak.to_list(depth2.prod(-3)) == [
        [159, 295, 427, 737, 71],
        [949, 1343, 1577, 2047, 97],
        [3131, 3811, 4387, 4687, 5311],
    ]
    assert depth2.typetracer.prod(-3).form == depth2.prod(-3).form

    content1 = ak.layout.NumpyArray(
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
    offsets1 = ak.layout.Index64(np.array([0, 5, 10, 14, 19, 24, 28], dtype=np.int64))
    offsets2 = ak.layout.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.layout.ListOffsetArray64(
        offsets2, ak.layout.ListOffsetArray64(offsets1, content1)
    )
    depth2 = v1_to_v2(depth2)

    assert ak.to_list(depth2) == [
        [[2, 3, 5, 7, 11], [13, 17, 19, 23, 29], [31, 37, 41, 43]],
        [[53, 59, 61, 67, 71], [73, 79, 83, 89, 97], [101, 103, 107, 109]],
    ]

    assert ak.to_list(depth2.prod(-3)) == [
        [106, 177, 305, 469, 781],
        [949, 1343, 1577, 2047, 2813],
        [3131, 3811, 4387, 4687],
    ]
    assert depth2.typetracer.prod(-3).form == depth2.prod(-3).form

    content1 = ak.layout.NumpyArray(
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
    offsets1 = ak.layout.Index64(np.array([0, 5, 9, 14, 19, 23, 28], dtype=np.int64))
    offsets2 = ak.layout.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.layout.ListOffsetArray64(
        offsets2, ak.layout.ListOffsetArray64(offsets1, content1)
    )
    depth2 = v1_to_v2(depth2)

    assert ak.to_list(depth2) == [
        [[2, 3, 5, 7, 11], [13, 17, 19, 23], [31, 37, 41, 43, 47]],
        [[53, 59, 61, 67, 71], [73, 79, 83, 89], [101, 103, 107, 109, 113]],
    ]

    assert ak.to_list(depth2.prod(-3)) == [
        [106, 177, 305, 469, 781],
        [949, 1343, 1577, 2047],
        [3131, 3811, 4387, 4687, 5311],
    ]
    assert depth2.typetracer.prod(-3).form == depth2.prod(-3).form

    content1 = ak.layout.NumpyArray(np.array(primes[:9], dtype=np.int64))
    offsets1 = ak.layout.Index64(np.array([0, 3, 4, 6, 6, 7, 9], dtype=np.int64))
    offsets2 = ak.layout.Index64(np.array([0, 2, 4, 6], dtype=np.int64))
    depth2 = ak.layout.ListOffsetArray64(
        offsets2, ak.layout.ListOffsetArray64(offsets1, content1)
    )
    depth2 = v1_to_v2(depth2)

    assert ak.to_list(depth2) == [[[2, 3, 5], [7]], [[11, 13], []], [[17], [19, 23]]]

    assert ak.to_list(depth2.prod(-3)) == [[2 * 11 * 17, 3 * 13, 5], [7 * 19, 23]]
    assert depth2.typetracer.prod(-3).form == depth2.prod(-3).form

    content1 = ak.layout.NumpyArray(np.array(primes[:9], dtype=np.int64))
    offsets1 = ak.layout.Index64(np.array([0, 3, 4, 6, 7, 9], dtype=np.int64))
    offsets2 = ak.layout.Index64(np.array([0, 2, 3, 5], dtype=np.int64))
    depth2 = ak.layout.ListOffsetArray64(
        offsets2, ak.layout.ListOffsetArray64(offsets1, content1)
    )
    depth2 = v1_to_v2(depth2)

    assert ak.to_list(depth2) == [[[2, 3, 5], [7]], [[11, 13]], [[17], [19, 23]]]

    assert ak.to_list(depth2.prod(-3)) == [[2 * 11 * 17, 3 * 13, 5], [7 * 19, 23]]
    assert depth2.typetracer.prod(-3).form == depth2.prod(-3).form

    content1 = ak.layout.NumpyArray(np.array(primes[:10], dtype=np.int64))
    offsets1 = ak.layout.Index64(np.array([0, 3, 5, 6, 8, 9, 10], dtype=np.int64))
    offsets2 = ak.layout.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.layout.ListOffsetArray64(
        offsets2, ak.layout.ListOffsetArray64(offsets1, content1)
    )
    depth2 = v1_to_v2(depth2)

    assert ak.to_list(depth2) == [[[2, 3, 5], [7, 11], [13]], [[17, 19], [23], [29]]]

    assert ak.to_list(depth2.prod(-3)) == [[34, 57, 5], [161, 11], [377]]
    assert depth2.typetracer.prod(-3).form == depth2.prod(-3).form

    content1 = ak.layout.NumpyArray(np.array(primes[:9], dtype=np.int64))
    offsets1 = ak.layout.Index64(np.array([0, 3, 3, 5, 6, 8, 9], dtype=np.int64))
    offsets2 = ak.layout.Index64(np.array([0, 4, 6], dtype=np.int64))
    depth2 = ak.layout.ListOffsetArray64(
        offsets2, ak.layout.ListOffsetArray64(offsets1, content1)
    )
    depth2 = v1_to_v2(depth2)

    assert ak.to_list(depth2) == [[[2, 3, 5], [], [7, 11], [13]], [[17, 19], [23]]]

    assert ak.to_list(depth2.prod(-3)) == [[34, 57, 5], [23], [7, 11], [13]]
    assert depth2.typetracer.prod(-3).form == depth2.prod(-3).form

    content1 = ak.layout.NumpyArray(np.array(primes[:9], dtype=np.int64))
    offsets1 = ak.layout.Index64(np.array([0, 3, 3, 5, 6, 8, 9], dtype=np.int64))
    offsets2 = ak.layout.Index64(np.array([0, 4, 4, 6], dtype=np.int64))
    depth2 = ak.layout.ListOffsetArray64(
        offsets2, ak.layout.ListOffsetArray64(offsets1, content1)
    )
    depth2 = v1_to_v2(depth2)

    assert ak.to_list(depth2) == [[[2, 3, 5], [], [7, 11], [13]], [], [[17, 19], [23]]]

    assert ak.to_list(depth2.prod(-3)) == [[34, 57, 5], [23], [7, 11], [13]]
    assert depth2.typetracer.prod(-3).form == depth2.prod(-3).form

    content1 = ak.layout.NumpyArray(np.array(primes[: 2 * 3 * 5], dtype=np.int64))
    offsets1 = ak.layout.Index64(np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.int64))
    offsets2 = ak.layout.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.layout.ListOffsetArray64(
        offsets2, ak.layout.ListOffsetArray64(offsets1, content1)
    )
    depth2 = v1_to_v2(depth2)

    assert ak.to_list(depth2) == [
        [[2, 3, 5, 7, 11], [13, 17, 19, 23, 29], [31, 37, 41, 43, 47]],
        [[53, 59, 61, 67, 71], [73, 79, 83, 89, 97], [101, 103, 107, 109, 113]],
    ]

    assert ak.to_list(depth2.prod(-1)) == [
        [2 * 3 * 5 * 7 * 11, 13 * 17 * 19 * 23 * 29, 31 * 37 * 41 * 43 * 47],
        [53 * 59 * 61 * 67 * 71, 73 * 79 * 83 * 89 * 97, 101 * 103 * 107 * 109 * 113],
    ]
    assert depth2.typetracer.prod(-1).form == depth2.prod(-1).form

    assert ak.to_list(depth2.prod(-2)) == [
        [2 * 13 * 31, 3 * 17 * 37, 5 * 19 * 41, 7 * 23 * 43, 11 * 29 * 47],
        [53 * 73 * 101, 59 * 79 * 103, 61 * 83 * 107, 67 * 89 * 109, 71 * 97 * 113],
    ]
    assert depth2.typetracer.prod(-2).form == depth2.prod(-2).form

    content1 = ak.layout.NumpyArray(np.array(primes[:9], dtype=np.int64))
    offsets1 = ak.layout.Index64(np.array([0, 3, 3, 5, 6, 8, 9], dtype=np.int64))
    offsets2 = ak.layout.Index64(np.array([0, 4, 4, 6], dtype=np.int64))
    depth2 = ak.layout.ListOffsetArray64(
        offsets2, ak.layout.ListOffsetArray64(offsets1, content1)
    )
    depth2 = v1_to_v2(depth2)

    assert ak.to_list(depth2) == [
        [[2, 3, 5], [], [7, 11], [13]],
        [],
        [[17, 19], [23]],
    ]

    assert ak.to_list(depth2.prod(-1)) == [
        [2 * 3 * 5, 1, 7 * 11, 13],
        [],
        [17 * 19, 23],
    ]
    assert depth2.typetracer.prod(-1).form == depth2.prod(-1).form

    assert ak.to_list(depth2.prod(-2)) == [[2 * 7 * 13, 3 * 11, 5], [], [17 * 23, 19]]
    assert depth2.typetracer.prod(-2).form == depth2.prod(-2).form

    assert ak.to_list(depth2.prod(-3)) == [[2 * 17, 3 * 19, 5], [23], [7, 11], [13]]
    assert depth2.typetracer.prod(-3).form == depth2.prod(-3).form


def test_complicated():
    offsets1 = ak.layout.Index64(np.array([0, 3, 3, 5], dtype=np.int64))
    content1 = ak.layout.ListOffsetArray64(
        offsets1, ak.layout.NumpyArray(np.array(primes[:5], dtype=np.int64))
    )
    offsets2 = ak.layout.Index64(np.array([0, 3, 3, 5, 6, 8, 9], dtype=np.int64))
    offsets3 = ak.layout.Index64(np.array([0, 4, 4, 6], dtype=np.int64))
    content2 = ak.layout.ListOffsetArray64(
        offsets3,
        ak.layout.ListOffsetArray64(
            offsets2, ak.layout.NumpyArray(np.array(primes[:9], dtype=np.int64))
        ),
    )
    offsets4 = ak.layout.Index64(np.array([0, 1, 1, 3], dtype=np.int64))
    complicated = ak.layout.ListOffsetArray64(
        offsets4, ak.layout.RecordArray([content1, content2], ["x", "y"])
    )
    complicated = v1_to_v2(complicated)

    assert ak.to_list(complicated) == [
        [{"x": [2, 3, 5], "y": [[2, 3, 5], [], [7, 11], [13]]}],
        [],
        [{"x": [], "y": []}, {"x": [7, 11], "y": [[17, 19], [23]]}],
    ]

    assert ak.to_list(complicated["x"]) == [[[2, 3, 5]], [], [[], [7, 11]]]
    assert complicated.typetracer["x"].form == complicated["x"].form
    assert ak.to_list(complicated["y"]) == [
        [[[2, 3, 5], [], [7, 11], [13]]],
        [],
        [[], [[17, 19], [23]]],
    ]
    assert complicated.typetracer["y"].form == complicated["y"].form

    assert ak.to_list(complicated.prod(-1)) == [
        {"x": [30], "y": [[30, 1, 77, 13]]},
        {"x": [], "y": []},
        {"x": [1, 77], "y": [[], [323, 23]]},
    ]
    assert complicated.typetracer.prod(-1).form == complicated.prod(-1).form
    assert ak.to_list(complicated["x"].prod(-1)) == [[30], [], [1, 77]]
    assert complicated.typetracer["x"].prod(-1).form == complicated["x"].prod(-1).form
    assert ak.to_list(complicated["y"].prod(-1)) == [
        [[30, 1, 77, 13]],
        [],
        [[], [323, 23]],
    ]
    assert complicated.typetracer["y"].prod(-1).form == complicated["y"].prod(-1).form

    assert ak.to_list(complicated.prod(-2)) == [
        {"x": [2, 3, 5], "y": [[182, 33, 5]]},
        {"x": [], "y": []},
        {"x": [7, 11], "y": [[], [391, 19]]},
    ]
    assert complicated.typetracer.prod(-2).form == complicated.prod(-2).form
    assert ak.to_list(complicated["x"].prod(-2)) == [[2, 3, 5], [], [7, 11]]
    assert complicated.typetracer["x"].prod(-2).form == complicated["x"].prod(-2).form
    assert ak.to_list(complicated["y"].prod(-2)) == [
        [[182, 33, 5]],
        [],
        [[], [391, 19]],
    ]
    assert complicated.typetracer["y"].prod(-2).form == complicated["y"].prod(-2).form

    assert ak.to_list(complicated[0]) == [
        {"x": [2, 3, 5], "y": [[2, 3, 5], [], [7, 11], [13]]}
    ]
    assert complicated.typetracer[0].form == complicated[0].form
    assert ak.to_list(complicated[0].prod(-1)) == {"x": [30], "y": [[30, 1, 77, 13]]}
    assert (
        complicated.typetracer[0].prod(-1).array.form
        == complicated[0].prod(-1).array.form
    )


def test_EmptyArray():
    offsets = ak.layout.Index64(np.array([0, 0, 0, 0], dtype=np.int64))
    array = ak.layout.ListOffsetArray64(offsets, ak.layout.EmptyArray())
    array = v1_to_v2(array)

    assert ak.to_list(array) == [[], [], []]

    assert ak.to_list(array.prod(-1)) == [1, 1, 1]
    assert array.typetracer.prod(-1).form == array.prod(-1).form

    offsets = ak.layout.Index64(np.array([0, 0, 0, 0], dtype=np.int64))
    array = ak.layout.ListOffsetArray64(
        offsets, ak.layout.NumpyArray(np.array([], dtype=np.int64))
    )
    array = v1_to_v2(array)

    assert ak.to_list(array) == [[], [], []]

    assert ak.to_list(array.prod(-1)) == [1, 1, 1]
    assert array.typetracer.prod(-1).form == array.prod(-1).form


def test_IndexedOptionArray():
    content = ak.layout.NumpyArray(np.array(primes[: 2 * 3 * 5], dtype=np.int64))
    offsets1 = ak.layout.Index64(np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.int64))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets1, content)
    index = ak.layout.Index64(np.array([5, 4, 3, 2, 1, 0], dtype=np.int64))
    indexedarray = ak.layout.IndexedArray64(index, listoffsetarray)
    offsets2 = ak.layout.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.layout.ListOffsetArray64(offsets2, indexedarray)
    depth2 = v1_to_v2(depth2)

    assert ak.to_list(depth2) == [
        [[101, 103, 107, 109, 113], [73, 79, 83, 89, 97], [53, 59, 61, 67, 71]],
        [[31, 37, 41, 43, 47], [13, 17, 19, 23, 29], [2, 3, 5, 7, 11]],
    ]

    assert ak.to_list(depth2.prod(-1)) == [
        [101 * 103 * 107 * 109 * 113, 73 * 79 * 83 * 89 * 97, 53 * 59 * 61 * 67 * 71],
        [31 * 37 * 41 * 43 * 47, 13 * 17 * 19 * 23 * 29, 2 * 3 * 5 * 7 * 11],
    ]
    assert depth2.typetracer.prod(-1).form == depth2.prod(-1).form

    assert ak.to_list(depth2.prod(-2)) == [
        [101 * 73 * 53, 103 * 79 * 59, 107 * 83 * 61, 109 * 89 * 67, 113 * 97 * 71],
        [31 * 13 * 2, 37 * 17 * 3, 41 * 19 * 5, 43 * 23 * 7, 47 * 29 * 11],
    ]
    assert depth2.typetracer.prod(-2).form == depth2.prod(-2).form

    assert ak.to_list(depth2.prod(-3)) == [
        [101 * 31, 103 * 37, 107 * 41, 109 * 43, 113 * 47],
        [73 * 13, 79 * 17, 83 * 19, 89 * 23, 97 * 29],
        [53 * 2, 59 * 3, 61 * 5, 67 * 7, 71 * 11],
    ]
    assert depth2.typetracer.prod(-3).form == depth2.prod(-3).form

    content = ak.layout.NumpyArray(
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
    offsets1 = ak.layout.Index64(np.array([0, 5, 10, 15, 20], dtype=np.int64))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets1, content)
    index = ak.layout.Index64(np.array([3, -1, 2, 1, -1, 0], dtype=np.int64))
    indexedoptionarray = ak.layout.IndexedOptionArray64(index, listoffsetarray)
    offsets2 = ak.layout.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.layout.ListOffsetArray64(offsets2, indexedoptionarray)
    depth2 = v1_to_v2(depth2)

    assert ak.to_list(depth2) == [
        [[101, 103, 107, 109, 113], None, [53, 59, 61, 67, 71]],
        [[31, 37, 41, 43, 47], None, [2, 3, 5, 7, 11]],
    ]

    assert ak.to_list(depth2.prod(-1)) == [
        [101 * 103 * 107 * 109 * 113, None, 53 * 59 * 61 * 67 * 71],
        [31 * 37 * 41 * 43 * 47, None, 2 * 3 * 5 * 7 * 11],
    ]
    assert depth2.typetracer.prod(-1).form == depth2.prod(-1).form

    assert ak.to_list(depth2.prod(-2)) == [
        [101 * 53, 103 * 59, 107 * 61, 109 * 67, 113 * 71],
        [31 * 2, 37 * 3, 41 * 5, 43 * 7, 47 * 11],
    ]
    assert depth2.typetracer.prod(-2).form == depth2.prod(-2).form

    assert ak.to_list(depth2.prod(-3)) == [
        [101 * 31, 103 * 37, 107 * 41, 109 * 43, 113 * 47],
        [],
        [53 * 2, 59 * 3, 61 * 5, 67 * 7, 71 * 11],
    ]
    assert depth2.typetracer.prod(-3).form == depth2.prod(-3).form

    content = ak.layout.NumpyArray(
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
    index = ak.layout.Index64(
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
    indexedoptionarray = ak.layout.IndexedOptionArray64(index, content)
    offsets1 = ak.layout.Index64(np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.int64))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets1, indexedoptionarray)
    offsets2 = ak.layout.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.layout.ListOffsetArray64(offsets2, listoffsetarray)
    depth2 = v1_to_v2(depth2)

    assert ak.to_list(depth2) == [
        [
            [101, 103, 107, 109, 113],
            [None, None, None, None, None],
            [53, 59, 61, 67, 71],
        ],
        [[31, 37, 41, 43, 47], [None, None, None, None, None], [2, 3, 5, 7, 11]],
    ]

    assert ak.to_list(depth2.prod(-1)) == [
        [101 * 103 * 107 * 109 * 113, 1 * 1 * 1 * 1 * 1, 53 * 59 * 61 * 67 * 71],
        [31 * 37 * 41 * 43 * 47, 1 * 1 * 1 * 1 * 1, 2 * 3 * 5 * 7 * 11],
    ]
    assert depth2.typetracer.prod(-1).form == depth2.prod(-1).form

    assert ak.to_list(depth2.prod(-2)) == [
        [101 * 53, 103 * 59, 107 * 61, 109 * 67, 113 * 71],
        [31 * 2, 37 * 3, 41 * 5, 43 * 7, 47 * 11],
    ]
    assert depth2.typetracer.prod(-2).form == depth2.prod(-2).form

    assert ak.to_list(depth2.prod(-3)) == [
        [101 * 31, 103 * 37, 107 * 41, 109 * 43, 113 * 47],
        [1, 1, 1, 1, 1],
        [53 * 2, 59 * 3, 61 * 5, 67 * 7, 71 * 11],
    ]
    assert depth2.typetracer.prod(-3).form == depth2.prod(-3).form

    content = ak.layout.NumpyArray(
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
    index = ak.layout.Index64(
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
    indexedoptionarray = ak.layout.IndexedOptionArray64(index, content)
    offsets1 = ak.layout.Index64(np.array([0, 5, 6, 11, 16, 17, 22], dtype=np.int64))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets1, indexedoptionarray)
    offsets2 = ak.layout.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.layout.ListOffsetArray64(offsets2, listoffsetarray)
    depth2 = v1_to_v2(depth2)

    assert ak.to_list(depth2) == [
        [[101, 103, 107, 109, 113], [None], [53, 59, 61, 67, 71]],
        [[31, 37, 41, 43, 47], [None], [2, 3, 5, 7, 11]],
    ]

    assert ak.to_list(depth2.prod(-1)) == [
        [101 * 103 * 107 * 109 * 113, 1, 53 * 59 * 61 * 67 * 71],
        [31 * 37 * 41 * 43 * 47, 1, 2 * 3 * 5 * 7 * 11],
    ]
    assert depth2.typetracer.prod(-1).form == depth2.prod(-1).form

    assert ak.to_list(depth2.prod(-2)) == [
        [101 * 53, 103 * 59, 107 * 61, 109 * 67, 113 * 71],
        [31 * 2, 37 * 3, 41 * 5, 43 * 7, 47 * 11],
    ]
    assert depth2.typetracer.prod(-2).form == depth2.prod(-2).form

    assert ak.to_list(depth2.prod(-3)) == [
        [101 * 31, 103 * 37, 107 * 41, 109 * 43, 113 * 47],
        [1],
        [53 * 2, 59 * 3, 61 * 5, 67 * 7, 71 * 11],
    ]
    assert depth2.typetracer.prod(-3).form == depth2.prod(-3).form


def test_UnionArray():
    content1 = ak.from_iter(
        [[[2, 3, 5, 7, 11], [13, 17, 19, 23, 29], [31, 37, 41, 43, 47]]],
        highlevel=False,
    )
    content2 = ak.from_iter(
        [[[53, 59, 61, 67, 71], [73, 79, 83, 89, 97], [101, 103, 107, 109, 113]]],
        highlevel=False,
    )

    tags = ak.layout.Index8(np.array([0, 1], dtype=np.int8))
    index = ak.layout.Index64(np.array([0, 0], dtype=np.int64))
    depth2 = ak.layout.UnionArray8_64(tags, index, [content1, content2])
    depth2 = v1_to_v2(depth2)

    assert ak.to_list(depth2) == [
        [[2, 3, 5, 7, 11], [13, 17, 19, 23, 29], [31, 37, 41, 43, 47]],
        [[53, 59, 61, 67, 71], [73, 79, 83, 89, 97], [101, 103, 107, 109, 113]],
    ]

    assert ak.to_list(depth2.prod(axis=-1)) == [
        [2 * 3 * 5 * 7 * 11, 13 * 17 * 19 * 23 * 29, 31 * 37 * 41 * 43 * 47],
        [53 * 59 * 61 * 67 * 71, 73 * 79 * 83 * 89 * 97, 101 * 103 * 107 * 109 * 113],
    ]
    assert depth2.typetracer.prod(axis=-1).form == depth2.prod(axis=-1).form
    assert ak.to_list(depth2.prod(axis=2)) == [
        [2 * 3 * 5 * 7 * 11, 13 * 17 * 19 * 23 * 29, 31 * 37 * 41 * 43 * 47],
        [53 * 59 * 61 * 67 * 71, 73 * 79 * 83 * 89 * 97, 101 * 103 * 107 * 109 * 113],
    ]
    assert depth2.typetracer.prod(axis=2).form == depth2.prod(axis=2).form

    assert ak.to_list(depth2.prod(axis=-2)) == [
        [2 * 13 * 31, 3 * 17 * 37, 5 * 19 * 41, 7 * 23 * 43, 11 * 29 * 47],
        [53 * 73 * 101, 59 * 79 * 103, 61 * 83 * 107, 67 * 89 * 109, 71 * 97 * 113],
    ]
    assert depth2.typetracer.prod(axis=-2).form == depth2.prod(axis=-2).form
    assert ak.to_list(depth2.prod(axis=1)) == [
        [2 * 13 * 31, 3 * 17 * 37, 5 * 19 * 41, 7 * 23 * 43, 11 * 29 * 47],
        [53 * 73 * 101, 59 * 79 * 103, 61 * 83 * 107, 67 * 89 * 109, 71 * 97 * 113],
    ]
    assert depth2.typetracer.prod(axis=1).form == depth2.prod(axis=1).form

    assert ak.to_list(depth2.prod(axis=-3)) == [
        [2 * 53, 3 * 59, 5 * 61, 7 * 67, 11 * 71],
        [13 * 73, 17 * 79, 19 * 83, 23 * 89, 29 * 97],
        [31 * 101, 37 * 103, 41 * 107, 43 * 109, 47 * 113],
    ]
    assert depth2.typetracer.prod(axis=-3).form == depth2.prod(axis=-3).form
    assert ak.to_list(depth2.prod(axis=0)) == [
        [2 * 53, 3 * 59, 5 * 61, 7 * 67, 11 * 71],
        [13 * 73, 17 * 79, 19 * 83, 23 * 89, 29 * 97],
        [31 * 101, 37 * 103, 41 * 107, 43 * 109, 47 * 113],
    ]
    assert depth2.typetracer.prod(axis=0).form == depth2.prod(axis=0).form

    content1 = ak.layout.NumpyArray(np.array(primes[: 2 * 3 * 5], dtype=np.int64))
    offsets1a = ak.layout.Index64(np.array([0, 5, 10, 15], dtype=np.int64))
    offsets1b = ak.layout.Index64(np.array([15, 20, 25, 30], dtype=np.int64))
    tags = ak.layout.Index8(np.array([0, 0, 0, 1, 1, 1], dtype=np.int8))
    index = ak.layout.Index64(np.array([0, 1, 2, 0, 1, 2], dtype=np.int64))
    unionarray = ak.layout.UnionArray8_64(
        tags,
        index,
        [
            ak.layout.ListOffsetArray64(offsets1a, content1),
            ak.layout.ListOffsetArray64(offsets1b, content1),
        ],
    )
    offsets2 = ak.layout.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.layout.ListOffsetArray64(offsets2, unionarray)
    depth2 = v1_to_v2(depth2)

    assert ak.to_list(depth2) == [
        [[2, 3, 5, 7, 11], [13, 17, 19, 23, 29], [31, 37, 41, 43, 47]],
        [[53, 59, 61, 67, 71], [73, 79, 83, 89, 97], [101, 103, 107, 109, 113]],
    ]

    assert ak.to_list(depth2.prod(axis=-1)) == [
        [2 * 3 * 5 * 7 * 11, 13 * 17 * 19 * 23 * 29, 31 * 37 * 41 * 43 * 47],
        [53 * 59 * 61 * 67 * 71, 73 * 79 * 83 * 89 * 97, 101 * 103 * 107 * 109 * 113],
    ]
    assert depth2.typetracer.prod(axis=-1).form == depth2.prod(axis=-1).form
    assert ak.to_list(depth2.prod(axis=2)) == [
        [2 * 3 * 5 * 7 * 11, 13 * 17 * 19 * 23 * 29, 31 * 37 * 41 * 43 * 47],
        [53 * 59 * 61 * 67 * 71, 73 * 79 * 83 * 89 * 97, 101 * 103 * 107 * 109 * 113],
    ]
    assert depth2.typetracer.prod(axis=2).form == depth2.prod(axis=2).form

    assert ak.to_list(depth2.prod(axis=-2)) == [
        [2 * 13 * 31, 3 * 17 * 37, 5 * 19 * 41, 7 * 23 * 43, 11 * 29 * 47],
        [53 * 73 * 101, 59 * 79 * 103, 61 * 83 * 107, 67 * 89 * 109, 71 * 97 * 113],
    ]
    assert depth2.typetracer.prod(axis=-2).form == depth2.prod(axis=-2).form
    assert ak.to_list(depth2.prod(axis=1)) == [
        [2 * 13 * 31, 3 * 17 * 37, 5 * 19 * 41, 7 * 23 * 43, 11 * 29 * 47],
        [53 * 73 * 101, 59 * 79 * 103, 61 * 83 * 107, 67 * 89 * 109, 71 * 97 * 113],
    ]
    assert depth2.typetracer.prod(axis=1).form == depth2.prod(axis=1).form

    assert ak.to_list(depth2.prod(axis=-3)) == [
        [2 * 53, 3 * 59, 5 * 61, 7 * 67, 11 * 71],
        [13 * 73, 17 * 79, 19 * 83, 23 * 89, 29 * 97],
        [31 * 101, 37 * 103, 41 * 107, 43 * 109, 47 * 113],
    ]
    assert depth2.typetracer.prod(axis=-3).form == depth2.prod(axis=-3).form
    assert ak.to_list(depth2.prod(axis=0)) == [
        [2 * 53, 3 * 59, 5 * 61, 7 * 67, 11 * 71],
        [13 * 73, 17 * 79, 19 * 83, 23 * 89, 29 * 97],
        [31 * 101, 37 * 103, 41 * 107, 43 * 109, 47 * 113],
    ]
    assert depth2.typetracer.prod(axis=0).form == depth2.prod(axis=0).form


def test_sum():
    content2 = ak.layout.NumpyArray(
        np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], dtype=np.int64)
    )
    offsets3 = ak.layout.Index64(np.array([0, 4, 8, 12], dtype=np.int64))
    depth1 = ak.layout.ListOffsetArray64(offsets3, content2)
    depth1 = v1_to_v2(depth1)

    assert ak.to_list(depth1.sum(-1)) == [
        1 + 2 + 4 + 8,
        16 + 32 + 64 + 128,
        256 + 512 + 1024 + 2048,
    ]
    assert depth1.typetracer.sum(-1).form == depth1.sum(-1).form
    assert ak.to_list(depth1.sum(1)) == [
        1 + 2 + 4 + 8,
        16 + 32 + 64 + 128,
        256 + 512 + 1024 + 2048,
    ]
    assert depth1.typetracer.sum(1).form == depth1.sum(1).form

    assert ak.to_list(depth1.sum(-2)) == [
        1 + 16 + 256,
        2 + 32 + 512,
        4 + 64 + 1024,
        8 + 128 + 2048,
    ]
    assert depth1.typetracer.sum(-2).form == depth1.sum(-2).form
    assert ak.to_list(depth1.sum(0)) == [
        1 + 16 + 256,
        2 + 32 + 512,
        4 + 64 + 1024,
        8 + 128 + 2048,
    ]
    assert depth1.typetracer.sum(0).form == depth1.sum(0).form


def test_sumprod_types_FIXME():
    array = np.array([[True, False, False], [True, False, False]])
    content2 = ak.layout.NumpyArray(array.reshape(-1))
    offsets3 = ak.layout.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth1 = ak.layout.ListOffsetArray64(offsets3, content2)
    depth1 = v1_to_v2(depth1)
    assert np.sum(array, axis=-1).dtype == np.asarray(depth1.sum(axis=-1)).dtype
    assert np.prod(array, axis=-1).dtype == np.asarray(depth1.prod(axis=-1)).dtype


def test_sumprod_types():
    def prod(xs):
        out = 1
        for x in xs:
            out *= x
        return out

    array = np.array([[True, False, False], [True, False, False]])
    content2 = ak.layout.NumpyArray(array.reshape(-1))
    offsets3 = ak.layout.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth1 = ak.layout.ListOffsetArray64(offsets3, content2)
    depth1 = v1_to_v2(depth1)

    assert sum(ak.to_list(np.sum(array, axis=-1))) == sum(
        ak.to_list(depth1.sum(axis=-1))
    )
    assert prod(ak.to_list(np.prod(array, axis=-1))) == prod(
        ak.to_list(depth1.prod(axis=-1))
    )

    array = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int8)
    content2 = ak.layout.NumpyArray(array.reshape(-1))
    offsets3 = ak.layout.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth1 = ak.layout.ListOffsetArray64(offsets3, content2)
    depth1 = v1_to_v2(depth1)

    assert np.sum(array, axis=-1).dtype == np.asarray(depth1.sum(axis=-1)).dtype
    assert np.prod(array, axis=-1).dtype == np.asarray(depth1.prod(axis=-1)).dtype
    assert sum(ak.to_list(np.sum(array, axis=-1))) == sum(
        ak.to_list(depth1.sum(axis=-1))
    )
    assert prod(ak.to_list(np.prod(array, axis=-1))) == prod(
        ak.to_list(depth1.prod(axis=-1))
    )

    array = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
    content2 = ak.layout.NumpyArray(array.reshape(-1))
    offsets3 = ak.layout.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth1 = ak.layout.ListOffsetArray64(offsets3, content2)
    depth1 = v1_to_v2(depth1)

    assert np.sum(array, axis=-1).dtype == np.asarray(depth1.sum(axis=-1)).dtype
    assert np.prod(array, axis=-1).dtype == np.asarray(depth1.prod(axis=-1)).dtype
    assert sum(ak.to_list(np.sum(array, axis=-1))) == sum(
        ak.to_list(depth1.sum(axis=-1))
    )
    assert prod(ak.to_list(np.prod(array, axis=-1))) == prod(
        ak.to_list(depth1.prod(axis=-1))
    )

    array = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int16)
    content2 = ak.layout.NumpyArray(array.reshape(-1))
    offsets3 = ak.layout.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth1 = ak.layout.ListOffsetArray64(offsets3, content2)
    depth1 = v1_to_v2(depth1)

    assert np.sum(array, axis=-1).dtype == np.asarray(depth1.sum(axis=-1)).dtype
    assert np.prod(array, axis=-1).dtype == np.asarray(depth1.prod(axis=-1)).dtype
    assert sum(ak.to_list(np.sum(array, axis=-1))) == sum(
        ak.to_list(depth1.sum(axis=-1))
    )
    assert prod(ak.to_list(np.prod(array, axis=-1))) == prod(
        ak.to_list(depth1.prod(axis=-1))
    )

    array = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint16)
    content2 = ak.layout.NumpyArray(array.reshape(-1))
    offsets3 = ak.layout.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth1 = ak.layout.ListOffsetArray64(offsets3, content2)
    depth1 = v1_to_v2(depth1)

    assert np.sum(array, axis=-1).dtype == np.asarray(depth1.sum(axis=-1)).dtype
    assert np.prod(array, axis=-1).dtype == np.asarray(depth1.prod(axis=-1)).dtype
    assert sum(ak.to_list(np.sum(array, axis=-1))) == sum(
        ak.to_list(depth1.sum(axis=-1))
    )
    assert prod(ak.to_list(np.prod(array, axis=-1))) == prod(
        ak.to_list(depth1.prod(axis=-1))
    )

    array = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    content2 = ak.layout.NumpyArray(array.reshape(-1))
    offsets3 = ak.layout.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth1 = ak.layout.ListOffsetArray64(offsets3, content2)
    depth1 = v1_to_v2(depth1)

    assert np.sum(array, axis=-1).dtype == np.asarray(depth1.sum(axis=-1)).dtype
    assert np.prod(array, axis=-1).dtype == np.asarray(depth1.prod(axis=-1)).dtype
    assert sum(ak.to_list(np.sum(array, axis=-1))) == sum(
        ak.to_list(depth1.sum(axis=-1))
    )
    assert prod(ak.to_list(np.prod(array, axis=-1))) == prod(
        ak.to_list(depth1.prod(axis=-1))
    )

    array = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)
    content2 = ak.layout.NumpyArray(array.reshape(-1))
    offsets3 = ak.layout.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth1 = ak.layout.ListOffsetArray64(offsets3, content2)
    depth1 = v1_to_v2(depth1)

    assert np.sum(array, axis=-1).dtype == np.asarray(depth1.sum(axis=-1)).dtype
    assert np.prod(array, axis=-1).dtype == np.asarray(depth1.prod(axis=-1)).dtype
    assert sum(ak.to_list(np.sum(array, axis=-1))) == sum(
        ak.to_list(depth1.sum(axis=-1))
    )
    assert prod(ak.to_list(np.prod(array, axis=-1))) == prod(
        ak.to_list(depth1.prod(axis=-1))
    )

    array = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    content2 = ak.layout.NumpyArray(array.reshape(-1))
    offsets3 = ak.layout.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth1 = ak.layout.ListOffsetArray64(offsets3, content2)
    depth1 = v1_to_v2(depth1)

    assert np.sum(array, axis=-1).dtype == np.asarray(depth1.sum(axis=-1)).dtype
    assert np.prod(array, axis=-1).dtype == np.asarray(depth1.prod(axis=-1)).dtype
    assert sum(ak.to_list(np.sum(array, axis=-1))) == sum(
        ak.to_list(depth1.sum(axis=-1))
    )
    assert prod(ak.to_list(np.prod(array, axis=-1))) == prod(
        ak.to_list(depth1.prod(axis=-1))
    )

    array = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint64)
    content2 = ak.layout.NumpyArray(array.reshape(-1))
    offsets3 = ak.layout.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth1 = ak.layout.ListOffsetArray64(offsets3, content2)
    depth1 = v1_to_v2(depth1)

    assert np.sum(array, axis=-1).dtype == np.asarray(depth1.sum(axis=-1)).dtype
    assert np.prod(array, axis=-1).dtype == np.asarray(depth1.prod(axis=-1)).dtype
    assert sum(ak.to_list(np.sum(array, axis=-1))) == sum(
        ak.to_list(depth1.sum(axis=-1))
    )
    assert prod(ak.to_list(np.prod(array, axis=-1))) == prod(
        ak.to_list(depth1.prod(axis=-1))
    )


def test_any():
    content2 = ak.layout.NumpyArray(
        np.array([1.1, 2.2, 3.3, 0.0, 2.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    offsets3 = ak.layout.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak.layout.ListOffsetArray64(offsets3, content2)
    depth1 = v1_to_v2(depth1)

    assert ak.to_list(depth1) == [
        [1.1, 2.2, 3.3],
        [0.0, 2.2, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]

    assert ak.to_list(depth1.any(-1)) == [True, True, False]
    assert depth1.typetracer.any(-1).form == depth1.any(-1).form
    assert ak.to_list(depth1.any(1)) == [True, True, False]
    assert depth1.typetracer.any(1).form == depth1.any(1).form

    assert ak.to_list(depth1.any(-2)) == [True, True, True, False]
    assert depth1.typetracer.any(-2).form == depth1.any(-2).form
    assert ak.to_list(depth1.any(0)) == [True, True, True, False]
    assert depth1.typetracer.any(0).form == depth1.any(0).form


def test_all():
    content2 = ak.layout.NumpyArray(
        np.array([1.1, 2.2, 3.3, 0.0, 2.2, 0.0, 0.0, 2.2, 0.0, 4.4])
    )
    offsets3 = ak.layout.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak.layout.ListOffsetArray64(offsets3, content2)
    depth1 = v1_to_v2(depth1)

    assert ak.to_list(depth1) == [
        [1.1, 2.2, 3.3],
        [0.0, 2.2, 0.0],
        [0.0, 2.2, 0.0, 4.4],
    ]

    assert ak.to_list(depth1.all(-1)) == [True, False, False]
    assert depth1.typetracer.all(-1).form == depth1.all(-1).form
    assert ak.to_list(depth1.all(1)) == [True, False, False]
    assert depth1.typetracer.all(1).form == depth1.all(1).form

    assert ak.to_list(depth1.all(-2)) == [False, True, False, True]
    assert depth1.typetracer.all(-2).form == depth1.all(-2).form
    assert ak.to_list(depth1.all(0)) == [False, True, False, True]
    assert depth1.typetracer.all(0).form == depth1.all(0).form


def test_count():
    content2 = ak.layout.NumpyArray(
        np.array([1.1, 2.2, 3.3, 0.0, 2.2, 0.0, 0.0, 2.2, 0.0, 4.4])
    )
    offsets3 = ak.layout.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak.layout.ListOffsetArray64(offsets3, content2)
    depth1 = v1_to_v2(depth1)

    assert ak.to_list(depth1) == [
        [1.1, 2.2, 3.3],
        [0.0, 2.2, 0.0],
        [0.0, 2.2, 0.0, 4.4],
    ]

    assert ak.to_list(depth1.count(-1)) == [3, 3, 4]
    assert depth1.typetracer.count(-1).form == depth1.count(-1).form
    assert ak.to_list(depth1.count(1)) == [3, 3, 4]
    assert depth1.typetracer.count(1).form == depth1.count(1).form

    assert ak.to_list(depth1.count(-2)) == [3, 3, 3, 1]
    assert depth1.typetracer.count(-2).form == depth1.count(-2).form
    assert ak.to_list(depth1.count(0)) == [3, 3, 3, 1]
    assert depth1.typetracer.count(0).form == depth1.count(0).form


def test_count_nonzero():
    content2 = ak.layout.NumpyArray(
        np.array([1.1, 2.2, 3.3, 0.0, 2.2, 0.0, 0.0, 2.2, 0.0, 4.4])
    )
    offsets3 = ak.layout.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak.layout.ListOffsetArray64(offsets3, content2)
    depth1 = v1_to_v2(depth1)

    assert ak.to_list(depth1) == [
        [1.1, 2.2, 3.3],
        [0.0, 2.2, 0.0],
        [0.0, 2.2, 0.0, 4.4],
    ]

    assert ak.to_list(depth1.count_nonzero(-1)) == [3, 1, 2]
    assert depth1.typetracer.count_nonzero(-1).form == depth1.count_nonzero(-1).form
    assert ak.to_list(depth1.count_nonzero(1)) == [3, 1, 2]
    assert depth1.typetracer.count_nonzero(1).form == depth1.count_nonzero(1).form

    assert ak.to_list(depth1.count_nonzero(-2)) == [1, 3, 1, 1]
    assert depth1.typetracer.count_nonzero(-2).form == depth1.count_nonzero(-2).form
    assert ak.to_list(depth1.count_nonzero(0)) == [1, 3, 1, 1]
    assert depth1.typetracer.count_nonzero(0).form == depth1.count_nonzero(0).form


def test_count_min():
    content2 = ak.layout.NumpyArray(
        np.array([1.1, 2.2, 3.3, 0.0, 2.2, 0.0, 0.0, 2.2, 0.0, 4.4])
    )
    offsets3 = ak.layout.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak.layout.ListOffsetArray64(offsets3, content2)
    depth1 = v1_to_v2(depth1)

    assert ak.to_list(depth1) == [
        [1.1, 2.2, 3.3],
        [0.0, 2.2, 0.0],
        [0.0, 2.2, 0.0, 4.4],
    ]

    assert ak.to_list(depth1.min(-1)) == [1.1, 0.0, 0.0]
    assert depth1.typetracer.min(-1).form == depth1.min(-1).form
    assert ak.to_list(depth1.min(1)) == [1.1, 0.0, 0.0]
    assert depth1.typetracer.min(1).form == depth1.min(1).form

    assert ak.to_list(depth1.min(-2)) == [0.0, 2.2, 0.0, 4.4]
    assert depth1.typetracer.min(-2).form == depth1.min(-2).form
    assert ak.to_list(depth1.min(0)) == [0.0, 2.2, 0.0, 4.4]
    assert depth1.typetracer.min(0).form == depth1.min(0).form

    content2 = ak.layout.NumpyArray(
        np.array([True, True, True, False, True, False, False, True, False, True])
    )
    offsets3 = ak.layout.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak.layout.ListOffsetArray64(offsets3, content2)
    depth1 = v1_to_v2(depth1)

    assert ak.to_list(depth1) == [
        [True, True, True],
        [False, True, False],
        [False, True, False, True],
    ]

    assert ak.to_list(depth1.min(-1)) == [True, False, False]
    assert depth1.typetracer.min(-1).form == depth1.min(-1).form
    assert ak.to_list(depth1.min(1)) == [True, False, False]
    assert depth1.typetracer.min(1).form == depth1.min(1).form

    assert ak.to_list(depth1.min(-2)) == [False, True, False, True]
    assert depth1.typetracer.min(-2).form == depth1.min(-2).form
    assert ak.to_list(depth1.min(0)) == [False, True, False, True]
    assert depth1.typetracer.min(0).form == depth1.min(0).form


def test_count_max():
    content2 = ak.layout.NumpyArray(
        np.array([1.1, 2.2, 3.3, 0.0, 2.2, 0.0, 0.0, 2.2, 0.0, 4.4])
    )
    offsets3 = ak.layout.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak.layout.ListOffsetArray64(offsets3, content2)
    depth1 = v1_to_v2(depth1)

    assert ak.to_list(depth1) == [
        [1.1, 2.2, 3.3],
        [0.0, 2.2, 0.0],
        [0.0, 2.2, 0.0, 4.4],
    ]

    assert ak.to_list(depth1.max(-1)) == [3.3, 2.2, 4.4]
    assert depth1.typetracer.max(-1).form == depth1.max(-1).form
    assert ak.to_list(depth1.max(1)) == [3.3, 2.2, 4.4]
    assert depth1.typetracer.max(1).form == depth1.max(1).form

    assert ak.to_list(depth1.max(-2)) == [1.1, 2.2, 3.3, 4.4]
    assert depth1.typetracer.max(-2).form == depth1.max(-2).form
    assert ak.to_list(depth1.max(0)) == [1.1, 2.2, 3.3, 4.4]
    assert depth1.typetracer.max(0).form == depth1.max(0).form

    content2 = ak.layout.NumpyArray(
        np.array([False, True, True, False, True, False, False, False, False, False])
    )
    offsets3 = ak.layout.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak.layout.ListOffsetArray64(offsets3, content2)
    depth1 = v1_to_v2(depth1)

    assert ak.to_list(depth1) == [
        [False, True, True],
        [False, True, False],
        [False, False, False, False],
    ]

    assert ak.to_list(depth1.max(-1)) == [True, True, False]
    assert depth1.typetracer.max(-1).form == depth1.max(-1).form
    assert ak.to_list(depth1.max(1)) == [True, True, False]
    assert depth1.typetracer.max(1).form == depth1.max(1).form

    assert ak.to_list(depth1.max(-2)) == [False, True, True, False]
    assert depth1.typetracer.max(-2).form == depth1.max(-2).form
    assert ak.to_list(depth1.max(0)) == [False, True, True, False]
    assert depth1.typetracer.max(0).form == depth1.max(0).form


def test_mask():
    content = ak.layout.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak.layout.Index64(np.array([0, 3, 3, 5, 6, 6, 6, 9], dtype=np.int64))
    array = ak.layout.ListOffsetArray64(offsets, content)
    array = v1_to_v2(array)

    assert ak.to_list(array.min(axis=-1, mask=False)) == [
        1.1,
        np.inf,
        4.4,
        6.6,
        np.inf,
        np.inf,
        7.7,
    ]
    assert (
        array.typetracer.min(axis=-1, mask=False).form
        == array.min(axis=-1, mask=False).form
    )
    assert ak.to_list(array.min(axis=-1, mask=True)) == [
        1.1,
        None,
        4.4,
        6.6,
        None,
        None,
        7.7,
    ]
    assert (
        array.typetracer.min(axis=-1, mask=True).form
        == array.min(axis=-1, mask=True).form
    )


def test_ByteMaskedArray():
    content = ak.from_iter(
        [
            [[1.1, 0.0, 2.2], [], [3.3, 4.4]],
            [],
            [[5.5]],
            [[6.6, 9.9, 8.8, 7.7]],
            [[], [12.2, 11.1, 10.0]],
        ],
        highlevel=False,
    )
    mask = ak.layout.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    v1_array = ak.layout.ByteMaskedArray(mask, content, valid_when=False)
    v2_array = v1_to_v2(v1_array)

    assert ak.to_list(v2_array) == [
        [[1.1, 0.0, 2.2], [], [3.3, 4.4]],
        [],
        None,
        None,
        [[], [12.2, 11.1, 10.0]],
    ]
    assert ak.to_list(v2_array.argmin(axis=-1)) == [
        [1, None, 0],
        [],
        None,
        None,
        [None, 2],
    ]
    assert v2_array.typetracer.argmin(axis=-1).form == v2_array.argmin(axis=-1).form


def test_keepdims():
    nparray = np.array(primes[: 2 * 3 * 5], dtype=np.int64).reshape(2, 3, 5)
    content1 = ak.layout.NumpyArray(np.array(primes[: 2 * 3 * 5], dtype=np.int64))
    offsets1 = ak.layout.Index64(np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.int64))
    offsets2 = ak.layout.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.layout.ListOffsetArray64(
        offsets2, ak.layout.ListOffsetArray64(offsets1, content1)
    )
    depth2 = v1_to_v2(depth2)

    assert ak.to_list(depth2) == [
        [[2, 3, 5, 7, 11], [13, 17, 19, 23, 29], [31, 37, 41, 43, 47]],
        [[53, 59, 61, 67, 71], [73, 79, 83, 89, 97], [101, 103, 107, 109, 113]],
    ]

    assert ak.to_list(depth2.prod(axis=-1, keepdims=False)) == ak.to_list(
        nparray.prod(axis=-1, keepdims=False)
    )
    assert (
        depth2.typetracer.prod(axis=-1, keepdims=False).form
        == depth2.prod(axis=-1, keepdims=False).form
    )
    assert ak.to_list(depth2.prod(axis=-2, keepdims=False)) == ak.to_list(
        nparray.prod(axis=-2, keepdims=False)
    )
    assert (
        depth2.typetracer.prod(axis=-2, keepdims=False).form
        == depth2.prod(axis=-2, keepdims=False).form
    )
    assert ak.to_list(depth2.prod(axis=-3, keepdims=False)) == ak.to_list(
        nparray.prod(axis=-3, keepdims=False)
    )
    assert (
        depth2.typetracer.prod(axis=-3, keepdims=False).form
        == depth2.prod(axis=-3, keepdims=False).form
    )

    assert ak.to_list(depth2.prod(axis=-1, keepdims=True)) == ak.to_list(
        nparray.prod(axis=-1, keepdims=True)
    )
    assert (
        depth2.typetracer.prod(axis=-1, keepdims=True).form
        == depth2.prod(axis=-1, keepdims=True).form
    )
    assert ak.to_list(depth2.prod(axis=-2, keepdims=True)) == ak.to_list(
        nparray.prod(axis=-2, keepdims=True)
    )
    assert (
        depth2.typetracer.prod(axis=-2, keepdims=True).form
        == depth2.prod(axis=-2, keepdims=True).form
    )
    assert ak.to_list(depth2.prod(axis=-3, keepdims=True)) == ak.to_list(
        nparray.prod(axis=-3, keepdims=True)
    )
    assert (
        depth2.typetracer.prod(axis=-3, keepdims=True).form
        == depth2.prod(axis=-3, keepdims=True).form
    )


def test_highlevel():
    array = ak._v2.highlevel.Array(
        [[[2, 3, 5], [], [7, 11], [13]], [], [[17, 19], [23]]], check_valid=True
    )

    assert ak._v2.operations.reducers.count(array) == 9
    assert ak.to_list(ak._v2.operations.reducers.count(array, axis=-1)) == [
        [3, 0, 2, 1],
        [],
        [2, 1],
    ]
    assert ak.to_list(ak._v2.operations.reducers.count(array, axis=2)) == [
        [3, 0, 2, 1],
        [],
        [2, 1],
    ]
    assert ak.to_list(
        ak._v2.operations.reducers.count(array, axis=-1, keepdims=True)
    ) == [
        [[3], [0], [2], [1]],
        [],
        [[2], [1]],
    ]
    assert ak.to_list(ak._v2.operations.reducers.count(array, axis=-2)) == [
        [3, 2, 1],
        [],
        [2, 1],
    ]
    assert ak.to_list(ak._v2.operations.reducers.count(array, axis=1)) == [
        [3, 2, 1],
        [],
        [2, 1],
    ]
    assert ak.to_list(
        ak._v2.operations.reducers.count(array, axis=-2, keepdims=True)
    ) == [
        [[3, 2, 1]],
        [[]],
        [[2, 1]],
    ]

    assert ak._v2.operations.reducers.count_nonzero(array) == 9
    assert ak.to_list(ak._v2.operations.reducers.count_nonzero(array, axis=-1)) == [
        [3, 0, 2, 1],
        [],
        [2, 1],
    ]
    assert ak.to_list(ak._v2.operations.reducers.count_nonzero(array, axis=-2)) == [
        [3, 2, 1],
        [],
        [2, 1],
    ]

    assert (
        ak._v2.operations.reducers.sum(array) == 2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23
    )
    assert ak.to_list(ak._v2.operations.reducers.sum(array, axis=-1)) == [
        [2 + 3 + 5, 0, 7 + 11, 13],
        [],
        [17 + 19, 23],
    ]
    assert ak.to_list(ak._v2.operations.reducers.sum(array, axis=-2)) == [
        [2 + 7 + 13, 3 + 11, 5],
        [],
        [17 + 23, 19],
    ]

    assert (
        ak._v2.operations.reducers.prod(array) == 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23
    )
    assert ak.to_list(ak._v2.operations.reducers.prod(array, axis=-1)) == [
        [2 * 3 * 5, 1, 7 * 11, 13],
        [],
        [17 * 19, 23],
    ]
    assert ak.to_list(ak._v2.operations.reducers.prod(array, axis=-2)) == [
        [2 * 7 * 13, 3 * 11, 5],
        [],
        [17 * 23, 19],
    ]

    assert ak._v2.operations.reducers.min(array) == 2
    assert ak.to_list(ak._v2.operations.reducers.min(array, axis=-1)) == [
        [2, None, 7, 13],
        [],
        [17, 23],
    ]
    assert ak.to_list(ak._v2.operations.reducers.min(array, axis=-2)) == [
        [2, 3, 5],
        [],
        [17, 19],
    ]

    assert ak._v2.operations.reducers.max(array) == 23
    assert ak.to_list(ak._v2.operations.reducers.max(array, axis=-1)) == [
        [5, None, 11, 13],
        [],
        [19, 23],
    ]
    assert ak.to_list(ak._v2.operations.reducers.max(array, axis=-2)) == [
        [13, 11, 5],
        [],
        [23, 19],
    ]

    array = ak._v2.highlevel.Array(
        [
            [[True, False, True], [], [False, False], [True]],
            [],
            [[False, True], [True]],
        ],
        check_valid=True,
    )

    assert ak._v2.operations.reducers.any(array) is np.bool_(True)
    assert ak.to_list(ak._v2.operations.reducers.any(array, axis=-1)) == [
        [True, False, False, True],
        [],
        [True, True],
    ]
    assert ak.to_list(ak._v2.operations.reducers.any(array, axis=-2)) == [
        [True, False, True],
        [],
        [True, True],
    ]

    assert ak._v2.operations.reducers.all(array) is np.bool_(False)
    assert ak.to_list(ak._v2.operations.reducers.all(array, axis=-1)) == [
        [False, True, False, True],
        [],
        [False, True],
    ]
    assert ak.to_list(ak._v2.operations.reducers.all(array, axis=-2)) == [
        [False, False, True],
        [],
        [False, True],
    ]


def test_nonreducers():
    x = ak._v2.highlevel.Array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], check_valid=True)
    y = ak._v2.highlevel.Array(
        [[1.1, 2.2, 2.9, 4.0, 5.1], [0.9, 2.1, 3.2, 4.1, 4.9]], check_valid=True
    )

    assert ak.mean(y) == np.mean(ak._v2.operations.convert.to_numpy(y))
    assert ak.var(y) == np.var(ak._v2.operations.convert.to_numpy(y))
    assert ak.var(y, ddof=1) == np.var(ak._v2.operations.convert.to_numpy(y), ddof=1)
    assert ak.std(y) == np.std(ak._v2.operations.convert.to_numpy(y))
    assert ak.std(y, ddof=1) == np.std(ak._v2.operations.convert.to_numpy(y), ddof=1)

    assert ak.moment(y, 1) == np.mean(ak._v2.operations.convert.to_numpy(y))
    assert ak.moment(y - ak.mean(y), 2) == np.var(ak._v2.operations.convert.to_numpy(y))
    assert ak.covar(y, y) == np.var(ak._v2.operations.convert.to_numpy(y))
    assert ak.corr(y, y) == 1.0

    assert ak.corr(x, y) == pytest.approx(0.9968772535047296)
    fit = ak.linear_fit(x, y)
    assert ak.to_list(fit) == pytest.approx(
        {
            "intercept": 0.07999999999999773,
            "slope": 0.99,
            "intercept_error": 0.7416198487095663,
            "slope_error": 0.22360679774997896,
        }
    )

    assert ak.to_list(ak.mean(y, axis=-1)) == pytest.approx(
        ak.to_list(np.mean(ak._v2.operations.convert.to_numpy(y), axis=-1))
    )
    assert ak.to_list(ak.var(y, axis=-1)) == pytest.approx(
        ak.to_list(np.var(ak._v2.operations.convert.to_numpy(y), axis=-1))
    )
    assert ak.to_list(ak.var(y, axis=-1, ddof=1)) == pytest.approx(
        ak.to_list(np.var(ak._v2.operations.convert.to_numpy(y), axis=-1, ddof=1))
    )
    assert ak.to_list(ak.std(y, axis=-1)) == pytest.approx(
        ak.to_list(np.std(ak._v2.operations.convert.to_numpy(y), axis=-1))
    )
    assert ak.to_list(ak.std(y, axis=-1, ddof=1)) == pytest.approx(
        ak.to_list(np.std(ak._v2.operations.convert.to_numpy(y), axis=-1, ddof=1))
    )

    assert ak.to_list(ak.moment(y, 1, axis=-1)) == pytest.approx(
        ak.to_list(np.mean(ak._v2.operations.convert.to_numpy(y), axis=-1))
    )
    assert ak.to_list(ak.moment(y - ak.mean(y, axis=-1), 2, axis=-1)) == pytest.approx(
        ak.to_list(np.var(ak._v2.operations.convert.to_numpy(y), axis=-1))
    )
    assert ak.to_list(ak.covar(y, y, axis=-1)) == pytest.approx(
        ak.to_list(np.var(ak._v2.operations.convert.to_numpy(y), axis=-1))
    )
    assert ak.to_list(ak.corr(y, y, axis=-1)) == pytest.approx([1.0, 1.0])

    assert ak.to_list(ak.corr(x, y, axis=-1)) == pytest.approx(
        [0.9975103695813371, 0.9964193240901015]
    )
    fit = ak.linear_fit(x, y, axis=-1)
    assert ak.to_list(fit[0]) == pytest.approx(
        {
            "intercept": 0.11999999999999772,
            "slope": 0.9800000000000005,
            "intercept_error": 1.0488088481701516,
            "slope_error": 0.31622776601683794,
        }
    )
    assert ak.to_list(fit[1]) == pytest.approx(
        {
            "intercept": 0.04000000000000228,
            "slope": 0.9999999999999994,
            "intercept_error": 1.0488088481701516,
            "slope_error": 0.31622776601683794,
        }
    )


def test_softmax():
    array = ak._v2.highlevel.Array(
        [[np.log(2), np.log(2), np.log(4)], [], [np.log(5), np.log(5)]],
        check_valid=True,
    )
    assert ak.to_list(ak.softmax(array, axis=-1)) == [
        pytest.approx([0.25, 0.25, 0.5]),
        [],
        pytest.approx([0.5, 0.5]),
    ]
