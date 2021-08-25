# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2, v2_to_v1


def test_numpyarray_sort():
    v1_array = ak.from_numpy(
        np.array([3.3, 2.2, 1.1, 5.5, 4.4]), regulararray=True, highlevel=False
    )
    v2_array = v1_to_v2(v1_array)
    assert ak.to_list(np.sort(v2_array)) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]
    assert ak.to_list(v2_array.sort()) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]


def test_bool_sort():
    v1_array = ak.from_numpy(
        np.array([True, False, True, False, False]), regulararray=True, highlevel=False
    )
    v2_array = v1_to_v2(v1_array)
    assert ak.to_list(v2_array.sort()) == [
        False,
        False,
        False,
        True,
        True,
    ]


def test_emptyarray_sort():
    v2_array = ak._v2.contents.emptyarray.EmptyArray()
    assert ak.to_list(v2_array.sort()) == []


def test_listarray_sort():
    v2_array = ak._v2.contents.listarray.ListArray(  # noqa: F841
        ak._v2.index.Index(np.array([4, 100, 1])),
        ak._v2.index.Index(np.array([7, 100, 3, 200])),
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    np.array([6.6, 4.4, 5.5, 7.7, 1.1, 2.2, 3.3, 8.8])
                )
            ],
            ["nest"],
        ),
    )

    assert ak.to_list(v2_to_v1(v2_array)) == [
        [{"nest": 1.1}, {"nest": 2.2}, {"nest": 3.3}],
        [],
        [{"nest": 4.4}, {"nest": 5.5}],
    ]
    # FIXME: assert ak.to_list(v2_to_v1(v2_array.sort())) == [[{'nest': 1.1}, {'nest': 2.2}, {'nest': 3.3}], [], [{'nest': 4.4}, {'nest': 5.5}]]


def test_listoffsetarray_sort():
    v2_array = ak._v2.contents.listoffsetarray.ListOffsetArray(  # noqa: F841
        ak._v2.index.Index(np.array([1, 4, 4, 6])),
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    [6.6, 1.1, 3.3, 2.2, 7.7, 5.5, 4.4]
                )
            ],
            ["nest"],
        ),
    )
    assert ak.to_list(v2_to_v1(v2_array)) == [
        [{"nest": 1.1}, {"nest": 3.3}, {"nest": 2.2}],
        [],
        [{"nest": 7.7}, {"nest": 5.5}],
    ]
    assert ak.to_list(v2_to_v1(v2_array.sort())) == [
        [{"nest": 1.1}, {"nest": 2.2}, {"nest": 3.3}],
        [],
        [{"nest": 5.5}, {"nest": 7.7}],
    ]

    v1_array = ak.from_iter(
        [[3.3, 2.2, 1.1], [], [5.5, 4.4], [6.6], [9.9, 7.7, 8.8, 10.1]], highlevel=False
    )
    v2_array = v1_to_v2(v1_array)
    assert ak.to_list(v2_to_v1(v2_array.sort())) == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
        [6.6],
        [7.7, 8.8, 9.9, 10.10],
    ]


def test_regulararray_sort():
    v1_array = ak.from_numpy(
        np.array(
            [
                [
                    [3.3, 1.1, 5.5, 2.2, 4.4],
                    [8.8, 6.6, 9.9, 7.7, 10.10],
                    [11.11, 14.14, 15.15, 12.12, 13.13],
                ],
                [
                    [-1.1, -2.2, -5.5, -3.3, -4.4],
                    [-7.7, -8.8, -9.9, -6.6, -10.1],
                    [-13.13, -11.11, -12.12, -14.14, -15.15],
                ],
            ]
        ),
        regulararray=True,
        highlevel=False,
    )
    v2_array = v1_to_v2(v1_array)
    assert ak.to_list(v2_array) == [
        [
            [3.3, 1.1, 5.5, 2.2, 4.4],
            [8.8, 6.6, 9.9, 7.7, 10.1],
            [11.11, 14.14, 15.15, 12.12, 13.13],
        ],
        [
            [-1.1, -2.2, -5.5, -3.3, -4.4],
            [-7.7, -8.8, -9.9, -6.6, -10.1],
            [-13.13, -11.11, -12.12, -14.14, -15.15],
        ],
    ]
    assert ak.to_list(v2_to_v1(v2_array.sort())) == [
        [
            [1.1, 2.2, 3.3, 4.4, 5.5],
            [6.6, 7.7, 8.8, 9.9, 10.1],
            [11.11, 12.12, 13.13, 14.14, 15.15],
        ],
        [
            [-5.5, -4.4, -3.3, -2.2, -1.1],
            [-10.1, -9.9, -8.8, -7.7, -6.6],
            [-15.15, -14.14, -13.13, -12.12, -11.11],
        ],
    ]


# def test_bytemaskedarray_sort():
#     content = ak.from_iter(
#         [
#             [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
#             [],
#             [[5.5]],
#             [[6.6, 7.7, 8.8, 9.9]],
#             [[], [10.0, 11.1, 12.2]],
#         ],
#         highlevel=False,
#     )
#     mask = ak.layout.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
#     v1_array = ak.layout.ByteMaskedArray(mask, content, valid_when=False)
#     v2_array = v1_to_v2(v1_array)
#
#     assert ak.to_list(v1_array) == [
#         [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
#         [],
#         None,
#         None,
#         [[], [10.0, 11.1, 12.2]],
#     ]
#     assert ak.to_list(v2_to_v1(v2_array.sort(axis=1))) == [
#         [0, 1, 2],
#         [],
#         None,
#         None,
#         [0, 1],
#     ]
#
#
# def test_bitmaskedarray_sort():
#     v2_array = ak._v2.contents.bitmaskedarray.BitMaskedArray(
#         ak._v2.index.Index(
#             np.packbits(
#                 np.array(
#                     [
#                         1,
#                         1,
#                         1,
#                         1,
#                         0,
#                         0,
#                         0,
#                         0,
#                         1,
#                         0,
#                         1,
#                         0,
#                         1,
#                     ],
#                     dtype=np.uint8,
#                 )
#             )
#         ),
#         ak._v2.contents.numpyarray.NumpyArray(
#             np.array(
#                 [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
#             )
#         ),
#         valid_when=True,
#         length=13,
#         lsb_order=False,
#     )
#
#     assert ak.to_list(v2_to_v1(v2_array.sort(-1))) == [
#         0,
#         1,
#         2,
#         3,
#         4,
#         5,
#         6,
#         7,
#         8,
#         9,
#         10,
#         11,
#         12,
#     ]
#
#
# def test_unmaskedarray_sort():
#     v2_array = ak._v2.contents.unmaskedarray.UnmaskedArray(
#         ak._v2.contents.numpyarray.NumpyArray(
#             np.array([0.0, 1.1, 2.2, 3.3], dtype=np.float64)
#         )
#     )
#     assert ak.to_list(v2_to_v1(v2_array.sort(-1))) == [0, 1, 2, 3]


def test_unionarray_sort():
    v2_array = ak._v2.contents.unionarray.UnionArray(
        ak._v2.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], dtype=np.int8)),
        ak._v2.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100])),
        [
            ak._v2.contents.numpyarray.NumpyArray(np.array([1, 2, 3])),
            ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
    )
    assert ak.to_list(v2_to_v1(v2_array)) == [5.5, 4.4, 1, 2, 3.3, 3, 5.5]

    with pytest.raises(ValueError) as err:
        v2_array.sort()
    assert str(err.value).startswith("cannot sort unsimplified UnionArray")


# def test_recordarray_sort():
#     v2_array = ak._v2.contents.regulararray.RegularArray(  # noqa: F841
#         ak._v2.contents.recordarray.RecordArray(
#             [
#                 ak._v2.contents.numpyarray.NumpyArray(
#                     np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
#                 )
#             ],
#             ["nest"],
#         ),
#         3,
#     )
#     assert ak.to_list(v2_to_v1(v2_array.sort(-1))) == [[0, 1, 2], [0, 1, 2]]
#
#
# def test_indexedarray_sort():
#     v2_array = ak._v2.contents.indexedarray.IndexedArray(  # noqa: F841
#         ak._v2.index.Index(np.array([2, 2, 0, 1, 4, 5, 4])),
#         ak._v2.contents.recordarray.RecordArray(
#             [
#                 ak._v2.contents.numpyarray.NumpyArray(
#                     np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
#                 )
#             ],
#             ["nest"],
#         ),
#     )
#     assert ak.to_list(v2_to_v1(v2_array.sort(-1))) == [0, 1, 2, 3, 4, 5, 6]
#
#
# def test_indexedoptionarray_sort():
#     v2_array = ak._v2.contents.indexedoptionarray.IndexedOptionArray(  # noqa: F841
#         ak._v2.index.Index(np.array([2, 2, -1, 1, -1, 5, 4])),
#         ak._v2.contents.recordarray.RecordArray(
#             [
#                 ak._v2.contents.numpyarray.NumpyArray(
#                     np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
#                 )
#             ],
#             ["nest"],
#         ),
#     )
#     assert ak.to_list(v2_to_v1(v2_array.sort(-1))) == [0, 1, 2, 3, 4, 5, 6]
#
#
# def test_bytemaskedarray_sort():
#     v2_array = ak._v2.contents.bytemaskedarray.ByteMaskedArray(  # noqa: F841
#         ak._v2.index.Index(np.array([1, 0, 1, 0, 1], dtype=np.int8)),
#         ak._v2.contents.recordarray.RecordArray(
#             [
#                 ak._v2.contents.numpyarray.NumpyArray(
#                     np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
#                 )
#             ],
#             ["nest"],
#         ),
#         valid_when=True,
#     )
#
#     assert ak.to_list(v2_to_v1(v2_array.sort(-1))) == [0, 1, 2, 3, 4]
#
#
# def test_bitmaskedarray_sort():
#     v2_array = ak._v2.contents.bitmaskedarray.BitMaskedArray(
#         ak._v2.index.Index(
#             np.packbits(
#                 np.array(
#                     [
#                         True,
#                         True,
#                         True,
#                         True,
#                         False,
#                         False,
#                         False,
#                         False,
#                         True,
#                         False,
#                         True,
#                         False,
#                         True,
#                     ]
#                 )
#             )
#         ),
#         ak._v2.contents.recordarray.RecordArray(
#             [
#                 ak._v2.contents.numpyarray.NumpyArray(
#                     np.array(
#                         [
#                             0.0,
#                             1.0,
#                             2.0,
#                             3.0,
#                             4.0,
#                             5.0,
#                             6.0,
#                             7.0,
#                             1.1,
#                             2.2,
#                             3.3,
#                             4.4,
#                             5.5,
#                             6.6,
#                         ]
#                     )
#                 )
#             ],
#             ["nest"],
#         ),
#         valid_when=True,
#         length=13,
#         lsb_order=False,
#     )
#
#     assert ak.to_list(v2_to_v1(v2_array.sort(-1))) == [
#         0,
#         1,
#         2,
#         3,
#         4,
#         5,
#         6,
#         7,
#         8,
#         9,
#         10,
#         11,
#         12,
#     ]
#
#
# def test_unionarray_sort():
#     v2_array = ak._v2.contents.unionarray.UnionArray(  # noqa: F841
#         ak._v2.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], dtype=np.int8)),
#         ak._v2.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100])),
#         [
#             ak._v2.contents.recordarray.RecordArray(
#                 [ak._v2.contents.numpyarray.NumpyArray(np.array([1, 2, 3]))], ["nest"]
#             ),
#             ak._v2.contents.recordarray.RecordArray(
#                 [
#                     ak._v2.contents.numpyarray.NumpyArray(
#                         np.array([1.1, 2.2, 3.3, 4.4, 5.5])
#                     )
#                 ],
#                 ["nest"],
#             ),
#         ],
#     )
#
#     assert ak.to_list(v2_to_v1(v2_array.sort(-1))) == [0, 1, 2, 3, 4, 5, 6]
