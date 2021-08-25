# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2, v2_to_v1


def test_numpyarray_sort1():
    v1_array = ak.from_numpy(
        np.array([3.3, 2.2, 1.1, 5.5, 4.4]), regulararray=True, highlevel=False
    )
    v2_array = v1_to_v2(v1_array)
    assert ak.to_list(np.sort(v2_array, -1)) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]


def test_listoffsetarray_sort1():
    v1_array = ak.from_iter(
        [[3.3, 2.2, 1.1], [], [5.5, 4.4], [6.6], [9.9, 7.7, 8.8, 10.1]], highlevel=False
    )
    v2_array = v1_to_v2(v1_array)
    assert ak.to_list(v2_to_v1(v2_array.sort(-1))) == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
        [6.6],
        [7.7, 8.8, 9.9, 10.10],
    ]


# def test_regulararray_sort():
#     v1_array = ak.from_numpy(
#         np.arange(2 * 3 * 5).reshape(2, 3, 5), regulararray=True, highlevel=False
#     )
#     v2_array = v1_to_v2(v1_array)
#     assert ak.to_list(v2_to_v1(v2_array.sort(-1))) == [
#         [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
#         [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
#     ]
#
#
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
# def test_numpyarray_sort():
#     v2_array = ak._v2.contents.numpyarray.NumpyArray(
#         np.array([0.0, 1.1, 2.2, 3.3], dtype=np.float64)
#     )
#     assert ak.to_list(v2_to_v1(v2_array.sort(axis=-1))) == [0, 1, 2, 3]
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
#
#
# def test_unionarray_sort():
#     v2_array = ak._v2.contents.unionarray.UnionArray(
#         ak._v2.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], dtype=np.int8)),
#         ak._v2.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100])),
#         [
#             ak._v2.contents.numpyarray.NumpyArray(np.array([1, 2, 3])),
#             ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])),
#         ],
#     )
#     assert ak.to_list(v2_to_v1(v2_array.sort(-1))) == [0, 1, 2, 3, 4, 5, 6]
#
#
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
# def test_listarray_sort():
#     v2_array = ak._v2.contents.listarray.ListArray(  # noqa: F841
#         ak._v2.index.Index(np.array([4, 100, 1])),
#         ak._v2.index.Index(np.array([7, 100, 3, 200])),
#         ak._v2.contents.recordarray.RecordArray(
#             [
#                 ak._v2.contents.numpyarray.NumpyArray(
#                     np.array([6.6, 4.4, 5.5, 7.7, 1.1, 2.2, 3.3, 8.8])
#                 )
#             ],
#             ["nest"],
#         ),
#     )
#
#     assert ak.to_list(v2_to_v1(v2_array.sort(-1))) == [[0, 1, 2], [], [0, 1]]
#
#
# def test_listoffsetarray_sort():
#     v2_array = ak._v2.contents.listoffsetarray.ListOffsetArray(  # noqa: F841
#         ak._v2.index.Index(np.array([1, 4, 4, 6])),
#         ak._v2.contents.recordarray.RecordArray(
#             [
#                 ak._v2.contents.numpyarray.NumpyArray(
#                     [6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7]
#                 )
#             ],
#             ["nest"],
#         ),
#     )
#     assert ak.to_list(v2_to_v1(v2_array.sort(-1))) == [[0, 1, 2], [], [0, 1]]
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
