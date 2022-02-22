# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

ROOT = pytest.importorskip("ROOT")

import awkward._v2._lookup  # noqa: E402
import awkward._v2._connect.cling  # noqa: E402


compiler = ROOT.gInterpreter.Declare


def debug_compiler(code):
    print(code)
    ROOT.gInterpreter.Declare(code)


def test_NumpyArray():
    v2a = ak._v2.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3]),
        parameters={"some": "stuff", "other": [1, 2, "three"]},
    )

    layout = v2a
    generator = ak._v2._connect.cling.togenerator(layout.form)
    lookup = ak._v2._lookup.Lookup(layout)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_NumpyArray_v2a(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.entry()};
  out[0] = obj.size();
  out[1] = obj[1];
  out[2] = obj.at(3);
}}
"""
    )
    out = np.zeros(3, dtype=np.float64)
    ROOT.roottest_NumpyArray_v2a(out, len(layout), lookup.arrayptrs)
    assert out.tolist() == [4.0, 1.1, 3.3]


def test_EmptyArray():
    v2a = ak._v2.contents.emptyarray.EmptyArray().toNumpyArray(np.dtype(np.float64))

    layout = v2a
    generator = ak._v2._connect.cling.togenerator(layout.form)
    lookup = ak._v2._lookup.Lookup(layout)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
size_t roottest_EmptyArray_v2a(ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.entry()};
  return obj.size();
}}
"""
    )
    assert ROOT.roottest_EmptyArray_v2a(len(layout), lookup.arrayptrs) == 0


def test_NumpyArray_shape():
    v2a = ak._v2.contents.numpyarray.NumpyArray(
        np.arange(2 * 3 * 5, dtype=np.int64).reshape(2, 3, 5)
    )

    layout = v2a
    generator = ak._v2._connect.cling.togenerator(layout.form)
    lookup = ak._v2._lookup.Lookup(layout)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_NumpyArray_shape_v2a(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.entry()};
  out[0] = obj.size();
  out[1] = obj[0].size();
  out[2] = obj[0][0].size();
  out[3] = obj[0][0][0];
  out[4] = obj[0][0][1];
  out[5] = obj[0][1][0];
  out[6] = obj[0][1][1];
  out[7] = obj[1][0][0];
  out[8] = obj[1][1][1];
}}
"""
    )
    out = np.zeros(9, dtype=np.float64)
    ROOT.roottest_NumpyArray_shape_v2a(out, len(layout), lookup.arrayptrs)
    assert out.tolist() == [2.0, 3.0, 5.0, 0.0, 1.0, 5.0, 6.0, 15.0, 21.0]


def test_RegularArray_NumpyArray():
    v2a = ak._v2.contents.regulararray.RegularArray(
        ak._v2.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        3,
    )

    layout = v2a
    generator = ak._v2._connect.cling.togenerator(layout.form)
    lookup = ak._v2._lookup.Lookup(layout)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_RegularArray_NumpyArray_v2a(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.entry()};
  out[0] = obj.size();
  out[1] = obj[0][0];
  out[2] = obj[0][1];
  out[3] = obj[1][0];
  out[4] = obj[1][1];
  out[5] = obj[1].size();
}}
"""
    )
    out = np.zeros(6, dtype=np.float64)
    ROOT.roottest_RegularArray_NumpyArray_v2a(out, len(layout), lookup.arrayptrs)
    assert out.tolist() == [2.0, 0.0, 1.1, 3.3, 4.4, 3.0]

    v2b = ak._v2.contents.regulararray.RegularArray(
        ak._v2.contents.emptyarray.EmptyArray().toNumpyArray(np.dtype(np.float64)),
        0,
        zeros_length=10,
    )

    layout = v2b
    generator = ak._v2._connect.cling.togenerator(layout.form)
    lookup = ak._v2._lookup.Lookup(layout)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_RegularArray_NumpyArray_v2b(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.entry()};
  out[0] = obj.size();
  out[1] = obj[0].size();
  out[2] = obj[1].size();
}}
"""
    )
    out = np.zeros(3, dtype=np.float64)
    ROOT.roottest_RegularArray_NumpyArray_v2b(out, len(layout), lookup.arrayptrs)
    assert out.tolist() == [10.0, 0.0, 0.0]


# def test_ListArray_NumpyArray():
#     v2a = ak._v2.contents.listarray.ListArray(
#         ak._v2.index.Index(np.array([4, 100, 1], np.int64)),
#         ak._v2.index.Index(np.array([7, 100, 3, 200], np.int64)),
#         ak._v2.contents.numpyarray.NumpyArray(
#             np.array([6.6, 4.4, 5.5, 7.7, 1.1, 2.2, 3.3, 8.8])
#         ),
#     )


# def test_ListOffsetArray_NumpyArray():
#     v2a = ak._v2.contents.listoffsetarray.ListOffsetArray(
#         ak._v2.index.Index(np.array([1, 4, 4, 6, 7], np.int64)),
#         ak._v2.contents.numpyarray.NumpyArray([6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7]),
#     )


# def test_RecordArray_NumpyArray():
#     v2a = ak._v2.contents.recordarray.RecordArray(
#         [
#             ak._v2.contents.numpyarray.NumpyArray(np.array([0, 1, 2, 3, 4], np.int64)),
#             ak._v2.contents.numpyarray.NumpyArray(
#                 np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])
#             ),
#         ],
#         ["x", "y"],
#     )

#     v2b = ak._v2.contents.recordarray.RecordArray(
#         [
#             ak._v2.contents.numpyarray.NumpyArray(np.array([0, 1, 2, 3, 4], np.int64)),
#             ak._v2.contents.numpyarray.NumpyArray(
#                 np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])
#             ),
#         ],
#         None,
#     )

#     v2c = ak._v2.contents.recordarray.RecordArray([], [], 10)

#     v2d = ak._v2.contents.recordarray.RecordArray([], None, 10)


# def test_IndexedArray_NumpyArray():
#     v2a = ak._v2.contents.indexedarray.IndexedArray(
#         ak._v2.index.Index(np.array([2, 2, 0, 1, 4, 5, 4], np.int64)),
#         ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
#     )


# def test_IndexedOptionArray_NumpyArray():
#     v2a = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
#         ak._v2.index.Index(np.array([2, 2, -1, 1, -1, 5, 4], np.int64)),
#         ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
#     )


# def test_ByteMaskedArray_NumpyArray():
#     v2a = ak._v2.contents.bytemaskedarray.ByteMaskedArray(
#         ak._v2.index.Index(np.array([1, 0, 1, 0, 1], np.int8)),
#         ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
#         valid_when=True,
#     )

#     v2b = ak._v2.contents.bytemaskedarray.ByteMaskedArray(
#         ak._v2.index.Index(np.array([0, 1, 0, 1, 0], np.int8)),
#         ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
#         valid_when=False,
#     )


# def test_BitMaskedArray_NumpyArray():
#     v2a = ak._v2.contents.bitmaskedarray.BitMaskedArray(
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
#                     np.uint8,
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

#     v2b = ak._v2.contents.bitmaskedarray.BitMaskedArray(
#         ak._v2.index.Index(
#             np.packbits(
#                 np.array(
#                     [
#                         0,
#                         0,
#                         0,
#                         0,
#                         1,
#                         1,
#                         1,
#                         1,
#                         0,
#                         1,
#                         0,
#                         1,
#                         0,
#                     ],
#                     np.uint8,
#                 )
#             )
#         ),
#         ak._v2.contents.numpyarray.NumpyArray(
#             np.array(
#                 [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
#             )
#         ),
#         valid_when=False,
#         length=13,
#         lsb_order=False,
#     )

#     v2c = ak._v2.contents.bitmaskedarray.BitMaskedArray(
#         ak._v2.index.Index(
#             np.packbits(
#                 np.array(
#                     [
#                         0,
#                         0,
#                         0,
#                         0,
#                         1,
#                         1,
#                         1,
#                         1,
#                         0,
#                         0,
#                         0,
#                         1,
#                         0,
#                         1,
#                         0,
#                         1,
#                     ],
#                     np.uint8,
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
#         lsb_order=True,
#     )

#     v2d = ak._v2.contents.bitmaskedarray.BitMaskedArray(
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
#                         0,
#                         0,
#                         0,
#                         0,
#                         1,
#                         0,
#                         1,
#                         0,
#                     ],
#                     np.uint8,
#                 )
#             )
#         ),
#         ak._v2.contents.numpyarray.NumpyArray(
#             np.array(
#                 [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
#             )
#         ),
#         valid_when=False,
#         length=13,
#         lsb_order=True,
#     )


# def test_UnmaskedArray_NumpyArray():
#     v2a = ak._v2.contents.unmaskedarray.UnmaskedArray(
#         ak._v2.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3]))
#     )


# def test_UnionArray_NumpyArray():
#     v2a = ak._v2.contents.unionarray.UnionArray(
#         ak._v2.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], np.int8)),
#         ak._v2.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100], np.int64)),
#         [
#             ak._v2.contents.numpyarray.NumpyArray(np.array([1, 2, 3], np.int64)),
#             ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])),
#         ],
#     )


# def test_RegularArray_RecordArray_NumpyArray():
#     v2a = ak._v2.contents.regulararray.RegularArray(
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

#     v2b = ak._v2.contents.regulararray.RegularArray(
#         ak._v2.contents.recordarray.RecordArray(
#             [
#                 ak._v2.contents.emptyarray.EmptyArray().toNumpyArray(
#                     np.dtype(np.float64)
#                 )
#             ],
#             ["nest"],
#         ),
#         0,
#         zeros_length=10,
#     )


# def test_ListArray_RecordArray_NumpyArray():
#     v2a = ak._v2.contents.listarray.ListArray(
#         ak._v2.index.Index(np.array([4, 100, 1], np.int64)),
#         ak._v2.index.Index(np.array([7, 100, 3, 200], np.int64)),
#         ak._v2.contents.recordarray.RecordArray(
#             [
#                 ak._v2.contents.numpyarray.NumpyArray(
#                     np.array([6.6, 4.4, 5.5, 7.7, 1.1, 2.2, 3.3, 8.8])
#                 )
#             ],
#             ["nest"],
#         ),
#     )


# def test_ListOffsetArray_RecordArray_NumpyArray():
#     v2a = ak._v2.contents.listoffsetarray.ListOffsetArray(
#         ak._v2.index.Index(np.array([1, 4, 4, 6], np.int64)),
#         ak._v2.contents.recordarray.RecordArray(
#             [
#                 ak._v2.contents.numpyarray.NumpyArray(
#                     [6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7]
#                 )
#             ],
#             ["nest"],
#         ),
#     )


# def test_IndexedArray_RecordArray_NumpyArray():
#     v2a = ak._v2.contents.indexedarray.IndexedArray(
#         ak._v2.index.Index(np.array([2, 2, 0, 1, 4, 5, 4], np.int64)),
#         ak._v2.contents.recordarray.RecordArray(
#             [
#                 ak._v2.contents.numpyarray.NumpyArray(
#                     np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
#                 )
#             ],
#             ["nest"],
#         ),
#     )


# def test_IndexedOptionArray_RecordArray_NumpyArray():
#     v2a = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
#         ak._v2.index.Index(np.array([2, 2, -1, 1, -1, 5, 4], np.int64)),
#         ak._v2.contents.recordarray.RecordArray(
#             [
#                 ak._v2.contents.numpyarray.NumpyArray(
#                     np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
#                 )
#             ],
#             ["nest"],
#         ),
#     )


# def test_ByteMaskedArray_RecordArray_NumpyArray():
#     v2a = ak._v2.contents.bytemaskedarray.ByteMaskedArray(
#         ak._v2.index.Index(np.array([1, 0, 1, 0, 1], np.int8)),
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

#     v2b = ak._v2.contents.bytemaskedarray.ByteMaskedArray(
#         ak._v2.index.Index(np.array([0, 1, 0, 1, 0], np.int8)),
#         ak._v2.contents.recordarray.RecordArray(
#             [
#                 ak._v2.contents.numpyarray.NumpyArray(
#                     np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
#                 )
#             ],
#             ["nest"],
#         ),
#         valid_when=False,
#     )


# def test_BitMaskedArray_RecordArray_NumpyArray():
#     v2a = ak._v2.contents.bitmaskedarray.BitMaskedArray(
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

#     v2b = ak._v2.contents.bitmaskedarray.BitMaskedArray(
#         ak._v2.index.Index(
#             np.packbits(
#                 np.array(
#                     [
#                         0,
#                         0,
#                         0,
#                         0,
#                         1,
#                         1,
#                         1,
#                         1,
#                         0,
#                         1,
#                         0,
#                         1,
#                         0,
#                     ],
#                     np.uint8,
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
#         valid_when=False,
#         length=13,
#         lsb_order=False,
#     )

#     v2c = ak._v2.contents.bitmaskedarray.BitMaskedArray(
#         ak._v2.index.Index(
#             np.packbits(
#                 np.array(
#                     [
#                         0,
#                         0,
#                         0,
#                         0,
#                         1,
#                         1,
#                         1,
#                         1,
#                         0,
#                         0,
#                         0,
#                         1,
#                         0,
#                         1,
#                         0,
#                         1,
#                     ],
#                     np.uint8,
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
#         lsb_order=True,
#     )

#     v2d = ak._v2.contents.bitmaskedarray.BitMaskedArray(
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
#                         0,
#                         0,
#                         0,
#                         0,
#                         1,
#                         0,
#                         1,
#                         0,
#                     ],
#                     np.uint8,
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
#         valid_when=False,
#         length=13,
#         lsb_order=True,
#     )


# def test_UnmaskedArray_RecordArray_NumpyArray():
#     v2a = ak._v2.contents.unmaskedarray.UnmaskedArray(
#         ak._v2.contents.recordarray.RecordArray(
#             [ak._v2.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3]))],
#             ["nest"],
#         )
#     )


# def test_UnionArray_RecordArray_NumpyArray():
#     v2a = ak._v2.contents.unionarray.UnionArray(
#         ak._v2.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], np.int8)),
#         ak._v2.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100], np.int64)),
#         [
#             ak._v2.contents.recordarray.RecordArray(
#                 [ak._v2.contents.numpyarray.NumpyArray(np.array([1, 2, 3], np.int64))],
#                 ["nest"],
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
