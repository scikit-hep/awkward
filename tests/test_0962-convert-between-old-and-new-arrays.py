# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import json

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1v2_equal, v1_to_v2, v2_to_v1

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def newform(oldform):
    if isinstance(oldform, dict):
        out = {}
        for k, v in oldform.items():
            if k == "has_identities":
                out["has_identifier"] = newform(v)
            elif k == "itemsize":
                pass
            elif k == "format":
                pass
            elif k == "class":
                if v == "ListArray32":
                    out[k] = "ListArray"
                elif v == "ListArrayU32":
                    out[k] = "ListArray"
                elif v == "ListArray64":
                    out[k] = "ListArray"
                elif v == "ListOffsetArray32":
                    out[k] = "ListOffsetArray"
                elif v == "ListOffsetArrayU32":
                    out[k] = "ListOffsetArray"
                elif v == "ListOffsetArray64":
                    out[k] = "ListOffsetArray"
                elif v == "IndexedArray32":
                    out[k] = "IndexedArray"
                elif v == "IndexedArrayU32":
                    out[k] = "IndexedArray"
                elif v == "IndexedArray64":
                    out[k] = "IndexedArray"
                elif v == "IndexedOptionArray32":
                    out[k] = "IndexedOptionArray"
                elif v == "IndexedOptionArray64":
                    out[k] = "IndexedOptionArray"
                elif v == "UnionArray8_32":
                    out[k] = "UnionArray"
                elif v == "UnionArray8_U32":
                    out[k] = "UnionArray"
                elif v == "UnionArray8_64":
                    out[k] = "UnionArray"
                else:
                    out[k] = v
            else:
                out[k] = newform(v)
        return out
    elif isinstance(oldform, list):
        return [newform(x) for x in oldform]
    else:
        return oldform


def test_EmptyArray():
    v1a = ak.layout.EmptyArray()
    v2a = ak._v2.contents.emptyarray.EmptyArray()
    assert v1v2_equal(v1a, v2a)
    assert v1v2_equal(v2_to_v1(v2a), v1_to_v2(v1a))
    assert ak.to_list(v1a) == ak.to_list(v2a)
    assert newform(json.loads(v1a.form.tojson())) == v2a.form.tolist()


def test_NumpyArray():
    v1a = ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3]))
    v2a = ak._v2.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3]))
    assert v1v2_equal(v1a, v2a)
    assert v1v2_equal(v2_to_v1(v2a), v1_to_v2(v1a))
    assert ak.to_list(v1a) == ak.to_list(v2a)
    assert newform(json.loads(v1a.form.tojson())) == v2a.form.tolist()

    v1b = ak.layout.NumpyArray(np.arange(2 * 3 * 5, dtype=np.int64).reshape(2, 3, 5))
    v2b = ak._v2.contents.numpyarray.NumpyArray(
        np.arange(2 * 3 * 5, dtype=np.int64).reshape(2, 3, 5)
    )
    assert v1v2_equal(v1b, v2b)
    assert v1v2_equal(v2_to_v1(v2b), v1_to_v2(v1b))
    assert ak.to_list(v1b) == ak.to_list(v2b)
    assert newform(json.loads(v1b.form.tojson())) == v2b.form.tolist()


def test_RegularArray_NumpyArray():
    v1a = ak.layout.RegularArray(
        ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        3,
    )
    v2a = ak._v2.contents.regulararray.RegularArray(
        ak._v2.contents.numpyarray.NumpyArray(
            np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
        ),
        3,
    )
    assert v1v2_equal(v1a, v2a)
    assert v1v2_equal(v2_to_v1(v2a), v1_to_v2(v1a))
    assert ak.to_list(v1a) == ak.to_list(v2a)
    assert newform(json.loads(v1a.form.tojson())) == v2a.form.tolist()

    v1b = ak.layout.RegularArray(ak.layout.EmptyArray(), 0, zeros_length=10)
    v2b = ak._v2.contents.regulararray.RegularArray(
        ak._v2.contents.emptyarray.EmptyArray(), 0, zeros_length=10
    )
    assert v1v2_equal(v1b, v2b)
    assert v1v2_equal(v2_to_v1(v2b), v1_to_v2(v1b))
    assert ak.to_list(v1b) == ak.to_list(v2b)
    assert newform(json.loads(v1b.form.tojson())) == v2b.form.tolist()


def test_ListArray_NumpyArray():
    v1a = ak.layout.ListArray64(
        ak.layout.Index64(np.array([4, 100, 1], dtype=np.int64)),
        ak.layout.Index64(np.array([7, 100, 3, 200], dtype=np.int64)),
        ak.layout.NumpyArray(np.array([6.6, 4.4, 5.5, 7.7, 1.1, 2.2, 3.3, 8.8])),
    )
    v2a = ak._v2.contents.listarray.ListArray(
        ak._v2.index.Index(np.array([4, 100, 1], dtype=np.int64)),
        ak._v2.index.Index(np.array([7, 100, 3, 200], dtype=np.int64)),
        ak._v2.contents.numpyarray.NumpyArray(
            np.array([6.6, 4.4, 5.5, 7.7, 1.1, 2.2, 3.3, 8.8])
        ),
    )
    assert v1v2_equal(v1a, v2a)
    assert v1v2_equal(v2_to_v1(v2a), v1_to_v2(v1a))
    assert ak.to_list(v1a) == ak.to_list(v2a)
    assert newform(json.loads(v1a.form.tojson())) == v2a.form.tolist()


def test_ListOffsetArray_NumpyArray():
    v1a = ak.layout.ListOffsetArray64(
        ak.layout.Index64(np.array([1, 4, 4, 6], dtype=np.int64)),
        ak.layout.NumpyArray([6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7]),
    )
    v2a = ak._v2.contents.listoffsetarray.ListOffsetArray(
        ak._v2.index.Index(np.array([1, 4, 4, 6], dtype=np.int64)),
        ak._v2.contents.numpyarray.NumpyArray([6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7]),
    )
    assert v1v2_equal(v1a, v2a)
    assert v1v2_equal(v2_to_v1(v2a), v1_to_v2(v1a))
    assert ak.to_list(v1a) == ak.to_list(v2a)
    assert newform(json.loads(v1a.form.tojson())) == v2a.form.tolist()


def test_RecordArray_NumpyArray():
    v1a = ak.layout.RecordArray(
        [
            ak.layout.NumpyArray(np.array([0, 1, 2, 3, 4], dtype=np.int64)),
            ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
        ["x", "y"],
    )
    v2a = ak._v2.contents.recordarray.RecordArray(
        [
            ak._v2.contents.numpyarray.NumpyArray(
                np.array([0, 1, 2, 3, 4], dtype=np.int64)
            ),
            ak._v2.contents.numpyarray.NumpyArray(
                np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])
            ),
        ],
        ["x", "y"],
    )
    assert v1v2_equal(v1a, v2a)
    assert v1v2_equal(v2_to_v1(v2a), v1_to_v2(v1a))
    assert ak.to_list(v1a) == ak.to_list(v2a)
    assert newform(json.loads(v1a.form.tojson())) == v2a.form.tolist()

    v1b = ak.layout.RecordArray(
        [
            ak.layout.NumpyArray(np.array([0, 1, 2, 3, 4], dtype=np.int64)),
            ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
        None,
    )
    v2b = ak._v2.contents.recordarray.RecordArray(
        [
            ak._v2.contents.numpyarray.NumpyArray(
                np.array([0, 1, 2, 3, 4], dtype=np.int64)
            ),
            ak._v2.contents.numpyarray.NumpyArray(
                np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])
            ),
        ],
        None,
    )
    assert v1v2_equal(v1b, v2b)
    assert v1v2_equal(v2_to_v1(v2b), v1_to_v2(v1b))
    assert ak.to_list(v1b) == ak.to_list(v2b)
    assert newform(json.loads(v1b.form.tojson())) == v2b.form.tolist()

    v1c = ak.layout.RecordArray([], [], 10)
    v2c = ak._v2.contents.recordarray.RecordArray([], [], 10)
    assert v1v2_equal(v1c, v2c)
    assert v1v2_equal(v2_to_v1(v2c), v1_to_v2(v1c))
    assert ak.to_list(v1c) == ak.to_list(v2c)
    assert newform(json.loads(v1c.form.tojson())) == v2c.form.tolist()

    v1d = ak.layout.RecordArray([], None, 10)
    v2d = ak._v2.contents.recordarray.RecordArray([], None, 10)
    assert v1v2_equal(v1d, v2d)
    assert v1v2_equal(v2_to_v1(v2d), v1_to_v2(v1d))
    assert ak.to_list(v1d) == ak.to_list(v2d)
    assert newform(json.loads(v1c.form.tojson())) == v2c.form.tolist()


def test_IndexedArray_NumpyArray():
    v1a = ak.layout.IndexedArray64(
        ak.layout.Index64(np.array([2, 2, 0, 1, 4, 5, 4], dtype=np.int64)),
        ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
    )
    v2a = ak._v2.contents.indexedarray.IndexedArray(
        ak._v2.index.Index(np.array([2, 2, 0, 1, 4, 5, 4], dtype=np.int64)),
        ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
    )
    assert v1v2_equal(v1a, v2a)
    assert v1v2_equal(v2_to_v1(v2a), v1_to_v2(v1a))
    assert ak.to_list(v1a) == ak.to_list(v2a)
    assert newform(json.loads(v1a.form.tojson())) == v2a.form.tolist()


def test_IndexedOptionArray_NumpyArray():
    v1a = ak.layout.IndexedOptionArray64(
        ak.layout.Index64(np.array([2, 2, -1, 1, -1, 5, 4], dtype=np.int64)),
        ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
    )
    v2a = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
        ak._v2.index.Index(np.array([2, 2, -1, 1, -1, 5, 4], dtype=np.int64)),
        ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
    )
    assert v1v2_equal(v1a, v2a)
    assert v1v2_equal(v2_to_v1(v2a), v1_to_v2(v1a))
    assert ak.to_list(v1a) == ak.to_list(v2a)
    assert newform(json.loads(v1a.form.tojson())) == v2a.form.tolist()


def test_ByteMaskedArray_NumpyArray():
    v1a = ak.layout.ByteMaskedArray(
        ak.layout.Index8(np.array([1, 0, 1, 0, 1], dtype=np.int8)),
        ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        valid_when=True,
    )
    v2a = ak._v2.contents.bytemaskedarray.ByteMaskedArray(
        ak._v2.index.Index(np.array([1, 0, 1, 0, 1], dtype=np.int8)),
        ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        valid_when=True,
    )
    assert v1v2_equal(v1a, v2a)
    assert v1v2_equal(v2_to_v1(v2a), v1_to_v2(v1a))
    assert ak.to_list(v1a) == ak.to_list(v2a)
    assert newform(json.loads(v1a.form.tojson())) == v2a.form.tolist()

    v1b = ak.layout.ByteMaskedArray(
        ak.layout.Index8(np.array([0, 1, 0, 1, 0], dtype=np.int8)),
        ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        valid_when=False,
    )
    v2b = ak._v2.contents.bytemaskedarray.ByteMaskedArray(
        ak._v2.index.Index(np.array([0, 1, 0, 1, 0], dtype=np.int8)),
        ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        valid_when=False,
    )
    assert v1v2_equal(v1b, v2b)
    assert v1v2_equal(v2_to_v1(v2b), v1_to_v2(v1b))
    assert ak.to_list(v1b) == ak.to_list(v2b)
    assert newform(json.loads(v1b.form.tojson())) == v2b.form.tolist()


def test_BitMaskedArray_NumpyArray():
    v1a = ak.layout.BitMaskedArray(
        ak.layout.IndexU8(
            np.packbits(
                np.array(
                    [
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                    ],
                    dtype=np.uint8,
                )
            )
        ),
        ak.layout.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=True,
        length=13,
        lsb_order=False,
    )
    v2a = ak._v2.contents.bitmaskedarray.BitMaskedArray(
        ak._v2.index.Index(
            np.packbits(
                np.array(
                    [
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                    ],
                    dtype=np.uint8,
                )
            )
        ),
        ak._v2.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=True,
        length=13,
        lsb_order=False,
    )
    assert v1v2_equal(v1a, v2a)
    assert v1v2_equal(v2_to_v1(v2a), v1_to_v2(v1a))
    assert ak.to_list(v1a) == ak.to_list(v2a)
    assert newform(json.loads(v1a.form.tojson())) == v2a.form.tolist()

    v1b = ak.layout.BitMaskedArray(
        ak.layout.IndexU8(
            np.packbits(
                np.array(
                    [
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        1,
                        0,
                        1,
                        0,
                    ],
                    dtype=np.uint8,
                )
            )
        ),
        ak.layout.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=False,
        length=13,
        lsb_order=False,
    )
    v2b = ak._v2.contents.bitmaskedarray.BitMaskedArray(
        ak._v2.index.Index(
            np.packbits(
                np.array(
                    [
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        1,
                        0,
                        1,
                        0,
                    ],
                    dtype=np.uint8,
                )
            )
        ),
        ak._v2.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=False,
        length=13,
        lsb_order=False,
    )
    assert v1v2_equal(v1b, v2b)
    assert v1v2_equal(v2_to_v1(v2b), v1_to_v2(v1b))
    assert ak.to_list(v1b) == ak.to_list(v2b)
    assert newform(json.loads(v1b.form.tojson())) == v2b.form.tolist()

    v1c = ak.layout.BitMaskedArray(
        ak.layout.IndexU8(
            np.packbits(
                np.array(
                    [
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                    ],
                    dtype=np.uint8,
                )
            )
        ),
        ak.layout.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=True,
        length=13,
        lsb_order=True,
    )
    v2c = ak._v2.contents.bitmaskedarray.BitMaskedArray(
        ak._v2.index.Index(
            np.packbits(
                np.array(
                    [
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                    ],
                    dtype=np.uint8,
                )
            )
        ),
        ak._v2.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=True,
        length=13,
        lsb_order=True,
    )
    assert v1v2_equal(v1c, v2c)
    assert v1v2_equal(v2_to_v1(v2c), v1_to_v2(v1c))
    assert ak.to_list(v1c) == ak.to_list(v2c)
    assert newform(json.loads(v1c.form.tojson())) == v2c.form.tolist()

    v1d = ak.layout.BitMaskedArray(
        ak.layout.IndexU8(
            np.packbits(
                np.array(
                    [
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                    ],
                    dtype=np.uint8,
                )
            )
        ),
        ak.layout.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=False,
        length=13,
        lsb_order=True,
    )
    v2d = ak._v2.contents.bitmaskedarray.BitMaskedArray(
        ak._v2.index.Index(
            np.packbits(
                np.array(
                    [
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                    ],
                    dtype=np.uint8,
                )
            )
        ),
        ak._v2.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=False,
        length=13,
        lsb_order=True,
    )
    assert v1v2_equal(v1d, v2d)
    assert v1v2_equal(v2_to_v1(v2d), v1_to_v2(v1d))
    assert ak.to_list(v1d) == ak.to_list(v2d)
    assert newform(json.loads(v1d.form.tojson())) == v2d.form.tolist()


def test_UnmaskedArray_NumpyArray():
    v1a = ak.layout.UnmaskedArray(ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3])))
    v2a = ak._v2.contents.unmaskedarray.UnmaskedArray(
        ak._v2.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3]))
    )
    assert v1v2_equal(v1a, v2a)
    assert v1v2_equal(v2_to_v1(v2a), v1_to_v2(v1a))
    assert ak.to_list(v1a) == ak.to_list(v2a)
    assert newform(json.loads(v1a.form.tojson())) == v2a.form.tolist()


def test_UnionArray_NumpyArray():
    v1a = ak.layout.UnionArray8_64(
        ak.layout.Index8(np.array([1, 1, 0, 0, 1, 0, 1], dtype=np.int8)),
        ak.layout.Index64(np.array([4, 3, 0, 1, 2, 2, 4, 100], dtype=np.int64)),
        [
            ak.layout.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
    )
    v2a = ak._v2.contents.unionarray.UnionArray(
        ak._v2.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], dtype=np.int8)),
        ak._v2.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100], dtype=np.int64)),
        [
            ak._v2.contents.numpyarray.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
    )
    assert v1v2_equal(v1a, v2a)
    assert v1v2_equal(v2_to_v1(v2a), v1_to_v2(v1a))
    assert ak.to_list(v1a) == ak.to_list(v2a)
    assert newform(json.loads(v1a.form.tojson())) == v2a.form.tolist()


def test_RegularArray_RecordArray_NumpyArray():
    v1a = ak.layout.RegularArray(
        ak.layout.RecordArray(
            [ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]))],
            ["nest"],
        ),
        3,
    )
    v2a = ak._v2.contents.regulararray.RegularArray(
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
        3,
    )
    assert v1v2_equal(v1a, v2a)
    assert v1v2_equal(v2_to_v1(v2a), v1_to_v2(v1a))
    assert ak.to_list(v1a) == ak.to_list(v2a)
    assert newform(json.loads(v1a.form.tojson())) == v2a.form.tolist()

    v1b = ak.layout.RegularArray(
        ak.layout.RecordArray([ak.layout.EmptyArray()], ["nest"]),
        0,
        zeros_length=10,
    )
    v2b = ak._v2.contents.regulararray.RegularArray(
        ak._v2.contents.recordarray.RecordArray(
            [ak._v2.contents.emptyarray.EmptyArray()], ["nest"]
        ),
        0,
        zeros_length=10,
    )
    assert v1v2_equal(v1b, v2b)
    assert v1v2_equal(v2_to_v1(v2b), v1_to_v2(v1b))
    assert ak.to_list(v1b) == ak.to_list(v2b)
    assert newform(json.loads(v1b.form.tojson())) == v2b.form.tolist()


def test_ListArray_RecordArray_NumpyArray():
    v1a = ak.layout.ListArray64(
        ak.layout.Index64(np.array([4, 100, 1], dtype=np.int64)),
        ak.layout.Index64(np.array([7, 100, 3, 200], dtype=np.int64)),
        ak.layout.RecordArray(
            [ak.layout.NumpyArray(np.array([6.6, 4.4, 5.5, 7.7, 1.1, 2.2, 3.3, 8.8]))],
            ["nest"],
        ),
    )
    v2a = ak._v2.contents.listarray.ListArray(
        ak._v2.index.Index(np.array([4, 100, 1], dtype=np.int64)),
        ak._v2.index.Index(np.array([7, 100, 3, 200], dtype=np.int64)),
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    np.array([6.6, 4.4, 5.5, 7.7, 1.1, 2.2, 3.3, 8.8])
                )
            ],
            ["nest"],
        ),
    )
    assert v1v2_equal(v1a, v2a)
    assert v1v2_equal(v2_to_v1(v2a), v1_to_v2(v1a))
    assert ak.to_list(v1a) == ak.to_list(v2a)
    assert newform(json.loads(v1a.form.tojson())) == v2a.form.tolist()


def test_ListOffsetArray_RecordArray_NumpyArray():
    v1a = ak.layout.ListOffsetArray64(
        ak.layout.Index64(np.array([1, 4, 4, 6], dtype=np.int64)),
        ak.layout.RecordArray(
            [ak.layout.NumpyArray([6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7])],
            ["nest"],
        ),
    )
    v2a = ak._v2.contents.listoffsetarray.ListOffsetArray(
        ak._v2.index.Index(np.array([1, 4, 4, 6], dtype=np.int64)),
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    [6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7]
                )
            ],
            ["nest"],
        ),
    )
    assert v1v2_equal(v1a, v2a)
    assert v1v2_equal(v2_to_v1(v2a), v1_to_v2(v1a))
    assert ak.to_list(v1a) == ak.to_list(v2a)
    assert newform(json.loads(v1a.form.tojson())) == v2a.form.tolist()


def test_IndexedArray_RecordArray_NumpyArray():
    v1a = ak.layout.IndexedArray64(
        ak.layout.Index64(np.array([2, 2, 0, 1, 4, 5, 4], dtype=np.int64)),
        ak.layout.RecordArray(
            [ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6]))],
            ["nest"],
        ),
    )
    v2a = ak._v2.contents.indexedarray.IndexedArray(
        ak._v2.index.Index(np.array([2, 2, 0, 1, 4, 5, 4], dtype=np.int64)),
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
    )
    assert v1v2_equal(v1a, v2a)
    assert v1v2_equal(v2_to_v1(v2a), v1_to_v2(v1a))
    assert ak.to_list(v1a) == ak.to_list(v2a)
    assert newform(json.loads(v1a.form.tojson())) == v2a.form.tolist()


def test_IndexedOptionArray_RecordArray_NumpyArray():
    v1a = ak.layout.IndexedOptionArray64(
        ak.layout.Index64(np.array([2, 2, -1, 1, -1, 5, 4], dtype=np.int64)),
        ak.layout.RecordArray(
            [ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6]))],
            ["nest"],
        ),
    )
    v2a = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
        ak._v2.index.Index(np.array([2, 2, -1, 1, -1, 5, 4], dtype=np.int64)),
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
    )
    assert v1v2_equal(v1a, v2a)
    assert v1v2_equal(v2_to_v1(v2a), v1_to_v2(v1a))
    assert ak.to_list(v1a) == ak.to_list(v2a)
    assert newform(json.loads(v1a.form.tojson())) == v2a.form.tolist()


def test_ByteMaskedArray_RecordArray_NumpyArray():
    v1a = ak.layout.ByteMaskedArray(
        ak.layout.Index8(np.array([1, 0, 1, 0, 1], dtype=np.int8)),
        ak.layout.RecordArray(
            [ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6]))],
            ["nest"],
        ),
        valid_when=True,
    )
    v2a = ak._v2.contents.bytemaskedarray.ByteMaskedArray(
        ak._v2.index.Index(np.array([1, 0, 1, 0, 1], dtype=np.int8)),
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
        valid_when=True,
    )
    assert v1v2_equal(v1a, v2a)
    assert v1v2_equal(v2_to_v1(v2a), v1_to_v2(v1a))
    assert ak.to_list(v1a) == ak.to_list(v2a)
    assert newform(json.loads(v1a.form.tojson())) == v2a.form.tolist()

    v1b = ak.layout.ByteMaskedArray(
        ak.layout.Index8(np.array([0, 1, 0, 1, 0], dtype=np.int8)),
        ak.layout.RecordArray(
            [ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6]))],
            ["nest"],
        ),
        valid_when=False,
    )
    v2b = ak._v2.contents.bytemaskedarray.ByteMaskedArray(
        ak._v2.index.Index(np.array([0, 1, 0, 1, 0], dtype=np.int8)),
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
        valid_when=False,
    )
    assert v1v2_equal(v1b, v2b)
    assert v1v2_equal(v2_to_v1(v2b), v1_to_v2(v1b))
    assert ak.to_list(v1b) == ak.to_list(v2b)
    assert newform(json.loads(v1b.form.tojson())) == v2b.form.tolist()


def test_BitMaskedArray_RecordArray_NumpyArray():
    v1a = ak.layout.BitMaskedArray(
        ak.layout.IndexU8(
            np.packbits(
                np.array(
                    [
                        True,
                        True,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        True,
                        False,
                        True,
                        False,
                        True,
                    ]
                )
            )
        ),
        ak.layout.RecordArray(
            [
                ak.layout.NumpyArray(
                    np.array(
                        [
                            0.0,
                            1.0,
                            2.0,
                            3.0,
                            4.0,
                            5.0,
                            6.0,
                            7.0,
                            1.1,
                            2.2,
                            3.3,
                            4.4,
                            5.5,
                            6.6,
                        ]
                    )
                )
            ],
            ["nest"],
        ),
        valid_when=True,
        length=13,
        lsb_order=False,
    )
    v2a = ak._v2.contents.bitmaskedarray.BitMaskedArray(
        ak._v2.index.Index(
            np.packbits(
                np.array(
                    [
                        True,
                        True,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        True,
                        False,
                        True,
                        False,
                        True,
                    ]
                )
            )
        ),
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    np.array(
                        [
                            0.0,
                            1.0,
                            2.0,
                            3.0,
                            4.0,
                            5.0,
                            6.0,
                            7.0,
                            1.1,
                            2.2,
                            3.3,
                            4.4,
                            5.5,
                            6.6,
                        ]
                    )
                )
            ],
            ["nest"],
        ),
        valid_when=True,
        length=13,
        lsb_order=False,
    )
    assert v1v2_equal(v1a, v2a)
    assert v1v2_equal(v2_to_v1(v2a), v1_to_v2(v1a))
    assert ak.to_list(v1a) == ak.to_list(v2a)
    assert newform(json.loads(v1a.form.tojson())) == v2a.form.tolist()

    v1b = ak.layout.BitMaskedArray(
        ak.layout.IndexU8(
            np.packbits(
                np.array(
                    [
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        1,
                        0,
                        1,
                        0,
                    ],
                    dtype=np.uint8,
                )
            )
        ),
        ak.layout.RecordArray(
            [
                ak.layout.NumpyArray(
                    np.array(
                        [
                            0.0,
                            1.0,
                            2.0,
                            3.0,
                            4.0,
                            5.0,
                            6.0,
                            7.0,
                            1.1,
                            2.2,
                            3.3,
                            4.4,
                            5.5,
                            6.6,
                        ]
                    )
                )
            ],
            ["nest"],
        ),
        valid_when=False,
        length=13,
        lsb_order=False,
    )
    v2b = ak._v2.contents.bitmaskedarray.BitMaskedArray(
        ak._v2.index.Index(
            np.packbits(
                np.array(
                    [
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        1,
                        0,
                        1,
                        0,
                    ],
                    dtype=np.uint8,
                )
            )
        ),
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    np.array(
                        [
                            0.0,
                            1.0,
                            2.0,
                            3.0,
                            4.0,
                            5.0,
                            6.0,
                            7.0,
                            1.1,
                            2.2,
                            3.3,
                            4.4,
                            5.5,
                            6.6,
                        ]
                    )
                )
            ],
            ["nest"],
        ),
        valid_when=False,
        length=13,
        lsb_order=False,
    )
    assert v1v2_equal(v1b, v2b)
    assert v1v2_equal(v2_to_v1(v2b), v1_to_v2(v1b))
    assert ak.to_list(v1b) == ak.to_list(v2b)
    assert newform(json.loads(v1b.form.tojson())) == v2b.form.tolist()

    v1c = ak.layout.BitMaskedArray(
        ak.layout.IndexU8(
            np.packbits(
                np.array(
                    [
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                    ],
                    dtype=np.uint8,
                )
            )
        ),
        ak.layout.RecordArray(
            [
                ak.layout.NumpyArray(
                    np.array(
                        [
                            0.0,
                            1.0,
                            2.0,
                            3.0,
                            4.0,
                            5.0,
                            6.0,
                            7.0,
                            1.1,
                            2.2,
                            3.3,
                            4.4,
                            5.5,
                            6.6,
                        ]
                    )
                )
            ],
            ["nest"],
        ),
        valid_when=True,
        length=13,
        lsb_order=True,
    )
    v2c = ak._v2.contents.bitmaskedarray.BitMaskedArray(
        ak._v2.index.Index(
            np.packbits(
                np.array(
                    [
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                    ],
                    dtype=np.uint8,
                )
            )
        ),
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    np.array(
                        [
                            0.0,
                            1.0,
                            2.0,
                            3.0,
                            4.0,
                            5.0,
                            6.0,
                            7.0,
                            1.1,
                            2.2,
                            3.3,
                            4.4,
                            5.5,
                            6.6,
                        ]
                    )
                )
            ],
            ["nest"],
        ),
        valid_when=True,
        length=13,
        lsb_order=True,
    )
    assert v1v2_equal(v1c, v2c)
    assert v1v2_equal(v2_to_v1(v2c), v1_to_v2(v1c))
    assert ak.to_list(v1c) == ak.to_list(v2c)
    assert newform(json.loads(v1c.form.tojson())) == v2c.form.tolist()

    v1d = ak.layout.BitMaskedArray(
        ak.layout.IndexU8(
            np.packbits(
                np.array(
                    [
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                    ],
                    dtype=np.uint8,
                )
            )
        ),
        ak.layout.RecordArray(
            [
                ak.layout.NumpyArray(
                    np.array(
                        [
                            0.0,
                            1.0,
                            2.0,
                            3.0,
                            4.0,
                            5.0,
                            6.0,
                            7.0,
                            1.1,
                            2.2,
                            3.3,
                            4.4,
                            5.5,
                            6.6,
                        ]
                    )
                )
            ],
            ["nest"],
        ),
        valid_when=False,
        length=13,
        lsb_order=True,
    )
    v2d = ak._v2.contents.bitmaskedarray.BitMaskedArray(
        ak._v2.index.Index(
            np.packbits(
                np.array(
                    [
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                    ],
                    dtype=np.uint8,
                )
            )
        ),
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    np.array(
                        [
                            0.0,
                            1.0,
                            2.0,
                            3.0,
                            4.0,
                            5.0,
                            6.0,
                            7.0,
                            1.1,
                            2.2,
                            3.3,
                            4.4,
                            5.5,
                            6.6,
                        ]
                    )
                )
            ],
            ["nest"],
        ),
        valid_when=False,
        length=13,
        lsb_order=True,
    )
    assert v1v2_equal(v1d, v2d)
    assert v1v2_equal(v2_to_v1(v2d), v1_to_v2(v1d))
    assert ak.to_list(v1d) == ak.to_list(v2d)
    assert newform(json.loads(v1d.form.tojson())) == v2d.form.tolist()


def test_UnmaskedArray_RecordArray_NumpyArray():
    v1a = ak.layout.UnmaskedArray(
        ak.layout.RecordArray(
            [ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3]))],
            ["nest"],
        )
    )
    v2a = ak._v2.contents.unmaskedarray.UnmaskedArray(
        ak._v2.contents.recordarray.RecordArray(
            [ak._v2.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3]))],
            ["nest"],
        )
    )
    assert v1v2_equal(v1a, v2a)
    assert v1v2_equal(v2_to_v1(v2a), v1_to_v2(v1a))
    assert ak.to_list(v1a) == ak.to_list(v2a)
    assert newform(json.loads(v1a.form.tojson())) == v2a.form.tolist()


def test_UnionArray_RecordArray_NumpyArray():
    v1a = ak.layout.UnionArray8_64(
        ak.layout.Index8(np.array([1, 1, 0, 0, 1, 0, 1], dtype=np.int8)),
        ak.layout.Index64(np.array([4, 3, 0, 1, 2, 2, 4, 100], dtype=np.int64)),
        [
            ak.layout.RecordArray(
                [ak.layout.NumpyArray(np.array([1, 2, 3], dtype=np.int64))], ["nest"]
            ),
            ak.layout.RecordArray(
                [ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))],
                ["nest"],
            ),
        ],
    )
    v2a = ak._v2.contents.unionarray.UnionArray(
        ak._v2.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], dtype=np.int8)),
        ak._v2.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100], dtype=np.int64)),
        [
            ak._v2.contents.recordarray.RecordArray(
                [
                    ak._v2.contents.numpyarray.NumpyArray(
                        np.array([1, 2, 3], dtype=np.int64)
                    )
                ],
                ["nest"],
            ),
            ak._v2.contents.recordarray.RecordArray(
                [
                    ak._v2.contents.numpyarray.NumpyArray(
                        np.array([1.1, 2.2, 3.3, 4.4, 5.5])
                    )
                ],
                ["nest"],
            ),
        ],
    )
    assert v1v2_equal(v1a, v2a)
    assert v1v2_equal(v2_to_v1(v2a), v1_to_v2(v1a))
    assert ak.to_list(v1a) == ak.to_list(v2a)
    assert newform(json.loads(v1a.form.tojson())) == v2a.form.tolist()
