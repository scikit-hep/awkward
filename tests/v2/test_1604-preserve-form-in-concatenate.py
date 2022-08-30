# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.forms import (
    ListOffsetForm,
    ByteMaskedForm,
    BitMaskedForm,
    UnmaskedForm,
    IndexedForm,
    NumpyForm,
    EmptyForm,
)


def test_ListOffsetArray():
    a = ak._v2.Array([[0.0, 1.1, 2.2], [], [3.3, 4.4]])
    b = ak._v2.Array([[5.5], [6.6, 7.7, 8.8, 9.9]])
    c = ak._v2.concatenate([a, b])
    ctt = ak._v2.concatenate([a.layout.typetracer, b.layout.typetracer])
    assert c.tolist() == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]
    assert c.layout.form == ListOffsetForm("i64", NumpyForm("float64"))
    assert c.layout.form == ctt.layout.form


def test_ListOffsetArray_ListOffsetArray():
    a = ak._v2.Array([[[0.0, 1.1, 2.2], []], [[3.3, 4.4]]])
    b = ak._v2.Array([[[5.5], [6.6, 7.7, 8.8, 9.9]]])
    c = ak._v2.concatenate([a, b])
    ctt = ak._v2.concatenate([a.layout.typetracer, b.layout.typetracer])
    assert c.tolist() == [
        [[0.0, 1.1, 2.2], []],
        [[3.3, 4.4]],
        [[5.5], [6.6, 7.7, 8.8, 9.9]],
    ]
    assert c.layout.form == ListOffsetForm(
        "i64", ListOffsetForm("i64", NumpyForm("float64"))
    )
    assert c.layout.form == ctt.layout.form


def test_OptionType_transformations():
    expected = [1, 2, None, 4, 5, 6, 7, 8, 9, None, None, None, 123]
    indexedoptionarray = ak._v2.from_iter(expected, highlevel=False)

    assert isinstance(indexedoptionarray, ak._v2.contents.IndexedOptionArray)

    for valid_when in [False, True]:
        bytemaskedarray = indexedoptionarray.toByteMaskedArray(valid_when)
        assert isinstance(bytemaskedarray, ak._v2.contents.ByteMaskedArray)
        assert bytemaskedarray.valid_when is valid_when
        assert ak._v2.Array(bytemaskedarray).tolist() == expected

    for valid_when in [False, True]:
        for lsb_order in [False, True]:
            bitmaskedarray = indexedoptionarray.toBitMaskedArray(valid_when, lsb_order)
            assert isinstance(bitmaskedarray, ak._v2.contents.BitMaskedArray)
            assert bitmaskedarray.valid_when is valid_when
            assert bitmaskedarray.lsb_order is lsb_order
            assert ak._v2.Array(bitmaskedarray).tolist() == expected

    unmaskedarray = ak._v2.contents.UnmaskedArray(
        ak._v2.contents.NumpyArray(np.arange(13))
    )

    for valid_when in [False, True]:
        bytemaskedarray = unmaskedarray.toByteMaskedArray(valid_when)
        assert isinstance(bytemaskedarray, ak._v2.contents.ByteMaskedArray)
        assert bytemaskedarray.valid_when is valid_when
        assert ak._v2.Array(bytemaskedarray).tolist() == list(range(13))

    for valid_when in [False, True]:
        for lsb_order in [False, True]:
            bitmaskedarray = unmaskedarray.toBitMaskedArray(valid_when, lsb_order)
            assert isinstance(bitmaskedarray, ak._v2.contents.BitMaskedArray)
            assert bitmaskedarray.valid_when is valid_when
            assert bitmaskedarray.lsb_order is lsb_order
            assert ak._v2.Array(bitmaskedarray).tolist() == list(range(13))


def test_ByteMaskedArray():
    a = ak._v2.contents.bytemaskedarray.ByteMaskedArray(
        ak._v2.index.Index(np.array([1, 0, 1, 0, 1], np.int8)),
        ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        valid_when=True,
    )
    b = ak._v2.contents.bytemaskedarray.ByteMaskedArray(
        ak._v2.index.Index(np.array([1, 1, 0], np.int8)),
        ak._v2.contents.numpyarray.NumpyArray(np.array([7.7, 8.8, 9.9])),
        valid_when=True,
    )
    c = ak._v2.concatenate([a, b])
    ctt = ak._v2.concatenate([a.typetracer, b.typetracer])

    assert c.tolist() == [1.1, None, 3.3, None, 5.5, 7.7, 8.8, None]
    assert isinstance(c.layout, ak._v2.contents.ByteMaskedArray)
    assert c.layout.valid_when
    assert c.layout.form == ByteMaskedForm("i8", NumpyForm("float64"), True)
    assert c.layout.form == ctt.layout.form


def test_BitMaskedArray():
    a = ak._v2.contents.bitmaskedarray.BitMaskedArray(
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
                    np.uint8,
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
    c = ak._v2.concatenate([a, a])
    ctt = ak._v2.concatenate([a.typetracer, a.typetracer])

    assert c.tolist() == [
        0.0,
        1.0,
        2.0,
        3.0,
        None,
        None,
        None,
        None,
        1.1,
        None,
        3.3,
        None,
        5.5,
        0.0,
        1.0,
        2.0,
        3.0,
        None,
        None,
        None,
        None,
        1.1,
        None,
        3.3,
        None,
        5.5,
    ]
    assert isinstance(c.layout, ak._v2.contents.BitMaskedArray)
    assert c.layout.valid_when
    assert not c.layout.lsb_order
    assert c.layout.form == BitMaskedForm("u8", NumpyForm("float64"), True, False)
    assert c.layout.form == ctt.layout.form


def test_UnmaskedArray():
    a = ak._v2.contents.unmaskedarray.UnmaskedArray(
        ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
    )
    b = ak._v2.contents.unmaskedarray.UnmaskedArray(
        ak._v2.contents.numpyarray.NumpyArray(np.array([7.7, 8.8, 9.9])),
    )
    c = ak._v2.concatenate([a, b])
    ctt = ak._v2.concatenate([a.typetracer, b.typetracer])

    assert c.tolist() == [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
    assert isinstance(c.layout, ak._v2.contents.UnmaskedArray)
    assert c.layout.form == UnmaskedForm(NumpyForm("float64"))
    assert c.layout.form == ctt.layout.form


def test_EmptyArray():
    a = ak._v2.contents.emptyarray.EmptyArray()
    c = ak._v2.concatenate([a, a])
    ctt = ak._v2.concatenate([a.typetracer, a.typetracer])

    assert c.tolist() == []
    assert isinstance(c.layout, ak._v2.contents.EmptyArray)
    assert c.layout.form == EmptyForm()
    assert c.layout.form == ctt.layout.form


def test_IndexedArray():
    a = ak._v2.contents.indexedarray.IndexedArray(
        ak._v2.index.Index(np.array([5, 4, 3, 2, 1, 0], np.int64)),
        ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
    )
    b = ak._v2.contents.indexedarray.IndexedArray(
        ak._v2.index.Index(np.array([0, 1, 2], np.int64)),
        ak._v2.contents.numpyarray.NumpyArray(np.array([7.7, 8.8, 9.9])),
    )
    c = ak._v2.concatenate([a, b])
    ctt = ak._v2.concatenate([a.typetracer, b.typetracer])

    assert c.tolist() == [6.6, 5.5, 4.4, 3.3, 2.2, 1.1, 7.7, 8.8, 9.9]
    assert isinstance(c.layout, ak._v2.contents.IndexedArray)
    assert c.layout.form == IndexedForm("i64", NumpyForm("float64"))
    assert c.layout.form == ctt.layout.form
