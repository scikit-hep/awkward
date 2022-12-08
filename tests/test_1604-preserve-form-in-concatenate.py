# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak
from awkward.forms import (
    BitMaskedForm,
    ByteMaskedForm,
    EmptyForm,
    IndexedForm,
    ListOffsetForm,
    NumpyForm,
    UnmaskedForm,
)


def test_ListOffsetArray():
    a = ak.Array([[0.0, 1.1, 2.2], [], [3.3, 4.4]])
    b = ak.Array([[5.5], [6.6, 7.7, 8.8, 9.9]])
    c = ak.concatenate([a, b])
    ctt = ak.concatenate([a.layout.to_typetracer(), b.layout.to_typetracer()])
    assert c.to_list() == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]
    assert c.layout.form == ListOffsetForm("i64", NumpyForm("float64"))
    assert c.layout.form == ctt.layout.form


def test_ListOffsetArray_ListOffsetArray():
    a = ak.Array([[[0.0, 1.1, 2.2], []], [[3.3, 4.4]]])
    b = ak.Array([[[5.5], [6.6, 7.7, 8.8, 9.9]]])
    c = ak.concatenate([a, b])
    ctt = ak.concatenate([a.layout.to_typetracer(), b.layout.to_typetracer()])
    assert c.to_list() == [
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
    indexedoptionarray = ak.from_iter(expected, highlevel=False)

    assert isinstance(indexedoptionarray, ak.contents.IndexedOptionArray)

    for valid_when in [False, True]:
        bytemaskedarray = indexedoptionarray.to_ByteMaskedArray(valid_when)
        assert isinstance(bytemaskedarray, ak.contents.ByteMaskedArray)
        assert bytemaskedarray.valid_when is valid_when
        assert ak.Array(bytemaskedarray).to_list() == expected

    for valid_when in [False, True]:
        for lsb_order in [False, True]:
            bitmaskedarray = indexedoptionarray.to_BitMaskedArray(valid_when, lsb_order)
            assert isinstance(bitmaskedarray, ak.contents.BitMaskedArray)
            assert bitmaskedarray.valid_when is valid_when
            assert bitmaskedarray.lsb_order is lsb_order
            assert ak.Array(bitmaskedarray).to_list() == expected

    unmaskedarray = ak.contents.UnmaskedArray(ak.contents.NumpyArray(np.arange(13)))

    for valid_when in [False, True]:
        bytemaskedarray = unmaskedarray.to_ByteMaskedArray(valid_when)
        assert isinstance(bytemaskedarray, ak.contents.ByteMaskedArray)
        assert bytemaskedarray.valid_when is valid_when
        assert ak.Array(bytemaskedarray).to_list() == list(range(13))

    for valid_when in [False, True]:
        for lsb_order in [False, True]:
            bitmaskedarray = unmaskedarray.to_BitMaskedArray(valid_when, lsb_order)
            assert isinstance(bitmaskedarray, ak.contents.BitMaskedArray)
            assert bitmaskedarray.valid_when is valid_when
            assert bitmaskedarray.lsb_order is lsb_order
            assert ak.Array(bitmaskedarray).to_list() == list(range(13))


def test_ByteMaskedArray():
    a = ak.contents.bytemaskedarray.ByteMaskedArray(
        ak.index.Index(np.array([1, 0, 1, 0, 1], np.int8)),
        ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        valid_when=True,
    )
    b = ak.contents.bytemaskedarray.ByteMaskedArray(
        ak.index.Index(np.array([1, 1, 0], np.int8)),
        ak.contents.numpyarray.NumpyArray(np.array([7.7, 8.8, 9.9])),
        valid_when=True,
    )
    c = ak.concatenate([a, b])
    ctt = ak.concatenate([a.to_typetracer(), b.to_typetracer()])

    assert c.to_list() == [1.1, None, 3.3, None, 5.5, 7.7, 8.8, None]
    assert isinstance(c.layout, ak.contents.ByteMaskedArray)
    assert c.layout.valid_when
    assert c.layout.form == ByteMaskedForm("i8", NumpyForm("float64"), True)
    assert c.layout.form == ctt.layout.form


def test_BitMaskedArray():
    a = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
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
        ak.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=True,
        length=13,
        lsb_order=False,
    )
    c = ak.concatenate([a, a])
    ctt = ak.concatenate([a.to_typetracer(), a.to_typetracer()])

    assert c.to_list() == [
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
    assert isinstance(c.layout, ak.contents.BitMaskedArray)
    assert c.layout.valid_when
    assert not c.layout.lsb_order
    assert c.layout.form == BitMaskedForm("u8", NumpyForm("float64"), True, False)
    assert c.layout.form == ctt.layout.form


def test_UnmaskedArray():
    a = ak.contents.unmaskedarray.UnmaskedArray(
        ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
    )
    b = ak.contents.unmaskedarray.UnmaskedArray(
        ak.contents.numpyarray.NumpyArray(np.array([7.7, 8.8, 9.9])),
    )
    c = ak.concatenate([a, b])
    ctt = ak.concatenate([a.to_typetracer(), b.to_typetracer()])

    assert c.to_list() == [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
    assert isinstance(c.layout, ak.contents.UnmaskedArray)
    assert c.layout.form == UnmaskedForm(NumpyForm("float64"))
    assert c.layout.form == ctt.layout.form


def test_EmptyArray():
    a = ak.contents.emptyarray.EmptyArray()
    c = ak.concatenate([a, a])
    ctt = ak.concatenate([a.to_typetracer(), a.to_typetracer()])

    assert c.to_list() == []
    assert isinstance(c.layout, ak.contents.EmptyArray)
    assert c.layout.form == EmptyForm()
    assert c.layout.form == ctt.layout.form


def test_IndexedArray():
    a = ak.contents.indexedarray.IndexedArray(
        ak.index.Index(np.array([5, 4, 3, 2, 1, 0], np.int64)),
        ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
    )
    b = ak.contents.indexedarray.IndexedArray(
        ak.index.Index(np.array([0, 1, 2], np.int64)),
        ak.contents.numpyarray.NumpyArray(np.array([7.7, 8.8, 9.9])),
    )
    c = ak.concatenate([a, b])
    ctt = ak.concatenate([a.to_typetracer(), b.to_typetracer()])

    assert c.to_list() == [6.6, 5.5, 4.4, 3.3, 2.2, 1.1, 7.7, 8.8, 9.9]
    assert isinstance(c.layout, ak.contents.IndexedArray)
    assert c.layout.form == IndexedForm("i64", NumpyForm("float64"))
    assert c.layout.form == ctt.layout.form
