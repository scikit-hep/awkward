# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.forms import ListOffsetForm, NumpyForm


def test_ListOffsetArray():
    a = ak._v2.Array([[0.0, 1.1, 2.2], [], [3.3, 4.4]])
    b = ak._v2.Array([[5.5], [6.6, 7.7, 8.8, 9.9]])
    c = ak._v2.concatenate([a, b])
    ctt = ak._v2.concatenate([a.layout.typetracer, b.layout.typetracer])
    assert c.layout.form == ListOffsetForm("i64", NumpyForm("float64"))
    assert c.layout.form == ctt.layout.form


def test_ListOffsetArray_ListOffsetArray():
    a = ak._v2.Array([[[0.0, 1.1, 2.2], []], [[3.3, 4.4]]])
    b = ak._v2.Array([[[5.5], [6.6, 7.7, 8.8, 9.9]]])
    c = ak._v2.concatenate([a, b])
    ctt = ak._v2.concatenate([a.layout.typetracer, b.layout.typetracer])
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


# def test_BitMaskedArray():
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
