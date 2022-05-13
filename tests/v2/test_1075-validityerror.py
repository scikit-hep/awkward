# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_ListOffsetArray():
    v2_array = ak._v2.highlevel.Array(
        [[0.0, 1.1, 2.2, 3.3], [], [4.4, 5.5, 6.6], [7.7], [8.8, 9.9, 10.0, 11.1, 12.2]]
    ).layout

    assert v2_array.validity_error() == ""
    assert v2_array.typetracer.validity_error() == ""

    v2_array = ak._v2.highlevel.Array(
        [
            [[0.0, 1.1, 2.2, 3.3], [], [4.4, 5.5, 6.6]],
            [],
            [[7.7], [8.8, 9.9, 10.0, 11.1, 12.2]],
        ]
    ).layout

    assert v2_array.validity_error() == ""
    assert v2_array.typetracer.validity_error() == ""


def test_RegularArray():
    v2_array = ak._v2.highlevel.Array(
        np.array([[0.0, 1.1, 2.2, 3.3], [4.4, 5.5, 6.6, 7.7]])
    ).layout

    assert v2_array.validity_error() == ""
    assert v2_array.typetracer.validity_error() == ""


def test_NumpyArray():
    v2_array = ak._v2.highlevel.Array([0.0, 1.1, 2.2, 3.3]).layout

    assert v2_array.validity_error() == ""
    assert v2_array.typetracer.validity_error() == ""


def test_IndexedArray():
    v2_array = ak._v2.highlevel.Array(
        [
            [0.0, 1.1, 2.2, 3.3],
            [],
            [4.4, 5.5, 6.6],
            None,
            [7.7],
            None,
            [8.8, 9.9, 10.0, 11.1, 12.2],
        ]
    ).layout

    assert v2_array.validity_error() == ""
    assert v2_array.typetracer.validity_error() == ""


def test_ByteMaskedArray():
    content = ak._v2.operations.from_iter(
        [[[0, 1, 2], [], [3, 4]], [], [[5]], [[6, 7, 8, 9]], [[], [10, 11, 12]]],
        highlevel=False,
    )
    mask = ak._v2.index.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    v2_array = ak._v2.contents.ByteMaskedArray(mask, content, valid_when=False)

    assert v2_array.validity_error() == ""
    assert v2_array.typetracer.validity_error() == ""


def test_IndexedOptionArray():
    content = ak._v2.operations.from_iter(
        [[[0, 1, 2], [], [3, 4]], [], [[5]], [[6, 7, 8, 9]], [[], [10, 11, 12]]],
        highlevel=False,
    )
    index = ak._v2.index.Index64(np.array([0, 1, -1, -1, 4], dtype=np.int64))
    v2_array = ak._v2.contents.IndexedOptionArray(index, content)

    assert v2_array.validity_error() == ""
    assert v2_array.typetracer.validity_error() == ""


def test_BitMaskedArray():
    v2_array = ak._v2.contents.bitmaskedarray.BitMaskedArray(
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

    assert v2_array.validity_error() == ""
    assert v2_array.typetracer.validity_error() == ""


def test_EmptyArray():
    v2_array = ak._v2.contents.emptyarray.EmptyArray()

    assert v2_array.validity_error() == ""
    assert v2_array.typetracer.validity_error() == ""


def test_RecordArray():
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

    assert v2_array.validity_error() == ""
    assert v2_array.typetracer.validity_error() == ""


def test_UnionArray():
    v2_array = ak._v2.contents.unionarray.UnionArray(  # noqa: F841
        ak._v2.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], dtype=np.int8)),
        ak._v2.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100])),
        [
            ak._v2.contents.recordarray.RecordArray(
                [ak._v2.contents.numpyarray.NumpyArray(np.array([1, 2, 3]))], ["nest"]
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

    assert v2_array.validity_error() == ""
    assert v2_array.typetracer.validity_error() == ""


def test_UnmaskedArray():
    v2_array = ak._v2.contents.unmaskedarray.UnmaskedArray(
        ak._v2.contents.numpyarray.NumpyArray(
            np.array([0.0, 1.1, 2.2, 3.3], dtype=np.float64)
        )
    )

    assert v2_array.validity_error() == ""
    assert v2_array.typetracer.validity_error() == ""
