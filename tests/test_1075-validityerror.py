# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test_ListOffsetArray():
    v2_array = ak.highlevel.Array(
        [[0.0, 1.1, 2.2, 3.3], [], [4.4, 5.5, 6.6], [7.7], [8.8, 9.9, 10.0, 11.1, 12.2]]
    ).layout

    assert ak.validity_error(v2_array) == ""
    assert ak.validity_error(v2_array.to_typetracer()) == ""

    v2_array = ak.highlevel.Array(
        [
            [[0.0, 1.1, 2.2, 3.3], [], [4.4, 5.5, 6.6]],
            [],
            [[7.7], [8.8, 9.9, 10.0, 11.1, 12.2]],
        ]
    ).layout

    assert ak.validity_error(v2_array) == ""
    assert ak.validity_error(v2_array.to_typetracer()) == ""


def test_RegularArray():
    v2_array = ak.highlevel.Array(
        np.array([[0.0, 1.1, 2.2, 3.3], [4.4, 5.5, 6.6, 7.7]])
    ).layout

    assert ak.validity_error(v2_array) == ""
    assert ak.validity_error(v2_array.to_typetracer()) == ""


def test_NumpyArray():
    v2_array = ak.highlevel.Array([0.0, 1.1, 2.2, 3.3]).layout

    assert ak.validity_error(v2_array) == ""
    assert ak.validity_error(v2_array.to_typetracer()) == ""


def test_IndexedArray():
    v2_array = ak.highlevel.Array(
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

    assert ak.validity_error(v2_array) == ""
    assert ak.validity_error(v2_array.to_typetracer()) == ""


def test_ByteMaskedArray():
    content = ak.operations.from_iter(
        [[[0, 1, 2], [], [3, 4]], [], [[5]], [[6, 7, 8, 9]], [[], [10, 11, 12]]],
        highlevel=False,
    )
    mask = ak.index.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    v2_array = ak.contents.ByteMaskedArray(mask, content, valid_when=False)

    assert ak.validity_error(v2_array) == ""
    assert ak.validity_error(v2_array.to_typetracer()) == ""


def test_IndexedOptionArray():
    content = ak.operations.from_iter(
        [[[0, 1, 2], [], [3, 4]], [], [[5]], [[6, 7, 8, 9]], [[], [10, 11, 12]]],
        highlevel=False,
    )
    index = ak.index.Index64(np.array([0, 1, -1, -1, 4], dtype=np.int64))
    v2_array = ak.contents.IndexedOptionArray(index, content)

    assert ak.validity_error(v2_array) == ""
    assert ak.validity_error(v2_array.to_typetracer()) == ""


def test_BitMaskedArray():
    v2_array = ak.contents.bitmaskedarray.BitMaskedArray(
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
                    dtype=np.uint8,
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

    assert ak.validity_error(v2_array) == ""
    assert ak.validity_error(v2_array.to_typetracer()) == ""


def test_EmptyArray():
    v2_array = ak.contents.emptyarray.EmptyArray()

    assert ak.validity_error(v2_array) == ""
    assert ak.validity_error(v2_array.to_typetracer()) == ""


def test_RecordArray():
    v2_array = ak.contents.listarray.ListArray(
        ak.index.Index(np.array([4, 100, 1])),
        ak.index.Index(np.array([7, 100, 3, 200])),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array([6.6, 4.4, 5.5, 7.7, 1.1, 2.2, 3.3, 8.8])
                )
            ],
            ["nest"],
        ),
    )

    assert ak.validity_error(v2_array) == ""
    assert ak.validity_error(v2_array.to_typetracer()) == ""


def test_UnionArray():
    v2_array = ak.contents.unionarray.UnionArray(
        ak.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], dtype=np.int8)),
        ak.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100])),
        [
            ak.contents.recordarray.RecordArray(
                [ak.from_iter(["1", "2", "3"], highlevel=False)], ["nest"]
            ),
            ak.contents.recordarray.RecordArray(
                [
                    ak.contents.numpyarray.NumpyArray(
                        np.array([1.1, 2.2, 3.3, 4.4, 5.5])
                    )
                ],
                ["nest"],
            ),
        ],
    )

    assert ak.validity_error(v2_array) == ""
    assert ak.validity_error(v2_array.to_typetracer()) == ""


def test_UnmaskedArray():
    v2_array = ak.contents.unmaskedarray.UnmaskedArray(
        ak.contents.numpyarray.NumpyArray(
            np.array([0.0, 1.1, 2.2, 3.3], dtype=np.float64)
        )
    )

    assert ak.validity_error(v2_array) == ""
    assert ak.validity_error(v2_array.to_typetracer()) == ""
