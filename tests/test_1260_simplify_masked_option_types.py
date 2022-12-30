# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test_byte_masked_array():
    array = ak.highlevel.Array(
        ak.contents.ByteMaskedArray(
            ak.index.Index8(np.array([True, True])),
            ak.contents.RecordArray(
                [
                    ak.contents.NumpyArray(np.arange(2)),
                    ak.contents.IndexedOptionArray(
                        ak.index.Index64(np.array([0, -1])),
                        ak.contents.NumpyArray(np.array([1], dtype=np.int64)),
                    ),
                ],
                ["a", "b"],
            ),
            # ak.Array({"a": [1, 2], "b": [1, None]}).layout,
            valid_when=True,
        )
    )

    assert str(array.b.type) == "2 * ?int64"


def test_bit_masked_array():
    array = ak.highlevel.Array(
        ak.contents.BitMaskedArray(
            ak.index.IndexU8(np.array([1 << 0 | 1 << 1], dtype=np.uint8)),
            ak.contents.RecordArray(
                [
                    ak.contents.NumpyArray(np.arange(2)),
                    ak.contents.IndexedOptionArray(
                        ak.index.Index64(np.array([0, -1])),
                        ak.contents.NumpyArray(np.array([1], dtype=np.int64)),
                    ),
                ],
                ["a", "b"],
            ),
            valid_when=True,
            length=2,
            lsb_order=True,
        )
    )

    assert str(array.b.type) == "2 * ?int64"
