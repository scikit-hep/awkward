# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_byte_masked_array():
    array = ak._v2.highlevel.Array(
        ak._v2.contents.ByteMaskedArray(
            ak._v2.index.Index8(np.array([True, True])),
            ak._v2.contents.RecordArray(
                [
                    ak._v2.contents.NumpyArray(np.arange(2)),
                    ak._v2.contents.IndexedOptionArray(
                        ak._v2.index.Index64(np.array([0, -1])),
                        ak._v2.contents.NumpyArray(np.array([1], dtype=np.int64)),
                    ),
                ],
                ["a", "b"],
            ),
            # ak._v2.Array({"a": [1, 2], "b": [1, None]}).layout,
            valid_when=True,
        )
    )

    assert str(array.b.type) == "2 * ?int64"


def test_bit_masked_array():
    array = ak._v2.highlevel.Array(
        ak._v2.contents.BitMaskedArray(
            ak._v2.index.IndexU8(np.array([1 << 0 | 1 << 1], dtype=np.uint8)),
            ak._v2.contents.RecordArray(
                [
                    ak._v2.contents.NumpyArray(np.arange(2)),
                    ak._v2.contents.IndexedOptionArray(
                        ak._v2.index.Index64(np.array([0, -1])),
                        ak._v2.contents.NumpyArray(np.array([1], dtype=np.int64)),
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
