# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_from_regular():
    array = ak._v2.contents.NumpyArray(
        np.arange(2 * 3 * 5, dtype=np.int64).reshape(2, 3, 5)
    )
    regular1 = ak._v2.operations.structure.from_regular(array, axis=1, highlevel=False)
    regular2 = ak._v2.operations.structure.from_regular(array, axis=2, highlevel=False)
    regularNone = ak._v2.operations.structure.from_regular(
        array, axis=None, highlevel=False
    )

    assert ak.to_list(regular1) == ak.to_list(array)
    assert ak.to_list(regular2) == ak.to_list(array)
    assert ak.to_list(regularNone) == ak.to_list(array)

    assert str(array.form.type) == "3 * 5 * int64"
    assert str(regular1.form.type) == "var * 5 * int64"
    assert str(regular2.form.type) == "3 * var * int64"
    assert str(regularNone.form.type) == "var * var * int64"

    array = ak._v2.contents.RegularArray(
        ak._v2.contents.RegularArray(
            ak._v2.contents.NumpyArray(np.arange(2 * 3 * 5, dtype=np.int64)),
            5,
        ),
        3,
    )
    regular1 = ak._v2.operations.structure.from_regular(array, axis=1, highlevel=False)
    regular2 = ak._v2.operations.structure.from_regular(array, axis=2, highlevel=False)
    regularNone = ak._v2.operations.structure.from_regular(
        array, axis=None, highlevel=False
    )

    assert ak.to_list(regular1) == ak.to_list(array)
    assert ak.to_list(regular2) == ak.to_list(array)
    assert ak.to_list(regularNone) == ak.to_list(array)

    assert str(array.form.type) == "3 * 5 * int64"
    assert str(regular1.form.type) == "var * 5 * int64"
    assert str(regular2.form.type) == "3 * var * int64"
    assert str(regularNone.form.type) == "var * var * int64"

    array = ak._v2.contents.ByteMaskedArray(
        ak._v2.index.Index8(np.array([False, True]).view(np.int8)),
        ak._v2.contents.RegularArray(
            ak._v2.contents.ByteMaskedArray(
                ak._v2.index.Index8(
                    np.array([False, True, True, True, False, True]).view(np.int8)
                ),
                ak._v2.contents.RegularArray(
                    ak._v2.contents.NumpyArray(np.arange(2 * 3 * 5, dtype=np.int64)),
                    5,
                ),
                valid_when=True,
            ),
            3,
        ),
        valid_when=True,
    )
    regular1 = ak._v2.operations.structure.from_regular(array, axis=1, highlevel=False)
    regular2 = ak._v2.operations.structure.from_regular(array, axis=2, highlevel=False)
    regularNone = ak._v2.operations.structure.from_regular(
        array, axis=None, highlevel=False
    )

    assert ak.to_list(regular1) == ak.to_list(array)
    assert ak.to_list(regular2) == ak.to_list(array)
    assert ak.to_list(regularNone) == ak.to_list(array)

    assert str(array.form.type) == "option[3 * option[5 * int64]]"
    assert str(regular1.form.type) == "option[var * option[5 * int64]]"
    assert str(regular2.form.type) == "option[3 * option[var * int64]]"
    assert str(regularNone.form.type) == "option[var * option[var * int64]]"
