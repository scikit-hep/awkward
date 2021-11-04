# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2


def test_from_regular():
    array = ak._v2.contents.NumpyArray(
        np.arange(2 * 3 * 5, dtype=np.int64).reshape(2, 3, 5)
    )
    irregular1 = ak._v2.operations.structure.from_regular(
        array, axis=1, highlevel=False
    )
    irregular2 = ak._v2.operations.structure.from_regular(
        array, axis=2, highlevel=False
    )
    irregularNone = ak._v2.operations.structure.from_regular(
        array, axis=None, highlevel=False
    )

    assert ak.to_list(irregular1) == ak.to_list(array)
    assert ak.to_list(irregular2) == ak.to_list(array)
    assert ak.to_list(irregularNone) == ak.to_list(array)

    assert str(array.form.type) == "3 * 5 * int64"
    assert str(irregular1.form.type) == "var * 5 * int64"
    assert str(irregular2.form.type) == "3 * var * int64"
    assert str(irregularNone.form.type) == "var * var * int64"

    array = ak._v2.contents.RegularArray(
        ak._v2.contents.RegularArray(
            ak._v2.contents.NumpyArray(np.arange(2 * 3 * 5, dtype=np.int64)),
            5,
        ),
        3,
    )
    irregular1 = ak._v2.operations.structure.from_regular(
        array, axis=1, highlevel=False
    )
    irregular2 = ak._v2.operations.structure.from_regular(
        array, axis=2, highlevel=False
    )
    irregularNone = ak._v2.operations.structure.from_regular(
        array, axis=None, highlevel=False
    )

    assert ak.to_list(irregular1) == ak.to_list(array)
    assert ak.to_list(irregular2) == ak.to_list(array)
    assert ak.to_list(irregularNone) == ak.to_list(array)

    assert str(array.form.type) == "3 * 5 * int64"
    assert str(irregular1.form.type) == "var * 5 * int64"
    assert str(irregular2.form.type) == "3 * var * int64"
    assert str(irregularNone.form.type) == "var * var * int64"

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
    irregular1 = ak._v2.operations.structure.from_regular(
        array, axis=1, highlevel=False
    )
    irregular2 = ak._v2.operations.structure.from_regular(
        array, axis=2, highlevel=False
    )
    irregularNone = ak._v2.operations.structure.from_regular(
        array, axis=None, highlevel=False
    )

    assert ak.to_list(irregular1) == ak.to_list(array)
    assert ak.to_list(irregular2) == ak.to_list(array)
    assert ak.to_list(irregularNone) == ak.to_list(array)

    assert str(array.form.type) == "option[3 * option[5 * int64]]"
    assert str(irregular1.form.type) == "option[var * option[5 * int64]]"
    assert str(irregular2.form.type) == "option[3 * option[var * int64]]"
    assert str(irregularNone.form.type) == "option[var * option[var * int64]]"


def test_to_regular():
    array = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 3, 6], dtype=np.int64)),
        ak._v2.contents.ListOffsetArray(
            ak._v2.index.Index64(np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.int64)),
            ak._v2.contents.NumpyArray(np.arange(2 * 3 * 5, dtype=np.int64)),
        ),
    )
    regular1 = ak._v2.operations.structure.to_regular(array, axis=1, highlevel=False)
    regular2 = ak._v2.operations.structure.to_regular(array, axis=2, highlevel=False)
    regularNone = ak._v2.operations.structure.to_regular(
        array, axis=None, highlevel=False
    )

    assert ak.to_list(regular1) == ak.to_list(array)
    assert ak.to_list(regular2) == ak.to_list(array)
    assert ak.to_list(regularNone) == ak.to_list(array)

    assert str(array.form.type) == "var * var * int64"
    assert str(regular1.form.type) == "3 * var * int64"
    assert str(regular2.form.type) == "var * 5 * int64"
    assert str(regularNone.form.type) == "3 * 5 * int64"

    array = ak._v2.contents.ByteMaskedArray(
        ak._v2.index.Index8(np.array([False, True]).view(np.int8)),
        ak._v2.contents.ListOffsetArray(
            ak._v2.index.Index64(np.array([0, 3, 6], dtype=np.int64)),
            ak._v2.contents.ByteMaskedArray(
                ak._v2.index.Index8(
                    np.array([False, True, True, True, False, True]).view(np.int8)
                ),
                ak._v2.contents.ListOffsetArray(
                    ak._v2.index.Index64(
                        np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.int64)
                    ),
                    ak._v2.contents.NumpyArray(np.arange(2 * 3 * 5, dtype=np.int64)),
                ),
                valid_when=True,
            ),
        ),
        valid_when=True,
    )
    regular1 = ak._v2.operations.structure.to_regular(array, axis=1, highlevel=False)
    regular2 = ak._v2.operations.structure.to_regular(array, axis=2, highlevel=False)
    regularNone = ak._v2.operations.structure.to_regular(
        array, axis=None, highlevel=False
    )

    assert ak.to_list(regular1) == ak.to_list(array)
    assert ak.to_list(regular2) == ak.to_list(array)
    assert ak.to_list(regularNone) == ak.to_list(array)

    assert str(array.form.type) == "option[var * option[var * int64]]"
    assert str(regular1.form.type) == "option[3 * option[var * int64]]"
    assert str(regular2.form.type) == "option[var * option[5 * int64]]"
    assert str(regularNone.form.type) == "option[3 * option[5 * int64]]"


def test_isclose():
    one = v1_to_v2(
        ak.from_iter([0.999999999, 1.999999999, 2.999999999], highlevel=False)
    )
    two = v1_to_v2(
        ak.from_iter([1.000000001, 2.000000001, 3.000000001], highlevel=False)
    )
    assert ak.to_list(ak._v2.operations.structure.isclose(one, two)) == [
        True,
        True,
        True,
    ]

    one = v1_to_v2(
        ak.from_iter([[0.999999999, 1.999999999], [], [2.999999999]], highlevel=False)
    )
    two = v1_to_v2(
        ak.from_iter([[1.000000001, 2.000000001], [], [3.000000001]], highlevel=False)
    )
    assert ak.to_list(ak._v2.operations.structure.isclose(one, two)) == [
        [True, True],
        [],
        [True],
    ]

    one = v1_to_v2(
        ak.from_iter(
            [[0.999999999, 1.999999999, None], [], [2.999999999]], highlevel=False
        )
    )
    two = v1_to_v2(
        ak.from_iter(
            [[1.000000001, 2.000000001, None], [], [3.000000001]], highlevel=False
        )
    )
    assert ak.to_list(ak._v2.operations.structure.isclose(one, two)) == [
        [True, True, None],
        [],
        [True],
    ]
