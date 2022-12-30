# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak

to_list = ak.operations.to_list


def test_from_regular():
    array = ak.contents.NumpyArray(
        np.arange(2 * 3 * 5, dtype=np.int64).reshape(2, 3, 5)
    )
    irregular1 = ak.operations.from_regular(array, axis=1, highlevel=False)
    irregular2 = ak.operations.from_regular(array, axis=2, highlevel=False)
    irregularNone = ak.operations.from_regular(array, axis=None, highlevel=False)

    assert to_list(irregular1) == to_list(array)
    assert to_list(irregular2) == to_list(array)
    assert to_list(irregularNone) == to_list(array)

    assert str(array.form.type) == "3 * 5 * int64"
    assert str(irregular1.form.type) == "var * 5 * int64"
    assert str(irregular2.form.type) == "3 * var * int64"
    assert str(irregularNone.form.type) == "var * var * int64"

    array = ak.contents.RegularArray(
        ak.contents.RegularArray(
            ak.contents.NumpyArray(np.arange(2 * 3 * 5, dtype=np.int64)),
            5,
        ),
        3,
    )
    irregular1 = ak.operations.from_regular(array, axis=1, highlevel=False)
    irregular2 = ak.operations.from_regular(array, axis=2, highlevel=False)
    irregularNone = ak.operations.from_regular(array, axis=None, highlevel=False)

    assert to_list(irregular1) == to_list(array)
    assert to_list(irregular2) == to_list(array)
    assert to_list(irregularNone) == to_list(array)

    assert str(array.form.type) == "3 * 5 * int64"
    assert str(irregular1.form.type) == "var * 5 * int64"
    assert str(irregular2.form.type) == "3 * var * int64"
    assert str(irregularNone.form.type) == "var * var * int64"

    array = ak.contents.ByteMaskedArray(
        ak.index.Index8(np.array([False, True]).view(np.int8)),
        ak.contents.RegularArray(
            ak.contents.ByteMaskedArray(
                ak.index.Index8(
                    np.array([False, True, True, True, False, True]).view(np.int8)
                ),
                ak.contents.RegularArray(
                    ak.contents.NumpyArray(np.arange(2 * 3 * 5, dtype=np.int64)),
                    5,
                ),
                valid_when=True,
            ),
            3,
        ),
        valid_when=True,
    )
    irregular1 = ak.operations.from_regular(array, axis=1, highlevel=False)
    irregular2 = ak.operations.from_regular(array, axis=2, highlevel=False)
    irregularNone = ak.operations.from_regular(array, axis=None, highlevel=False)

    assert to_list(irregular1) == to_list(array)
    assert to_list(irregular2) == to_list(array)
    assert to_list(irregularNone) == to_list(array)

    assert str(array.form.type) == "option[3 * option[5 * int64]]"
    assert str(irregular1.form.type) == "option[var * option[5 * int64]]"
    assert str(irregular2.form.type) == "option[3 * option[var * int64]]"
    assert str(irregularNone.form.type) == "option[var * option[var * int64]]"


def test_to_regular():
    array = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 3, 6], dtype=np.int64)),
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.int64)),
            ak.contents.NumpyArray(np.arange(2 * 3 * 5, dtype=np.int64)),
        ),
    )
    regular1 = ak.operations.to_regular(array, axis=1, highlevel=False)
    regular2 = ak.operations.to_regular(array, axis=2, highlevel=False)
    regularNone = ak.operations.to_regular(array, axis=None, highlevel=False)

    assert to_list(regular1) == to_list(array)
    assert to_list(regular2) == to_list(array)
    assert to_list(regularNone) == to_list(array)

    assert str(array.form.type) == "var * var * int64"
    assert str(regular1.form.type) == "3 * var * int64"
    assert str(regular2.form.type) == "var * 5 * int64"
    assert str(regularNone.form.type) == "3 * 5 * int64"

    array = ak.contents.ByteMaskedArray(
        ak.index.Index8(np.array([False, True]).view(np.int8)),
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 3, 6], dtype=np.int64)),
            ak.contents.ByteMaskedArray(
                ak.index.Index8(
                    np.array([False, True, True, True, False, True]).view(np.int8)
                ),
                ak.contents.ListOffsetArray(
                    ak.index.Index64(
                        np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.int64)
                    ),
                    ak.contents.NumpyArray(np.arange(2 * 3 * 5, dtype=np.int64)),
                ),
                valid_when=True,
            ),
        ),
        valid_when=True,
    )
    regular1 = ak.operations.to_regular(array, axis=1, highlevel=False)
    regular2 = ak.operations.to_regular(array, axis=2, highlevel=False)
    regularNone = ak.operations.to_regular(array, axis=None, highlevel=False)

    assert to_list(regular1) == to_list(array)
    assert to_list(regular2) == to_list(array)
    assert to_list(regularNone) == to_list(array)

    assert str(array.form.type) == "option[var * option[var * int64]]"
    assert str(regular1.form.type) == "option[3 * option[var * int64]]"
    assert str(regular2.form.type) == "option[var * option[5 * int64]]"
    assert str(regularNone.form.type) == "option[3 * option[5 * int64]]"


def test_isclose():
    one = ak.operations.from_iter([0.99999, 1.99999, 2.99999], highlevel=False)
    two = ak.operations.from_iter([1.00001, 2.00001, 3.00001], highlevel=False)
    assert to_list(ak.operations.isclose(one, two)) == [
        False,
        True,
        True,
    ]

    one = ak.operations.from_iter([[0.99999, 1.99999], [], [2.99999]], highlevel=False)
    two = ak.operations.from_iter([[1.00001, 2.00001], [], [3.00001]], highlevel=False)
    assert to_list(ak.operations.isclose(one, two)) == [
        [False, True],
        [],
        [True],
    ]

    one = ak.operations.from_iter(
        [[0.99999, 1.99999, None], [], [2.99999]], highlevel=False
    )
    two = ak.operations.from_iter(
        [[1.00001, 2.00001, None], [], [3.00001]], highlevel=False
    )
    assert to_list(ak.operations.isclose(one, two)) == [
        [False, True, None],
        [],
        [True],
    ]
