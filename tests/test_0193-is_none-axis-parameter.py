# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak


def test():
    assert ak.operations.is_none(ak.Array([1, 2, 3, None, 5])).to_list() == [
        False,
        False,
        False,
        True,
        False,
    ]
    assert ak.operations.is_none(ak.Array([[1, 2, 3], [], [None, 5]])).to_list() == [
        False,
        False,
        False,
    ]
    assert ak.operations.is_none(
        ak.Array([[1, 2, 3], [], [None, 5]]), axis=1
    ).to_list() == [
        [False, False, False],
        [],
        [True, False],
    ]
    assert ak.operations.is_none(
        ak.Array([[1, None, 2, 3], [], [None, 5]]), axis=1
    ).to_list() == [
        [False, True, False, False],
        [],
        [True, False],
    ]
    assert ak.operations.is_none(
        ak.Array([[1, None, 2, 3], [], [None, 5]]), axis=-1
    ).to_list() == [
        [False, True, False, False],
        [],
        [True, False],
    ]
    assert ak.operations.is_none(
        ak.Array([[1, None, 2, 3], [], [None, 5]]), axis=-2
    ).to_list() == [
        False,
        False,
        False,
    ]
    assert ak.operations.is_none(
        ak.Array([[1, None, 2, 3], None, [None, 5]]), axis=-2
    ).to_list() == [False, True, False]
    with pytest.raises(ValueError):
        ak.operations.is_none(ak.Array([[1, None, 2, 3], [], [None, 5]]), axis=-3)

    one = ak.operations.from_iter([1, None, 3], highlevel=False)
    two = ak.operations.from_iter([[], [1], None, [3, 3, 3]], highlevel=False)
    tags = ak.index.Index8(np.array([0, 1, 1, 0, 0, 1, 1], dtype=np.int8))
    index = ak.index.Index64(np.array([0, 0, 1, 1, 2, 2, 3], dtype=np.int64))
    array = ak.Array(ak.contents.UnionArray(tags, index, [one, two]), check_valid=True)
    assert ak.to_list(array) == [1, [], [1], None, 3, None, [3, 3, 3]]

    assert ak.to_list(ak.operations.is_none(array)) == [
        False,
        False,
        False,
        True,
        False,
        True,
        False,
    ]
