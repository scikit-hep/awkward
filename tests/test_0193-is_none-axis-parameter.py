# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    assert ak._v2.operations.is_none(ak._v2.Array([1, 2, 3, None, 5])).tolist() == [
        False,
        False,
        False,
        True,
        False,
    ]
    assert ak._v2.operations.is_none(
        ak._v2.Array([[1, 2, 3], [], [None, 5]])
    ).tolist() == [
        False,
        False,
        False,
    ]
    assert ak._v2.operations.is_none(
        ak._v2.Array([[1, 2, 3], [], [None, 5]]), axis=1
    ).tolist() == [
        [False, False, False],
        [],
        [True, False],
    ]
    assert ak._v2.operations.is_none(
        ak._v2.Array([[1, None, 2, 3], [], [None, 5]]), axis=1
    ).tolist() == [
        [False, True, False, False],
        [],
        [True, False],
    ]
    assert ak._v2.operations.is_none(
        ak._v2.Array([[1, None, 2, 3], [], [None, 5]]), axis=-1
    ).tolist() == [
        [False, True, False, False],
        [],
        [True, False],
    ]
    assert ak._v2.operations.is_none(
        ak._v2.Array([[1, None, 2, 3], [], [None, 5]]), axis=-2
    ).tolist() == [
        False,
        False,
        False,
    ]
    assert ak._v2.operations.is_none(
        ak._v2.Array([[1, None, 2, 3], None, [None, 5]]), axis=-2
    ).tolist() == [False, True, False]
    with pytest.raises(ValueError):
        ak._v2.operations.is_none(
            ak._v2.Array([[1, None, 2, 3], [], [None, 5]]), axis=-3
        )

    one = ak._v2.operations.from_iter([1, None, 3], highlevel=False)
    two = ak._v2.operations.from_iter([[], [1], None, [3, 3, 3]], highlevel=False)
    tags = ak._v2.index.Index8(np.array([0, 1, 1, 0, 0, 1, 1], dtype=np.int8))
    index = ak._v2.index.Index64(np.array([0, 0, 1, 1, 2, 2, 3], dtype=np.int64))
    array = ak._v2.Array(
        ak._v2.contents.UnionArray(tags, index, [one, two]), check_valid=True
    )
    assert ak._v2.to_list(array) == [1, [], [1], None, 3, None, [3, 3, 3]]

    assert ak._v2.to_list(ak._v2.operations.is_none(array)) == [
        False,
        False,
        False,
        True,
        False,
        True,
        False,
    ]
