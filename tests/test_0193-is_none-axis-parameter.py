# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    assert ak.is_none(ak.Array([1, 2, 3, None, 5])).tolist() == [
        False,
        False,
        False,
        True,
        False,
    ]
    assert ak.is_none(ak.Array([[1, 2, 3], [], [None, 5]])).tolist() == [
        False,
        False,
        False,
    ]
    assert ak.is_none(ak.Array([[1, 2, 3], [], [None, 5]]), axis=1).tolist() == [
        [False, False, False],
        [],
        [True, False],
    ]
    assert ak.is_none(ak.Array([[1, None, 2, 3], [], [None, 5]]), axis=1).tolist() == [
        [False, True, False, False],
        [],
        [True, False],
    ]
    assert ak.is_none(ak.Array([[1, None, 2, 3], [], [None, 5]]), axis=-1).tolist() == [
        [False, True, False, False],
        [],
        [True, False],
    ]
    assert ak.is_none(ak.Array([[1, None, 2, 3], [], [None, 5]]), axis=-2).tolist() == [
        False,
        False,
        False,
    ]
    assert ak.is_none(
        ak.Array([[1, None, 2, 3], None, [None, 5]]), axis=-2
    ).tolist() == [False, True, False]
    with pytest.raises(ValueError):
        ak.is_none(ak.Array([[1, None, 2, 3], [], [None, 5]]), axis=-3)
