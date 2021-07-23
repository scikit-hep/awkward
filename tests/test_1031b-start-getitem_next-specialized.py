# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2


def test_NumpyArray():
    a = ak._v2.contents.RegularArray(
        v1_to_v2(ak.from_numpy(np.arange(2 * 3 * 5).reshape(-1, 5)).layout), 3
    )
    assert ak.to_list(a[1]) == [
        [15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29],
    ]
    assert ak.to_list(a[1, -2]) == [20, 21, 22, 23, 24]
    assert a[1, -2, 2] == 22
    with pytest.raises(IndexError):
        a[1, -2, 2, 0]
    assert ak.to_list(a[1, -2, 2:]) == [22, 23, 24]
    with pytest.raises(IndexError):
        a[1, -2, 2:, 0]
    with pytest.raises(IndexError):
        a[1, -2, "hello"]
    with pytest.raises(IndexError):
        a[1, -2, ["hello", "there"]]
    assert ak.to_list(a[1, -2, np.newaxis, 2]) == [22]
    assert ak.to_list(a[1, -2, np.newaxis, np.newaxis, 2]) == [[22]]
    assert ak.to_list(a[1, -2, ...]) == [20, 21, 22, 23, 24]
    assert a[1, -2, ..., 2] == 22
    with pytest.raises(IndexError):
        a[1, -2, ..., 2, 2]
    assert ak.to_list(a[1, -2, [3, 1, 1, 2]]) == [23, 21, 21, 22]
    with pytest.raises(IndexError):
        a[1, -2, [3, 1, 1, 2], 2]


def test_RegularArray():
    old = ak.layout.RegularArray(
        ak.from_numpy(np.arange(2 * 3 * 5).reshape(-1, 5)).layout, 3
    )
    new = v1_to_v2(old)

    assert ak.to_list(old[1, 1:]) == [[20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]
    assert ak.to_list(new[1, 1:]) == [[20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]

    with pytest.raises(IndexError):
        new[1, "hello"]

    with pytest.raises(IndexError):
        new[1, ["hello", "there"]]

    assert ak.to_list(new[1, np.newaxis, -2]) == [[20, 21, 22, 23, 24]]
    assert ak.to_list(new[1, np.newaxis, np.newaxis, -2]) == [[[20, 21, 22, 23, 24]]]

    assert old.minmax_depth == (3, 3)
    assert new.minmax_depth == (3, 3)
