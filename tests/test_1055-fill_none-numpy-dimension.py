# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    a = ak.values_astype(ak.Array([1, None]), np.float32)

    assert ak.fill_none(a, np.float32(0)).tolist() == [1, 0]
    assert str(ak.fill_none(a, np.float32(0)).type) == "2 * float32"

    assert ak.fill_none(a, np.array(0, np.float32)).tolist() == [1, 0]
    assert str(ak.fill_none(a, np.array(0, np.float32)).type) == "2 * float32"

    assert ak.fill_none(a, np.array([0], np.float32)).tolist() == [1, [0]]
    assert (
        str(ak.fill_none(a, np.array([0], np.float32)).type)
        == "2 * union[float32, 1 * float32]"
    )

    assert ak.fill_none(a, np.array([[0]], np.float32)).tolist() == [1, [[0]]]
    assert (
        str(ak.fill_none(a, np.array([[0]], np.float32)).type)
        == "2 * union[float32, 1 * 1 * float32]"
    )

    assert ak.fill_none(a, 0).tolist() == [1, 0]
    assert str(ak.fill_none(a, 0).type) == "2 * float64"

    assert ak.fill_none(a, [0]).tolist() == [1, [0]]
    assert str(ak.fill_none(a, [0]).type) == "2 * union[float32, 1 * int64]"

    assert ak.fill_none(a, [[0]]).tolist() == [1, [[0]]]
    assert str(ak.fill_none(a, [[0]]).type) == "2 * union[float32, 1 * var * int64]"
