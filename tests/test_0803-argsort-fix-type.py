# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_argsort():
    array = ak.Array([[1.1, 2.2], [3.3, 3.1]])
    assert ak.argsort(array).tolist() == [[0, 1], [1, 0]]
    assert str(ak.type(ak.argsort(array))) == "2 * var * int64"

    empty_array = ak.Array([[], []])
    assert ak.argsort(empty_array).tolist() == [[], []]
    assert str(ak.type(ak.argsort(empty_array))) == "2 * var * int64"

    select_array = array[array > 5]
    assert select_array.tolist() == [[], []]
    assert str(ak.type(select_array)) == "2 * var * float64"

    assert ak.argsort(select_array).tolist() == [[], []]
    assert str(ak.type(ak.argsort(select_array))) == "2 * var * int64"

    assert ak.argsort(array[array > 5]).tolist() == [[], []]
    assert str(ak.type(ak.argsort(array[array > 5]))) == "2 * var * int64"
