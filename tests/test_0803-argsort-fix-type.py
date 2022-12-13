# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak


def test_argsort():
    array = ak.Array([[1.1, 2.2], [3.3, 3.1]])
    assert ak.operations.argsort(array).to_list() == [[0, 1], [1, 0]]
    assert str(ak.operations.type(ak.operations.argsort(array))) == "2 * var * int64"

    empty_array = ak.Array([[], []])
    assert ak.operations.argsort(empty_array).to_list() == [[], []]
    assert (
        str(ak.operations.type(ak.operations.argsort(empty_array))) == "2 * var * int64"
    )

    select_array = array[array > 5]
    assert select_array.to_list() == [[], []]
    assert str(ak.operations.type(select_array)) == "2 * var * float64"

    assert ak.operations.argsort(select_array).to_list() == [[], []]
    assert (
        str(ak.operations.type(ak.operations.argsort(select_array)))
        == "2 * var * int64"
    )

    assert ak.operations.argsort(array[array > 5]).to_list() == [[], []]
    assert (
        str(ak.operations.type(ak.operations.argsort(array[array > 5])))
        == "2 * var * int64"
    )
