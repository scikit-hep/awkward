# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_argsort():
    array = ak._v2.Array([[1.1, 2.2], [3.3, 3.1]])
    assert ak._v2.operations.argsort(array).tolist() == [[0, 1], [1, 0]]
    assert (
        str(ak._v2.operations.type(ak._v2.operations.argsort(array)))
        == "2 * var * int64"
    )

    empty_array = ak._v2.Array([[], []])
    assert ak._v2.operations.argsort(empty_array).tolist() == [[], []]
    assert (
        str(ak._v2.operations.type(ak._v2.operations.argsort(empty_array)))
        == "2 * var * int64"
    )

    select_array = array[array > 5]
    assert select_array.tolist() == [[], []]
    assert str(ak._v2.operations.type(select_array)) == "2 * var * float64"

    assert ak._v2.operations.argsort(select_array).tolist() == [[], []]
    assert (
        str(ak._v2.operations.type(ak._v2.operations.argsort(select_array)))
        == "2 * var * int64"
    )

    assert ak._v2.operations.argsort(array[array > 5]).tolist() == [[], []]
    assert (
        str(ak._v2.operations.type(ak._v2.operations.argsort(array[array > 5])))
        == "2 * var * int64"
    )
