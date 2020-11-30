# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest
import numpy as np
import awkward1 as ak


def test():
    data = ak.Array([{"x": i} for i in range(10)])
    y = ak.Array(np.array([[i, i] for i in range(10)]))
    data["y"] = y
    assert data.tolist() == [{"x": 0, "y": [0, 0]}, {"x": 1, "y": [1, 1]}, {"x": 2, "y": [2, 2]}, {"x": 3, "y": [3, 3]}, {"x": 4, "y": [4, 4]}, {"x": 5, "y": [5, 5]}, {"x": 6, "y": [6, 6]}, {"x": 7, "y": [7, 7]}, {"x": 8, "y": [8, 8]}, {"x": 9, "y": [9, 9]}]


def test_regular():
    regular = ak.Array(np.array([[i, i] for i in range(10)], dtype=np.int64))
    assert str(regular.type) == "10 * 2 * int64"

    assert ak.to_list(ak.to_regular(regular)) == ak.to_list(regular)
    assert ak.type(ak.to_regular(regular)) == ak.type(regular)

    irregular = ak.from_regular(regular)
    assert ak.to_list(irregular) == ak.to_list(regular)
    assert str(irregular.type) == "10 * var * int64"

    assert ak.to_list(ak.from_regular(irregular)) == ak.to_list(irregular)
    assert ak.type(ak.from_regular(irregular)) == ak.type(irregular)

    reregular = ak.to_regular(irregular)
    assert ak.to_list(reregular) == ak.to_list(regular)
    assert str(reregular.type) == "10 * 2 * int64"

def test_regular_deep():
    regular = ak.Array(np.array([[[i, i, i], [i, i, i]] for i in range(10)], dtype=np.int64))
    assert str(regular.type) == "10 * 2 * 3 * int64"

    irregular = ak.from_regular(regular, axis=1)
    assert ak.to_list(irregular) == ak.to_list(regular)
    assert str(irregular.type) == "10 * var * 3 * int64"

    reregular = ak.to_regular(irregular, axis=1)
    assert ak.to_list(reregular) == ak.to_list(regular)
    assert str(reregular.type) == "10 * 2 * 3 * int64"

    irregular = ak.from_regular(regular, axis=2)
    assert ak.to_list(irregular) == ak.to_list(regular)
    assert str(irregular.type) == "10 * 2 * var * int64"

    reregular = ak.to_regular(irregular, axis=2)
    assert ak.to_list(reregular) == ak.to_list(regular)
    assert str(reregular.type) == "10 * 2 * 3 * int64"

    irregular = ak.from_regular(regular, axis=-1)
    assert ak.to_list(irregular) == ak.to_list(regular)
    assert str(irregular.type) == "10 * 2 * var * int64"

    reregular = ak.to_regular(irregular, axis=-1)
    assert ak.to_list(reregular) == ak.to_list(regular)
    assert str(reregular.type) == "10 * 2 * 3 * int64"

    irregular = ak.from_regular(regular, axis=-2)
    assert ak.to_list(irregular) == ak.to_list(regular)
    assert str(irregular.type) == "10 * var * 3 * int64"

    reregular = ak.to_regular(irregular, axis=-2)
    assert ak.to_list(reregular) == ak.to_list(regular)
    assert str(reregular.type) == "10 * 2 * 3 * int64"

    with pytest.raises(ValueError):
        ak.from_regular(regular, axis=-3)

    assert ak.to_list(ak.from_regular(regular, axis=0)) == ak.to_list(regular)
    assert ak.type(ak.from_regular(regular, axis=0)) == ak.type(regular)
