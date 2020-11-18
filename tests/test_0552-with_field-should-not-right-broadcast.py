# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import numpy
import awkward1


def test():
    data = awkward1.Array([{"x": i} for i in range(10)])
    y = awkward1.Array(numpy.array([[i, i] for i in range(10)]))
    data["y"] = y
    assert data.tolist() == [{"x": 0, "y": [0, 0]}, {"x": 1, "y": [1, 1]}, {"x": 2, "y": [2, 2]}, {"x": 3, "y": [3, 3]}, {"x": 4, "y": [4, 4]}, {"x": 5, "y": [5, 5]}, {"x": 6, "y": [6, 6]}, {"x": 7, "y": [7, 7]}, {"x": 8, "y": [8, 8]}, {"x": 9, "y": [9, 9]}]


def test_regular():
    regular = awkward1.Array(numpy.array([[i, i] for i in range(10)], dtype=numpy.int64))
    assert str(regular.type) == "10 * 2 * int64"

    assert awkward1.to_list(awkward1.to_regular(regular)) == awkward1.to_list(regular)
    assert awkward1.type(awkward1.to_regular(regular)) == awkward1.type(regular)

    irregular = awkward1.from_regular(regular)
    assert awkward1.to_list(irregular) == awkward1.to_list(regular)
    assert str(irregular.type) == "10 * var * int64"

    assert awkward1.to_list(awkward1.from_regular(irregular)) == awkward1.to_list(irregular)
    assert awkward1.type(awkward1.from_regular(irregular)) == awkward1.type(irregular)

    reregular = awkward1.to_regular(irregular)
    assert awkward1.to_list(reregular) == awkward1.to_list(regular)
    assert str(reregular.type) == "10 * 2 * int64"

def test_regular_deep():
    regular = awkward1.Array(numpy.array([[[i, i, i], [i, i, i]] for i in range(10)], dtype=numpy.int64))
    assert str(regular.type) == "10 * 2 * 3 * int64"

    irregular = awkward1.from_regular(regular, axis=1)
    assert awkward1.to_list(irregular) == awkward1.to_list(regular)
    assert str(irregular.type) == "10 * var * 3 * int64"

    reregular = awkward1.to_regular(irregular, axis=1)
    assert awkward1.to_list(reregular) == awkward1.to_list(regular)
    assert str(reregular.type) == "10 * 2 * 3 * int64"

    irregular = awkward1.from_regular(regular, axis=2)
    assert awkward1.to_list(irregular) == awkward1.to_list(regular)
    assert str(irregular.type) == "10 * 2 * var * int64"

    reregular = awkward1.to_regular(irregular, axis=2)
    assert awkward1.to_list(reregular) == awkward1.to_list(regular)
    assert str(reregular.type) == "10 * 2 * 3 * int64"

    irregular = awkward1.from_regular(regular, axis=-1)
    assert awkward1.to_list(irregular) == awkward1.to_list(regular)
    assert str(irregular.type) == "10 * 2 * var * int64"

    reregular = awkward1.to_regular(irregular, axis=-1)
    assert awkward1.to_list(reregular) == awkward1.to_list(regular)
    assert str(reregular.type) == "10 * 2 * 3 * int64"

    irregular = awkward1.from_regular(regular, axis=-2)
    assert awkward1.to_list(irregular) == awkward1.to_list(regular)
    assert str(irregular.type) == "10 * var * 3 * int64"

    reregular = awkward1.to_regular(irregular, axis=-2)
    assert awkward1.to_list(reregular) == awkward1.to_list(regular)
    assert str(reregular.type) == "10 * 2 * 3 * int64"

    with pytest.raises(ValueError):
        awkward1.from_regular(regular, axis=-3)

    assert awkward1.to_list(awkward1.from_regular(regular, axis=0)) == awkward1.to_list(regular)
    assert awkward1.type(awkward1.from_regular(regular, axis=0)) == awkward1.type(regular)
