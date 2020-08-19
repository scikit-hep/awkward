# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_flatten():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 10], dtype=numpy.int64))
    array = awkward1.layout.ListOffsetArray64(offsets, content)

    assert awkward1.to_list(array) == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]
    assert awkward1.to_list(array.flatten(axis=1)) == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
    assert awkward1.to_list(array.flatten(axis=-1)) == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
    with pytest.raises(ValueError) as err:
        assert awkward1.to_list(array.flatten(axis=-2))
    assert str(err.value).startswith("axis=0 not allowed for flatten")

    array2 = array[2:-1]
    assert awkward1.to_list(array2.flatten(axis=1)) == [3.3, 4.4, 5.5]
    assert awkward1.to_list(array2.flatten(axis=-1)) == [3.3, 4.4, 5.5]
