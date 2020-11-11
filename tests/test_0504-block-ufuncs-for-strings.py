# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import numpy
import awkward1


def test():
    awkward1.deprecations_as_errors = True

    array = awkward1.to_categorical(awkward1.Array([321, 1.1, 123, 1.1, 999, 1.1, 2.0]))
    assert awkward1.to_list(array * 10) == [3210, 11, 1230, 11, 9990, 11, 20]

    array = awkward1.Array(["HAL"])
    with pytest.raises(ValueError):
        array + 1
