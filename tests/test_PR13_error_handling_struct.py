# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy
numba = pytest.importorskip("numba")

import awkward1
awkward1_numba_util = pytest.importorskip("awkward1._numba.util")

py27 = (sys.version_info[0] < 3)

def test_numpyarray():
    array = awkward1.layout.NumpyArray(numpy.arange(10)*1.1)

    with pytest.raises(ValueError) as excinfo:
        array[20]
    assert str(excinfo.value) == "in NumpyArray attempting to get 20, index out of range"

    with pytest.raises(ValueError) as excinfo:
        array[-20]
    assert str(excinfo.value) == "in NumpyArray attempting to get -20, index out of range"

    array[-20:20]

    with pytest.raises(ValueError) as excinfo:
        array[20,]
    assert str(excinfo.value) == "in NumpyArray attempting to get 20, index out of range"

    with pytest.raises(ValueError) as excinfo:
        array[-20,]
    assert str(excinfo.value) == "in NumpyArray attempting to get -20, index out of range"

    array[-20:20,]

    with pytest.raises(ValueError) as excinfo:
        array[2, 3]
    assert str(excinfo.value) == "in NumpyArray, too many dimensions in slice"

    with pytest.raises(ValueError) as excinfo:
        array[[5, 3, 20, 8]]
    assert str(excinfo.value) == "in NumpyArray attempting to get 20, index out of range"

    with pytest.raises(ValueError) as excinfo:
        array[[5, 3, -20, 8]]
    assert str(excinfo.value) == "in NumpyArray attempting to get -20, index out of range"

    array.setid()

    with pytest.raises(ValueError) as excinfo:
        array[20]
    assert str(excinfo.value) == "in NumpyArray attempting to get 20, index out of range"

    with pytest.raises(ValueError) as excinfo:
        array[-20]
    assert str(excinfo.value) == "in NumpyArray attempting to get -20, index out of range"

    array[-20:20]

    with pytest.raises(ValueError) as excinfo:
        array[20,]
    assert str(excinfo.value) == "in NumpyArray attempting to get 20, index out of range"

    with pytest.raises(ValueError) as excinfo:
        array[-20,]
    assert str(excinfo.value) == "in NumpyArray attempting to get -20, index out of range"

    array[-20:20,]

    with pytest.raises(ValueError) as excinfo:
        array[2, 3]
    assert str(excinfo.value) == "in NumpyArray, too many dimensions in slice"

    with pytest.raises(ValueError) as excinfo:
        array[[5, 3, 20, 8]]
    assert str(excinfo.value) == "in NumpyArray attempting to get 20, index out of range"

    with pytest.raises(ValueError) as excinfo:
        array[[5, 3, -20, 8]]
    assert str(excinfo.value) == "in NumpyArray attempting to get -20, index out of range"
