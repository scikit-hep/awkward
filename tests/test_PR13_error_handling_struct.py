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

def test_listarray_numpyarray():
    starts  = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6]))
    stops   = awkward1.layout.Index64(numpy.array([3, 3, 5, 6]))
    content = awkward1.layout.NumpyArray(numpy.arange(10)*1.1)
    array   = awkward1.layout.ListArray64(starts, stops, content)

    with pytest.raises(ValueError) as excinfo:
        array[4]
    assert str(excinfo.value) == "in ListArray64, len(stops) < len(starts)"

    starts  = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6]))
    stops   = awkward1.layout.Index64(numpy.array([3, 3, 5, 6, 10]))
    content = awkward1.layout.NumpyArray(numpy.arange(10)*1.1)
    array   = awkward1.layout.ListArray64(starts, stops, content)

    with pytest.raises(ValueError) as excinfo:
        array[20]
    assert str(excinfo.value) == "in ListArray64 attempting to get 20, index out of range"

    with pytest.raises(ValueError) as excinfo:
        array[-20]
    assert str(excinfo.value) == "in ListArray64 attempting to get -20, index out of range"

    array[-20:20]

    with pytest.raises(ValueError) as excinfo:
        array[20,]
    assert str(excinfo.value) == "in ListArray64 attempting to get 20, index out of range"

    with pytest.raises(ValueError) as excinfo:
        array[-20,]
    assert str(excinfo.value) == "in ListArray64 attempting to get -20, index out of range"

    array[-20:20,]

    with pytest.raises(ValueError) as excinfo:
        array[2, 1, 0]
    assert str(excinfo.value) == "in NumpyArray, too many dimensions in slice"

    with pytest.raises(ValueError) as excinfo:
        array[[2, 0, 0, 20, 3]]
    assert str(excinfo.value) == "in ListArray64 attempting to get 20, index out of range"

    with pytest.raises(ValueError) as excinfo:
        array[[2, 0, 0, -20, 3]]
    assert str(excinfo.value) == "in ListArray64 attempting to get -20, index out of range"

def test_current():
    starts  = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6]))
    stops   = awkward1.layout.Index64(numpy.array([3, 3, 5, 6, 10]))
    content = awkward1.layout.NumpyArray(numpy.arange(10)*1.1)
    array   = awkward1.layout.ListArray64(starts, stops, content)

    array.setid()

    with pytest.raises(ValueError) as excinfo:
        array[2, 20]
    assert str(excinfo.value) == "in ListArray64 at id[2] attempting to get 20, index out of range"
