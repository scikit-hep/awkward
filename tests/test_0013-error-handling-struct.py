# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

py27 = (sys.version_info[0] < 3)

def test_numpyarray():
    array = awkward1.layout.NumpyArray(numpy.arange(10)*1.1)

    with pytest.raises(ValueError) as excinfo:
        array[20]
    assert " attempting to get 20, index out of range" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array[-20]
    assert " attempting to get -20, index out of range" in str(excinfo.value)

    array[-20:20]

    with pytest.raises(ValueError) as excinfo:
        array[20,]
    assert " attempting to get 20, index out of range" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array[-20,]
    assert " attempting to get -20, index out of range" in str(excinfo.value)

    array[-20:20,]

    with pytest.raises(ValueError) as excinfo:
        array[2, 3]
    assert ", too many dimensions in slice" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array[[5, 3, 20, 8]]
    assert " attempting to get 20, index out of range" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array[[5, 3, -20, 8]]
    assert " attempting to get -20, index out of range" in str(excinfo.value)

    array.setidentities()

    with pytest.raises(ValueError) as excinfo:
        array[20]
    assert " attempting to get 20, index out of range" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array[-20]
    assert " attempting to get -20, index out of range" in str(excinfo.value)

    array[-20:20]

    with pytest.raises(ValueError) as excinfo:
        array[20,]
    assert " attempting to get 20, index out of range" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array[-20,]
    assert " attempting to get -20, index out of range" in str(excinfo.value)

    array[-20:20,]

    with pytest.raises(ValueError) as excinfo:
        array[2, 3]
    assert ", too many dimensions in slice" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array[[5, 3, 20, 8]]
    assert " attempting to get 20, index out of range" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array[[5, 3, -20, 8]]
    assert " attempting to get -20, index out of range" in str(excinfo.value)

def test_listarray_numpyarray():
    starts  = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6]))
    stops   = awkward1.layout.Index64(numpy.array([3, 3, 5, 6]))
    content = awkward1.layout.NumpyArray(numpy.arange(10)*1.1)
    with pytest.raises(ValueError) as excinfo:
        array   = awkward1.layout.ListArray64(starts, stops, content)

    starts  = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6]))
    stops   = awkward1.layout.Index64(numpy.array([3, 3, 5, 6, 10]))
    content = awkward1.layout.NumpyArray(numpy.arange(10)*1.1)
    array   = awkward1.layout.ListArray64(starts, stops, content)

    with pytest.raises(ValueError) as excinfo:
        array[20]
    assert " attempting to get 20, index out of range" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array[-20]
    assert " attempting to get -20, index out of range" in str(excinfo.value)

    array[-20:20]

    with pytest.raises(ValueError) as excinfo:
        array[20,]
    assert " attempting to get 20, index out of range" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array[-20,]
    assert " attempting to get -20, index out of range" in str(excinfo.value)

    array[-20:20,]

    with pytest.raises(ValueError) as excinfo:
        array[2, 1, 0]
    assert ", too many dimensions in slice" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array[[2, 0, 0, 20, 3]]
    assert " attempting to get 20, index out of range" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array[[2, 0, 0, -20, 3]]
    assert " attempting to get -20, index out of range" in str(excinfo.value)

    starts  = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6]))
    stops   = awkward1.layout.Index64(numpy.array([3, 3, 5, 6, 10]))
    content = awkward1.layout.NumpyArray(numpy.arange(10)*1.1)
    array   = awkward1.layout.ListArray64(starts, stops, content)

    array.setidentities()

    with pytest.raises(ValueError) as excinfo:
        array[2, 20]
    assert " with identity [2] attempting to get 20, index out of range" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array[2, -20]
    assert " with identity [2] attempting to get -20, index out of range" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array[1:][2, 20]
    assert " with identity [3] attempting to get 20, index out of range" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array[1:][2, -20]
    assert " with identity [3] attempting to get -20, index out of range" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array[2, [1, 0, 0, 20]]
    assert " with identity [2] attempting to get 20, index out of range" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array[2, [1, 0, 0, -20]]
    assert " with identity [2] attempting to get -20, index out of range" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array[1:][2, [0, 20]]
    assert " with identity [3] attempting to get 20, index out of range" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array[1:][2, [0, -20]]
    assert " with identity [3] attempting to get -20, index out of range" in str(excinfo.value)

def test_listarray_listarray_numpyarray():
    content  = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    starts1  = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6]))
    stops1   = awkward1.layout.Index64(numpy.array([3, 3, 5, 6, 9]))
    offsets1 = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 9]))
    starts2  = awkward1.layout.Index64(numpy.array([0, 2, 3, 3]))
    stops2   = awkward1.layout.Index64(numpy.array([2, 3, 3, 5]))
    offsets2 = awkward1.layout.Index64(numpy.array([0, 2, 3, 3, 5]))

    array1 = awkward1.layout.ListArray64(starts1, stops1, content)
    array2 = awkward1.layout.ListArray64(starts2, stops2, array1)

    with pytest.raises(ValueError) as excinfo:
        array2[20]
    assert " attempting to get 20, index out of range" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array2[20,]
    assert " attempting to get 20, index out of range" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array2[2, 20]
    assert " attempting to get 20, index out of range" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array2[-20]
    assert " attempting to get -20, index out of range" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array2[-20,]
    assert " attempting to get -20, index out of range" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array2[2, -20]
    assert " attempting to get -20, index out of range" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array2[1, 0, 20]
    assert " attempting to get 20, index out of range" in str(excinfo.value)

    array2.setidentities()

    with pytest.raises(ValueError) as excinfo:
        array2[20]
    assert " attempting to get 20, index out of range" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array2[20,]
    assert " attempting to get 20, index out of range" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array2[2, 20]
    assert " with identity [2] attempting to get 20, index out of range" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array2[1:][2, 20]
    assert " with identity [3] attempting to get 20, index out of range" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array2[-20]
    assert " attempting to get -20, index out of range" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array2[-20,]
    assert " attempting to get -20, index out of range" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array2[2, -20]
    assert " with identity [2] attempting to get -20, index out of range" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array2[1:][2, -20]
    assert " with identity [3] attempting to get -20, index out of range" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array2[1, 0, 20]
    assert " with identity [1, 0] attempting to get 20, index out of range" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array2[1:][2, 0, 20]
    assert " with identity [3, 0] attempting to get 20, index out of range" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        array2[:, 1:][3, 0, 20]
    assert " with identity [3, 1] attempting to get 20, index out of range" in str(excinfo.value)
