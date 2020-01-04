# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy

import awkward1

py27 = (sys.version_info[0] < 3)

def test_parameters_on_arrays():
    a = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    assert a.parameters == {}
    a.setparameter("one", ["two", 3, {"four": 5}])
    assert a.parameters == {"one": ["two", 3, {"four": 5}]}

    b = awkward1.layout.ListOffsetArray64(awkward1.layout.Index64(numpy.array([0, 3, 3, 5], dtype=numpy.int64)), a)
    assert b.parameters == {}
    assert b.content.parameters == {"one": ["two", 3, {"four": 5}]}
    b.setparameter("what", "ever")
    assert b.parameters == {"what": "ever"}
    assert b.content.parameters == {"one": ["two", 3, {"four": 5}]}

def test_string2():
    content = awkward1.layout.NumpyArray(numpy.array([ord(x) for x in "heythere"], dtype=numpy.uint8), parameters=awkward1.utf8.parameters)
    listoffsetarray = awkward1.layout.ListOffsetArray64(awkward1.layout.Index64(numpy.array([0, 3, 3, 8])), content, parameters=awkward1.string.parameters)
    a = awkward1.Array(listoffsetarray)

    assert isinstance(a, awkward1.Array)
    assert awkward1.tolist(a) == ['hey', '', 'there']

    assert repr(a.type) == "3 * string"
    assert repr(a[0].type) == "3 * utf8"
    assert repr(a[1].type) == "0 * utf8"
    assert repr(a[2].type) == "5 * utf8"

    if py27:
        assert repr(a) == "<Array [u'hey', u'', u'there'] type='3 * string'>"
        assert repr(a[0]) == "u'hey'"
        assert repr(a[1]) == "u''"
        assert repr(a[2]) == "u'there'"
    else:
        assert repr(a) == "<Array ['hey', '', 'there'] type='3 * string'>"
        assert repr(a[0]) == "'hey'"
        assert repr(a[1]) == "''"
        assert repr(a[2]) == "'there'"

def test_dress():
    class Dummy(awkward1.highlevel.Array):
        def __repr__(self):
            return "<Dummy {0}>".format(str(self))
    ns = {"Dummy": Dummy}

    x = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    a = awkward1.Array(x, type=awkward1.layout.ArrayType(x.type, 5, {"__class__": "Dummy", "__str__": "D[5 * float64]"}), namespace=ns)
    assert repr(a) == "<Dummy [1.1, 2.2, 3.3, 4.4, 5.5]>"

    x2 = awkward1.layout.ListOffsetArray64(awkward1.layout.Index64(numpy.array([0, 3, 3, 5], dtype=numpy.int64)), awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5]), parameters={"__class__": "Dummy"}))
    a2 = awkward1.Array(x2, namespace=ns)
    assert repr(a2) == "<Array [<Dummy [1.1, 2.2, 3.3]>, ... ] type='3 * var * float64[parameters={\"__cl...'>"
    assert repr(a2[0]) == "<Dummy [1.1, 2.2, 3.3]>"
    assert repr(a2[1]) == "<Dummy []>"
    assert repr(a2[2]) == "<Dummy [4.4, 5.5]>"

numba = pytest.importorskip("numba")

class D(awkward1.highlevel.Array):
    pass

def test_numpyarray():
    array1 = awkward1.layout.NumpyArray(numpy.arange(2*3*5, dtype=numpy.int64).reshape(2, 3, 5), parameters={"__class__": "D", "__str__": "D[int64]"})

    @numba.njit
    def f1(q):
        return q

    array2 = f1(array1)

    assert repr(array2.type) == "3 * 5 * D[int64]"
    assert repr(array2[0].type) == "5 * D[int64]"
    assert repr(array2[0, 0].type) == "D[int64]"
    assert array2[-1, -1, -1] == 29

def test_regulararray():
    array1 = awkward1.layout.RegularArray(awkward1.layout.NumpyArray(numpy.arange(10, dtype=numpy.int64)), 5, parameters={"__class__": "D", "__str__": "D[5 * int64]"})

    @numba.njit
    def f1(q):
        return q

    array2 = f1(array1)

    assert repr(array2.type) == "D[5 * int64]"

def test_listoffsetarray():
    array1 = awkward1.layout.ListOffsetArray64(awkward1.layout.Index64(numpy.array([0, 3, 3, 5], dtype=numpy.int64)), awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5], dtype=numpy.int64)), parameters={"__class__": "D", "__str__": "D[var * int64]"})

    @numba.njit
    def f1(q):
        return q

    array2 = f1(array1)

    assert repr(array2.type) == "D[var * int64]"

def test_listarray():
    array1 = awkward1.layout.ListArray64(awkward1.layout.Index64(numpy.array([0, 3, 3], dtype=numpy.int64)), awkward1.layout.Index64(numpy.array([3, 3, 5], dtype=numpy.int64)), awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5], dtype=numpy.int64)), parameters={"__class__": "D", "__str__": "D[var * int64]"})

    @numba.njit
    def f1(q):
        return q

    array2 = f1(array1)

    assert repr(array2.type) == "D[var * int64]"

def test_recordarray():
    content1 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3], dtype=numpy.int64))
    content2 = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3], dtype=numpy.float64))
    array1 = awkward1.layout.RecordArray({"one": content1, "two": content2}, parameters={"__class__": "D", "__str__": "D[{\"one\": int64, \"two\": float64}]"})

    @numba.njit
    def f1(q):
        return q

    array2 = f1(array1)

    assert repr(array2.type) in ('D[{"one": int64, "two": float64}]', ' D[{"two": float64, "one": int64}]')
    assert repr(array2[0].type) in ('D[{"one": int64, "two": float64}]', 'D[{"two": float64, "one": int64}]')
