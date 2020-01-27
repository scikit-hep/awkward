# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy

import awkward1

py27 = (sys.version_info[0] < 3)

pandas = pytest.importorskip("pandas")

def test_basic():
    nparray = numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    akarray = awkward1.Array(nparray)
    dfarray = pandas.DataFrame({"x": akarray})

    assert dfarray.x[2] == 2.2

    if not py27:
        # Fails for MacOS and Windows Python 2.7,
        # but I don't care a whole lot about *any* Python 2.7.
        nparray[2] = 999
        assert dfarray.x[2] == 999

def test_interesting():
    akarray = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    dfarray = pandas.DataFrame({"x": akarray})
    dfarray2 = dfarray * 2

    assert isinstance(dfarray2.x.values, awkward1.Array)
    assert awkward1.tolist(dfarray2.x.values) == [[2.2, 4.4, 6.6], [], [8.8, 11]]

    akarray.nbytes == dfarray.x.nbytes



# Not ready to do the full testing suite, yet.

# pandas_tests_extension_base = pytest.importorskip("pandas.tests.extension.base")
# class TestConstructors(pandas_tests_extension_base.BaseConstructorsTests):
#     pass
