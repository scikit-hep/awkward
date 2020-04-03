# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1
import awkward1._connect._pandas

py27 = (sys.version_info[0] < 3)

pandas = pytest.importorskip("pandas")

def test_basic():
    nparray = numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    akarray = awkward1.Array(nparray, check_valid=True)
    dfarray = pandas.DataFrame({"x": akarray})

    assert dfarray.x[2] == 2.2

    if not py27:
        # Fails for MacOS and Windows Python 2.7,
        # but I don't care a whole lot about *any* Python 2.7.
        nparray[2] = 999
        assert dfarray.x[2] == 999

def test_interesting():
    akarray = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], check_valid=True)
    dfarray = pandas.DataFrame({"x": akarray})
    dfarray2 = dfarray * 2

    assert isinstance(dfarray2.x.values, awkward1.Array)
    assert awkward1.to_list(dfarray2.x.values) == [[2.2, 4.4, 6.6], [], [8.8, 11]]

    akarray.nbytes == dfarray.x.nbytes

    assert awkward1.to_list(akarray) == awkward1.to_list(akarray.copy())

    assert awkward1.to_list(akarray.take([2, 0, 0, 1, -1, 2])) == [[4.4, 5.5], [1.1, 2.2, 3.3], [1.1, 2.2, 3.3], [], [4.4, 5.5], [4.4, 5.5]]
    assert awkward1.to_list(akarray.take([2, 0, 0, 1, -1, 2], allow_fill=True)) == [[4.4, 5.5], [1.1, 2.2, 3.3], [1.1, 2.2, 3.3], [], None, [4.4, 5.5]]
    assert awkward1.to_list(akarray.take([2, 0, 0, 1, -1, 2], allow_fill=True, fill_value=999)) == [[4.4, 5.5], [1.1, 2.2, 3.3], [1.1, 2.2, 3.3], [], 999, [1.1, 2.2, 3.3]]

def test_constructor():
    akarray = awkward1.Array([[1.1, 2.2, 3.3], [], None, [4.4, 5.5]], check_valid=True)
    dfarray = pandas.DataFrame(akarray)
    assert isinstance(dfarray[dfarray.columns[0]].values, awkward1.Array)
    assert awkward1.to_list(dfarray[dfarray.columns[0]].values) == [[1.1, 2.2, 3.3], [], None, [4.4, 5.5]]

    akarray = awkward1.Array([[{"x": 1, "y": 1.1}, {"x": 2, "y": 2}, {"x": 3, "y": 3.3}], [], None, [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}]], check_valid=True)
    dfarray = pandas.DataFrame(akarray)
    assert isinstance(dfarray["x"].values, awkward1.Array)
    assert awkward1.to_list(dfarray["x"].values) == [[1, 2, 3], [], None, [4, 5]]

pandas_tests_extension_base = pytest.importorskip("pandas.tests.extension.base")

@pytest.fixture
def dtype():
    """A fixture providing the ExtensionDtype to validate."""
    return awkward1._connect._pandas.get_dtype()()

@pytest.fixture
def data():
    """Length-100 array for this type.
    * data[0] and data[1] should both be non missing
    * data[0] and data[1] should not be equal
    """
    out = [list(range(i)) for i in range(100)]
    for i in range(10, 99):
        out[i] = None
    return awkward1.Array(out, check_valid=True)

@pytest.fixture
def data_for_twos():
    """Length-100 array in which all the elements are two."""
    raise NotImplementedError

@pytest.fixture
def data_missing():
    """Length-2 array with [NA, Valid]"""
    raise NotImplementedError

@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    """Parametrized fixture giving 'data' and 'data_missing'"""
    if request.param == "data":
        return data
    elif request.param == "data_missing":
        return data_missing

@pytest.fixture
def data_repeated(data):
    """
    Generate many datasets.
    Parameters
    ----------
    data : fixture implementing `data`
    Returns
    -------
    Callable[[int], Generator]:
        A callable that takes a `count` argument and
        returns a generator yielding `count` datasets.
    """

    def gen(count):
        for _ in range(count):
            yield data

    return gen

@pytest.fixture
def data_for_sorting():
    """Length-3 array with a known sort order.
    This should be three items [B, C, A] with
    A < B < C
    """
    raise NotImplementedError

@pytest.fixture
def data_missing_for_sorting():
    """Length-3 array with a known sort order.
    This should be three items [B, NA, A] with
    A < B and NA missing.
    """
    raise NotImplementedError

@pytest.fixture
def na_cmp():
    """Binary operator for comparing NA values.
    Should return a function of two arguments that returns
    True if both arguments are (scalar) NA for your type.
    By default, uses ``operator.is_``
    """
    return operator.is_

@pytest.fixture
def na_value():
    """The scalar missing value for this type. Default 'None'"""
    return None

@pytest.fixture
def data_for_grouping():
    """Data for factorization, grouping, and unique tests.
    Expected to be like [B, B, NA, NA, A, A, B, C]
    Where A < B < C and NA is missing
    """
    raise NotImplementedError

@pytest.fixture(params=[True, False])
def box_in_series(request):
    """Whether to box the data in a Series"""
    return request.param

@pytest.fixture(
    params=[
        lambda x: 1,
        lambda x: [1] * len(x),
        lambda x: pandas.Series([1] * len(x)),
        lambda x: x,
    ],
    ids=["scalar", "list", "series", "object"],
)
def groupby_apply_op(request):
    """
    Functions to test groupby.apply().
    """
    return request.param

@pytest.fixture(params=[True, False])
def as_frame(request):
    """
    Boolean fixture to support Series and Series.to_frame() comparison testing.
    """
    return request.param

@pytest.fixture(params=[True, False])
def as_series(request):
    """
    Boolean fixture to support arr and Series(arr) comparison testing.
    """
    return request.param

@pytest.fixture(params=[True, False])
def use_numpy(request):
    """
    Boolean fixture to support comparison testing of ExtensionDtype array
    and numpy array.
    """
    return request.param

@pytest.fixture(params=["ffill", "bfill"])
def fillna_method(request):
    """
    Parametrized fixture giving method parameters 'ffill' and 'bfill' for
    Series.fillna(method=<method>) testing.
    """
    return request.param

@pytest.fixture(params=[True, False])
def as_array(request):
    """
    Boolean fixture to support ExtensionDtype _from_sequence method testing.
    """
    return request.param

class TestConstructors(pandas_tests_extension_base.BaseConstructorsTests):
    # old version of the test
    def test_from_dtype(self, data):
        dtype = data.dtype

        expected = pandas.Series(data)
        result = pandas.Series(list(data), dtype=dtype)
        self.assert_series_equal(result, expected)

        result = pandas.Series(list(data), dtype=str(dtype))
        self.assert_series_equal(result, expected)

    def test_pandas_array(self, data):
        pass
