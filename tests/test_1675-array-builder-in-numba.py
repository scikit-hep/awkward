# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward._v2 as ak  # noqa: F401

nb = pytest.importorskip("numba")

ak_numba_arrayview = pytest.importorskip("awkward._v2._connect.numba.arrayview")
ak_numba_builder = pytest.importorskip("awkward._v2._connect.numba.builder")

ak.numba.register_and_check()


def test_ArrayBuilder_of_booleans():
    @nb.njit
    def add_a_boolean(builder, boolean):
        builder.boolean(boolean)
        return builder

    builder = add_a_boolean(ak.ArrayBuilder(), True)
    out = builder.snapshot()
    assert out.to_list() == [True]


def test_ArrayBuilder_of_integers():
    @nb.njit
    def add_an_integer(builder, integer):
        builder.integer(integer)
        return builder

    builder = add_an_integer(ak.ArrayBuilder(), 1)
    out = builder.snapshot()
    assert out.to_list() == [1]


def test_ArrayBuilder_of_reals():
    @nb.njit
    def add_a_real(builder, real):
        builder.real(real)
        return builder

    builder = add_a_real(ak.ArrayBuilder(), 1.1)
    out = builder.snapshot()
    assert out.to_list() == [1.1]


def test_ArrayBuilder_of_complex():
    @nb.njit
    def add_a_complex(builder, complex):
        builder.complex(complex)
        return builder

    builder = add_a_complex(ak.ArrayBuilder(), 1.0 + 0.1j)
    out = builder.snapshot()
    assert out.to_list() == [1.0 + 0.1j]

    builder = add_a_complex(builder, 2.0 + 0.2j)
    out = builder.snapshot()
    assert out.to_list() == [(1.0 + 0.1j), (2.0 + 0.2j)]


def test_ArrayBuilder_of_datetimes():
    @nb.njit
    def add_a_datetime(builder, datetime):
        builder.datetime(datetime)
        return builder

    builder = add_a_datetime(ak.ArrayBuilder(), np.datetime64("2020-09-04"))
    out = builder.snapshot()
    assert out.to_list() == [np.datetime64("2020-09-04")]


def test_ArrayBuilder_of_timedeltas():
    @nb.njit
    def add_a_timedelta(builder, timedelta):
        builder.timedelta(timedelta)
        return builder

    builder = add_a_timedelta(ak.ArrayBuilder(), np.timedelta64(5, "s"))
    out = builder.snapshot()
    assert out.to_list() == [np.timedelta64(5, "s")]


def test_ArrayBuilder_of_strings():
    @nb.njit
    def add_a_string(builder, string):
        builder.string(string)
        return builder

    builder = add_a_string(ak.ArrayBuilder(), "hello")
    builder = add_a_string(builder, "world")
    out = builder.snapshot()
    assert out.to_list() == ["hello", "world"]


def test_ArrayBuilder_of_bytestrings():
    @nb.njit(debug=True)
    def add_a_bytestring(builder, bytestring):
        builder.bytestring(bytestring)
        return builder

    builder = add_a_bytestring(ak.ArrayBuilder(), b"hello\0world")
    out = builder.snapshot()
    assert out.to_list() == [b"hello\0world"]
