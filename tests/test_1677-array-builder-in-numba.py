# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

nb = pytest.importorskip("numba")

ak_numba_arrayview = pytest.importorskip("awkward._connect.numba.arrayview")
ak_numba_builder = pytest.importorskip("awkward._connect.numba.builder")

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

    builder = add_a_complex(builder, 2)
    out = builder.snapshot()
    assert out.to_list() == [(1.0 + 0.1j), (2.0 + 0.2j), (2.0 + 0j)]

    builder = add_a_complex(builder, 2.0)
    out = builder.snapshot()
    assert out.to_list() == [(1.0 + 0.1j), (2.0 + 0.2j), (2.0 + 0j), (2.0 + 0j)]


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


def test_ArrayBuilder_append():
    @nb.njit
    def append(builder, value):
        builder.append(value)
        return builder

    builder = append(ak.ArrayBuilder(), True)
    out = builder.snapshot()
    assert out.to_list() == [True]

    builder = append(builder, 1)
    out = builder.snapshot()
    assert out.to_list() == [True, 1]

    builder = append(builder, 1.1)
    out = builder.snapshot()
    assert out.to_list() == [True, 1, 1.1]

    z = 1.1 + 0.1j
    builder = append(builder, z)
    out = builder.snapshot()
    assert out.to_list() == [True, 1, 1.1, (1.1 + 0.1j)]

    builder = append(builder, np.datetime64("2020-09-04"))
    out = builder.snapshot()
    assert out.to_list() == [True, 1, 1.1, (1.1 + 0.1j), np.datetime64("2020-09-04")]

    builder = append(builder, np.timedelta64(5, "s"))
    out = builder.snapshot()
    assert out.to_list() == [
        True,
        1,
        1.1,
        (1.1 + 0.1j),
        np.datetime64("2020-09-04"),
        np.timedelta64(5, "s"),
    ]

    builder = append(builder, "hello")
    out = builder.snapshot()
    assert out.to_list() == [
        True,
        1,
        1.1,
        (1.1 + 0.1j),
        np.datetime64("2020-09-04"),
        np.timedelta64(5, "s"),
        "hello",
    ]

    builder = append(
        builder, b"arrow \xe2\x86\x92 zero \x00 not the end!".decode("utf-8")
    )
    out = builder.snapshot()
    assert out.to_list() == [
        True,
        1,
        1.1,
        (1.1 + 0.1j),
        np.datetime64("2020-09-04"),
        np.timedelta64(5, "s"),
        "hello",
        b"arrow \xe2\x86\x92 zero \x00 not the end!".decode("utf-8"),
    ]
