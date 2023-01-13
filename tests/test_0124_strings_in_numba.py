# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest

import awkward as ak

numba = pytest.importorskip("numba")

ak.numba.register_and_check()


def test_string():
    array = ak.highlevel.Array(["one", "two", "three", "four", "five"])

    def f1(x, i):
        return x[i]

    assert f1(array, 0) == "one"
    assert f1(array, 1) == "two"
    assert f1(array, 2) == "three"

    f1 = numba.njit(f1)

    assert f1(array, 0) == "one"
    assert f1(array, 1) == "two"
    assert f1(array, 2) == "three"

    def f2(x, i, j):
        return x[i] + x[j]

    assert f2(array, 1, 3) == "twofour"
    assert numba.njit(f2)(array, 1, 3) == "twofour"


def test_bytestring():
    array = ak.highlevel.Array([b"one", b"two", b"three", b"four", b"five"])

    def f1(x, i):
        return x[i]

    assert f1(array, 0) == b"one"
    assert f1(array, 1) == b"two"
    assert f1(array, 2) == b"three"

    f1 = numba.njit(f1)

    assert f1(array, 0) == b"one"
    assert f1(array, 1) == b"two"
    assert f1(array, 2) == b"three"
