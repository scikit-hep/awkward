# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import awkward as ak

STRUCTURED = np.array([(1, 2.5)], dtype=[("a", "i4"), ("b", "f8")])


@pytest.mark.parametrize("zero_d", [False, True])
@pytest.mark.parametrize(
    ("value", "expected_type"),
    [
        (np.longdouble(1.5), "float64"),
        (np.complex64(1.5 + 2.5j), "complex128"),
        (np.datetime64("2020-01-02T03:04:05.000000006", "ns"), "datetime64[ns]"),
        (np.datetime64("NaT", "s"), "datetime64[s]"),
        (np.timedelta64(5, "s"), "timedelta64[s]"),
        (np.timedelta64(5, "ns"), "timedelta64[ns]"),
        (STRUCTURED[0], "{a: int64, b: float64}"),
    ],
)
def test_matches_from_numpy(value, expected_type, zero_d):
    result = ak.from_iter([np.array(value) if zero_d else value])
    assert str(result.type) == f"1 * {expected_type}"
    assert result.to_list() == ak.from_numpy(np.asarray(value).reshape(1)).to_list()


@pytest.mark.parametrize("zero_d", [False, True])
def test_unstructured_void(zero_d):
    value = np.void(b"ab")
    result = ak.from_iter([np.array(value) if zero_d else value])
    assert str(result.type) == "1 * bytes"
    assert result.to_list() == [b"ab"]


def run(code):
    # A stack overflow kills the interpreter, so it must fail this test rather than pytest.
    return subprocess.run(
        [sys.executable, "-c", f"import numpy as np, awkward as ak\n{code}"],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("[np.clongdouble(1.5 + 2.5j)]", "1 * complex128 [(1.5+2.5j)]"),
        ("[np.array(np.clongdouble(1.5 + 2.5j))]", "1 * complex128 [(1.5+2.5j)]"),
        (
            "np.full((2, 1), 1.5 + 2.5j, dtype=np.clongdouble)",
            "2 * var * complex128 [[(1.5+2.5j)], [(1.5+2.5j)]]",
        ),
    ],
)
def test_clongdouble(expr, expected):
    out = run(f"a = ak.from_iter({expr})\nprint(a.type, a.to_list())")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == expected


def test_tolist_returning_same_type():
    out = run(
        "class S:\n"
        "    def tolist(self):\n"
        "        return S()\n"
        "try:\n"
        "    ak.from_iter([S()])\n"
        "except TypeError as err:\n"
        "    print('tolist() returns the same type' in str(err))"
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "True"
