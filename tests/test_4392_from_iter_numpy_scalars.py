# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import functools
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


# A stack overflow kills the interpreter, so it must fail the test rather than pytest.
needs_subprocess = pytest.mark.skipif(
    sys.platform.startswith("emscripten"), reason="no subprocess on emscripten"
)


def run(code):
    return subprocess.run(
        [sys.executable, "-c", f"import numpy as np, awkward as ak\n{code}"],
        capture_output=True,
        text=True,
        check=False,
    )


@needs_subprocess
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


class Countdown:
    def __init__(self, n):
        self.n = n

    def tolist(self):
        return Countdown(self.n - 1) if self.n else self.n


CYCLES = """
class S:
    def tolist(self):
        return S()
class A:
    def tolist(self):
        return B()
class B:
    def tolist(self):
        return A()
class T:
    def to_list(self):
        return T()
"""


@needs_subprocess
@pytest.mark.parametrize(
    "make_x",
    [
        "x = S()",
        "x = A()",
        "x = T()",
        "x = np.empty((), dtype=object)\nx[()] = x",
        "x = 1\nfor _ in range(200_000):\n    x = [x]",
        "x = 1\nfor _ in range(200_000):\n    x = (x,)",
        "x = 1\nfor _ in range(200_000):\n    x = {'a': x}",
        "import sys\nsys.setrecursionlimit(100_000)\nx = S()",
        "import sys\nsys.setrecursionlimit(100_000)\nx = 1\nfor _ in range(200_000):\n    x = (x,)",
        "import sys\nsys.setrecursionlimit(100_000)\nx = 1\nfor _ in range(200_000):\n    x = {'a': x}",
    ],
)
def test_unbounded_recursion(make_x):
    out = run(
        f"{CYCLES}{make_x}\n"
        "try:\n"
        "    ak.from_iter([x])\n"
        "except RecursionError:\n"
        "    print('RecursionError')"
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "RecursionError"


def test_bounded_recursion():
    assert ak.from_iter([Countdown(50)]).to_list() == [0]
    x = 1
    for _ in range(200):
        x = [x]
    assert str(ak.from_iter(x).type) == "1 * " + "var * " * 199 + "int64"


@pytest.mark.parametrize(
    "value",
    [
        [1, 2, 3],
        (1,),
        {"b": 1, "a": [2, (3,)]},
        [[1, [2, [3]]]],
        ["x" * 100],
        list(range(100)),
        functools.reduce(lambda x, _: {"a": x}, range(100), 1),
    ],
)
def test_argument_text_matches_repr(value):
    text = repr(value)
    expected = text if len(text) <= 72 else text[:69] + "..."
    assert ak._errors.ErrorContext().format_argument(72, value) == expected


def test_argument_whose_repr_raises():
    class Items(dict):
        def items(self):
            raise ValueError

    # reprlib picks its formatter by the type's name (and, in newer Pythons, its module)
    Items.__name__, Items.__module__ = "dict", "builtins"
    assert (
        ak._errors.ErrorContext().format_argument(72, Items(a=1))
        == "repr-raised-ValueError"
    )
