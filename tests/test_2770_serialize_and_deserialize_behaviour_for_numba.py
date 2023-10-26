# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest

import awkward as ak

numba = pytest.importorskip("numba")


def test_ArrayBuilder_behavior():
    SOME_ATTRS = {"FOO": "BAR"}
    builder = ak.ArrayBuilder(behavior=SOME_ATTRS)

    @numba.njit
    def func(array):
        return array

    assert builder.behavior is SOME_ATTRS
    assert func(builder).behavior is SOME_ATTRS

    def make_add_xyr():
        def add_xyr(left, right):
            x = left.x + right.x
            y = left.y + right.y
            return ak.zip(
                {
                    "x": x,
                    "y": y,
                    "r": np.sqrt(x**2 + y**2),
                },
                with_name="xyr",
            )

        return add_xyr


    behavior = {(np.add, "xyr", "xyr"): make_add_xyr()}


    a = ak.Array([{"x": 3, "y": 4, "r": 5}], with_name="xyr", behavior=behavior)
    b = ak.Array([{"x": 3, "y": 4, "r": 5}], with_name="xyr", behavior=behavior)

    builder = ak.ArrayBuilder(behavior=behavior)
    assert func(builder).behavior is behavior
