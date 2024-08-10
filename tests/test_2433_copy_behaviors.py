# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy
import pytest

import awkward as ak


def test():
    class SuperVector:
        def add(self, other):
            """Add two vectors together elementwise using `x` and `y` components"""
            return ak.zip(
                {"x": self.x + other.x, "y": self.y + other.y},
                with_name="VectorTwoD",
                behavior=self.behavior,
            )

    # first sub-class
    @ak.mixin_class(ak.behavior)
    class VectorTwoD(SuperVector):
        def __eq__(self, other):
            return ak.all(self.x == other.x) and ak.all(self.y == other.y)

    v = ak.Array(
        [
            [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}],
            [],
            [{"x": 3, "y": 3.3}],
            [
                {"x": 4, "y": 4.4},
                {"x": 5, "y": 5.5},
                {"x": 6, "y": 6.6},
            ],
        ],
        with_name="VectorTwoD",
        behavior=ak.behavior,
    )
    v_added = ak.Array(
        [
            [{"x": 2, "y": 2.2}, {"x": 4, "y": 4.4}],
            [],
            [{"x": 6, "y": 6.6}],
            [
                {"x": 8, "y": 8.8},
                {"x": 10, "y": 11},
                {"x": 12, "y": 13.2},
            ],
        ],
        with_name="VectorTwoD",
        behavior=ak.behavior,
    )

    # add method works but the binary operator does not
    assert v.add(v) == v_added
    with pytest.raises(TypeError):
        v + v

    # registering the operator makes everything work
    ak.behavior[numpy.add, "VectorTwoD", "VectorTwoD"] = lambda v1, v2: v1.add(v2)
    assert v + v == v_added

    # instead of registering every operator again, just copy the behaviors of
    # another class to this class
    ak.behavior.update(
        ak._util.copy_behaviors("VectorTwoD", "VectorTwoDAgain", ak.behavior)
    )

    # second sub-class
    @ak.mixin_class(ak.behavior)
    class VectorTwoDAgain(VectorTwoD):
        pass

    v = ak.Array(
        [
            [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}],
            [],
            [{"x": 3, "y": 3.3}],
            [
                {"x": 4, "y": 4.4},
                {"x": 5, "y": 5.5},
                {"x": 6, "y": 6.6},
            ],
        ],
        with_name="VectorTwoDAgain",
        behavior=ak.behavior,
    )
    assert v.add(v) == v_added
    assert v + v == v_added

    # instead of registering every operator again, just copy the behaviors of
    # another class to this class
    ak.behavior.update(
        ak._util.copy_behaviors("VectorTwoDAgain", "VectorTwoDAgainAgain", ak.behavior)
    )

    # third sub-class
    @ak.mixin_class(ak.behavior)
    class VectorTwoDAgainAgain(VectorTwoDAgain):
        pass

    v = ak.Array(
        [
            [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}],
            [],
            [{"x": 3, "y": 3.3}],
            [
                {"x": 4, "y": 4.4},
                {"x": 5, "y": 5.5},
                {"x": 6, "y": 6.6},
            ],
        ],
        with_name="VectorTwoDAgainAgain",
        behavior=ak.behavior,
    )
    assert v.add(v) == v_added
    assert v + v == v_added
