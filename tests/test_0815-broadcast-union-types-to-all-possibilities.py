# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    @ak.behaviors.mixins.mixin_class(ak.behavior)
    class Point:
        @ak.behaviors.mixins.mixin_class_method(np.add, {"Point"})
        def point_add(self, other):
            return ak.operations.zip(
                {"x": self.x + other.x, "y": self.y + other.y},
                with_name="Point",
            )

    @ak.behaviors.mixins.mixin_class(ak.behavior)
    class Point2(Point):
        pass

    def make(name):
        return ak.operations.zip(
            {
                "x": ak.Array([[1, 2], [3]]),
                "y": ak.Array([[1, 2], [3]]),
            },
            with_name=name,
        )

    a = ak.operations.zip(
        {
            "x": ak.Array([1, 1]),
            "y": ak.Array([1, 1]),
        },
        with_name="Point",
    )
    b, c = make("Point2"), make("Point")

    assert (a + b).to_list() == [
        [{"x": 2, "y": 2}, {"x": 3, "y": 3}],
        [{"x": 4, "y": 4}],
    ]
    assert (b + c).to_list() == [
        [{"x": 2, "y": 2}, {"x": 4, "y": 4}],
        [{"x": 6, "y": 6}],
    ]
    assert (a + c).to_list() == [
        [{"x": 2, "y": 2}, {"x": 3, "y": 3}],
        [{"x": 4, "y": 4}],
    ]
    d = ak.operations.concatenate([b, c], axis=1)
    assert (a + d).to_list() == [
        [{"x": 2, "y": 2}, {"x": 3, "y": 3}, {"x": 2, "y": 2}, {"x": 3, "y": 3}],
        [{"x": 4, "y": 4}, {"x": 4, "y": 4}],
    ]

    e = ak.operations.concatenate([b[b.x < 0], c[c.x < 0]], axis=1)

    assert (a + e).to_list() == [[], []]
