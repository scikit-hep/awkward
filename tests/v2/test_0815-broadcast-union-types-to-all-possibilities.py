# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    @ak._v2.behaviors.mixins.mixin_class(ak._v2.behavior)
    class Point:
        @ak._v2.behaviors.mixins.mixin_class_method(np.add, {"Point"})
        def point_add(self, other):
            return ak._v2.operations.zip(
                {"x": self.x + other.x, "y": self.y + other.y},
                with_name="Point",
            )

    @ak._v2.behaviors.mixins.mixin_class(ak._v2.behavior)
    class Point2(Point):
        pass

    def make(name):
        return ak._v2.operations.zip(
            {
                "x": ak.Array([[1, 2], [3]]),
                "y": ak.Array([[1, 2], [3]]),
            },
            with_name=name,
        )

    a = ak._v2.operations.zip(
        {
            "x": ak.Array([1, 1]),
            "y": ak.Array([1, 1]),
        },
        with_name="Point",
    )
    b, c = make("Point2"), make("Point")

    assert (a + b).tolist() == [
        [{"x": 2, "y": 2}, {"x": 3, "y": 3}],
        [{"x": 4, "y": 4}],
    ]
    assert (b + c).tolist() == [
        [{"x": 2, "y": 2}, {"x": 4, "y": 4}],
        [{"x": 6, "y": 6}],
    ]
    assert (a + c).tolist() == [
        [{"x": 2, "y": 2}, {"x": 3, "y": 3}],
        [{"x": 4, "y": 4}],
    ]
    d = ak._v2.operations.concatenate([b, c], axis=1)
    assert (a + d).tolist() == [
        [{"x": 2, "y": 2}, {"x": 3, "y": 3}, {"x": 2, "y": 2}, {"x": 3, "y": 3}],
        [{"x": 4, "y": 4}, {"x": 4, "y": 4}],
    ]

    e = ak._v2.operations.concatenate([b[b.x < 0], c[c.x < 0]], axis=1)

    assert (a + e).tolist() == [[], []]
