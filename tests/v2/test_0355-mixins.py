# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

to_list = ak._v2.operations.to_list


def test_make_mixins():
    @ak._v2.behaviors.mixins.mixin_class(ak._v2.behavior)
    class Point:
        def distance(self, other):
            return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

        @ak._v2.behaviors.mixins.mixin_class_method(np.equal, {"Point"})
        def point_equal(self, other):
            return np.logical_and(self.x == other.x, self.y == other.y)

        @ak._v2.behaviors.mixins.mixin_class_method(np.abs)
        def point_abs(self):
            return np.sqrt(self.x**2 + self.y**2)

        @ak._v2.behaviors.mixins.mixin_class_method(np.add, {"Point"})
        def point_add(self, other):
            return ak._v2.operations.zip(
                {"x": self.x + other.x, "y": self.y + other.y},
                with_name="Point",
            )

    @ak._v2.behaviors.mixins.mixin_class(ak._v2.behavior)
    class WeightedPoint(Point):
        @ak._v2.behaviors.mixins.mixin_class_method(np.equal, {"WeightedPoint"})
        def weighted_equal(self, other):
            return np.logical_and(self.point_equal(other), self.weight == other.weight)

        @ak._v2.behaviors.mixins.mixin_class_method(np.add, {"WeightedPoint"})
        def weighted_add(self, other):
            sumw = self.weight + other.weight
            return ak._v2.operations.zip(
                {
                    "x": (self.x * self.weight + other.x * other.weight) / sumw,
                    "y": (self.y * self.weight + other.y * other.weight) / sumw,
                    "weight": sumw,
                },
                with_name="WeightedPoint",
            )

    one = ak._v2.Array(
        [
            [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}],
            [],
            [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}],
        ],
        with_name="Point",
    )
    wone = ak._v2.Array(
        ak._v2.operations.with_field(one, abs(one), "weight"),
        with_name="WeightedPoint",
    )

    assert to_list(one + wone) == [
        [{"x": 2, "y": 2.2}, {"x": 4, "y": 4.4}, {"x": 6, "y": 6.6}],
        [],
        [{"x": 8, "y": 8.8}, {"x": 10, "y": 11.0}],
    ]
