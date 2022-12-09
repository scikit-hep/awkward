# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak

to_list = ak.operations.to_list


def test():
    behavior = {}

    @ak.mixin_class(behavior)
    class Point:
        def distance(self, other):
            return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

        @ak.mixin_class_method(np.equal, {"Point"})
        def point_equal(self, other):
            return np.logical_and(self.x == other.x, self.y == other.y)

        @ak.mixin_class_method(np.abs)
        def point_abs(self):
            return np.sqrt(self.x**2 + self.y**2)

        @ak.mixin_class_method(np.add, {"Point"})
        def point_add(self, other):
            return ak.zip(
                {"x": self.x + other.x, "y": self.y + other.y},
                with_name="Point",
            )

    test = ak.Array(
        [
            [{"x": float(i), "y": float(i)}, {"x": float(i) * 0.5, "y": float(i) * 0.5}]
            for i in range(3)
        ],
        with_name="Point",
        behavior=behavior,
    )

    one = test + test

    assert type(one).__name__ == "PointArray"

    assert one.to_list() == [
        [{"x": 0.0, "y": 0.0}, {"x": 0.0, "y": 0.0}],
        [{"x": 2.0, "y": 2.0}, {"x": 1.0, "y": 1.0}],
        [{"x": 4.0, "y": 4.0}, {"x": 2.0, "y": 2.0}],
    ]

    combos = ak.combinations(test, 2)

    two = combos["0"] + combos["1"]

    assert type(two).__name__ == "PointArray"

    assert two.to_list() == [
        [{"x": 0.0, "y": 0.0}],
        [{"x": 1.5, "y": 1.5}],
        [{"x": 3.0, "y": 3.0}],
    ]
