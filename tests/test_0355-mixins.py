# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import numpy as np

import awkward as ak

to_list = ak.operations.to_list


def test_make_mixins():
    behavior = {}

    @ak.behaviors.mixins.mixin_class(behavior)
    class Point:
        def distance(self, other):
            return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

        @ak.behaviors.mixins.mixin_class_method(np.equal, {"Point"})
        def point_equal(self, other):
            return np.logical_and(self.x == other.x, self.y == other.y)

        @ak.behaviors.mixins.mixin_class_method(np.abs)
        def point_abs(self):
            return np.sqrt(self.x**2 + self.y**2)

        @ak.behaviors.mixins.mixin_class_method(np.add, {"Point"})
        def point_add(self, other):
            return ak.operations.zip(
                {"x": self.x + other.x, "y": self.y + other.y},
                with_name="Point",
            )

    @ak.behaviors.mixins.mixin_class(behavior)
    class WeightedPoint(Point):
        @ak.behaviors.mixins.mixin_class_method(np.equal, {"WeightedPoint"})
        def weighted_equal(self, other):
            return np.logical_and(self.point_equal(other), self.weight == other.weight)

        @ak.behaviors.mixins.mixin_class_method(np.add, {"WeightedPoint"})
        def weighted_add(self, other):
            sumw = self.weight + other.weight
            return ak.operations.zip(
                {
                    "x": (self.x * self.weight + other.x * other.weight) / sumw,
                    "y": (self.y * self.weight + other.y * other.weight) / sumw,
                    "weight": sumw,
                },
                with_name="WeightedPoint",
            )

    one = ak.Array(
        [
            [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}],
            [],
            [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}],
        ],
        with_name="Point",
        behavior=behavior,
    )
    two = ak.Array(
        [
            [{"x": 0.9, "y": 1}, {"x": 2, "y": 2.2}, {"x": 2.9, "y": 3}],
            [],
            [{"x": 3.9, "y": 4}, {"x": 5, "y": 5.5}],
        ],
        with_name="Point",
        behavior=behavior,
    )
    wone = ak.Array(
        ak.operations.with_field(one, abs(one), "weight"),
        with_name="WeightedPoint",
        behavior=behavior,
    )
    wtwo = ak.Array(
        ak.operations.with_field(two, abs(two), "weight"),
        with_name="WeightedPoint",
        behavior=behavior,
    )
    assert ak._util.arrays_approx_equal(
        one + wone,
        ak.Array(
            [
                [{"x": 2, "y": 2.2}, {"x": 4, "y": 4.4}, {"x": 6, "y": 6.6}],
                [],
                [{"x": 8, "y": 8.8}, {"x": 10, "y": 11.0}],
            ],
            behavior=behavior,
            with_name="Point",
        ),
        dtype_exact=False,
    )
    assert ak._util.arrays_approx_equal(
        wone + wtwo,
        ak.Array(
            [
                [
                    {
                        "x": 0.9524937500390619,
                        "y": 1.052493750039062,
                        "weight": 2.831969279439222,
                    },
                    {"x": 2.0, "y": 2.2, "weight": 5.946427498927402},
                    {
                        "x": 2.9516640394605282,
                        "y": 3.1549921183815837,
                        "weight": 8.632349833200564,
                    },
                ],
                [],
                [
                    {
                        "x": 3.9515600270076154,
                        "y": 4.206240108030463,
                        "weight": 11.533018588312771,
                    },
                    {"x": 5.0, "y": 5.5, "weight": 14.866068747318506},
                ],
            ],
            behavior=behavior,
            with_name="WeightedPoint",
        ),
        dtype_exact=False,
    )
    assert ak._util.arrays_approx_equal(
        abs(one),
        ak.Array(
            [
                [1.4866068747318506, 2.973213749463701, 4.459820624195552],
                [],
                [5.946427498927402, 7.433034373659253],
            ],
            behavior=behavior,
            with_name="Point",
        ),
        dtype_exact=False,
    )
    assert ak._util.arrays_approx_equal(
        one.distance(wtwo),
        [
            [0.14142135623730953, 0.0, 0.31622776601683783],
            [],
            [0.4123105625617664, 0.0],
        ],
        dtype_exact=False,
        check_parameters=False,
    )
