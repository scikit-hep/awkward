# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pickle

import numpy as np
import pytest  # noqa: F401

import awkward as ak

behavior = {}


@ak.mixin_class(behavior, "vector")
class Vector:
    @ak.mixin_class_method(np.add, set())
    def add(self, other):
        return ak.zip(
            {"x": self.x + other.x, "y": self.y + other.y, "z": self.z + other.z},
            with_name="vector",
        )


def test():
    vec = ak.Array(
        [
            {"x": 1, "y": 2, "z": 3},
            {"x": 4, "y": 5, "z": 9},
        ],
        with_name="vector",
        behavior=behavior,
    )

    assert ak.almost_equal(
        (vec + vec),
        ak.Array(
            [
                {"x": 2, "y": 4, "z": 6},
                {"x": 8, "y": 10, "z": 18},
            ],
            with_name="vector",
            behavior=behavior,
        ),
    )

    assert ak.almost_equal(pickle.loads(pickle.dumps(vec)), vec)
