# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    builder = ak.ArrayBuilder()
    with builder.list():
        with builder.record():
            builder.field("x").integer(1)
            builder.field("y").real(1.1)
        with builder.record():
            builder.field("x").integer(2)
            builder.field("y").real(2.2)
        with builder.record():
            builder.field("x").integer(3)
            builder.field("y").real(3.3)
    with builder.list():
        pass
    with builder.list():
        with builder.record():
            builder.field("x").integer(4)
            builder.field("y").real(4.4)
        with builder.record():
            builder.field("x").integer(5)
            builder.field("y").real(5.5)
            builder.field("z").string("five")
    with builder.list():
        with builder.record():
            builder.field("x").integer(6)
            builder.field("y").real(6.6)
            builder.field("z").string("six")
    with builder.list():
        with builder.record():
            builder.field("x").integer(7)
            builder.field("y").real(7.7)
        with builder.record():
            builder.field("x").integer(8)
            builder.field("y").real(8.8)
    assert ak.to_list(builder) == [
        [
            {"x": 1, "y": 1.1, "z": None},
            {"x": 2, "y": 2.2, "z": None},
            {"x": 3, "y": 3.3, "z": None},
        ],
        [],
        [{"x": 4, "y": 4.4, "z": None}, {"x": 5, "y": 5.5, "z": "five"}],
        [{"x": 6, "y": 6.6, "z": "six"}],
        [{"x": 7, "y": 7.7, "z": None}, {"x": 8, "y": 8.8, "z": None}],
    ]
