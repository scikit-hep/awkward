# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak  # noqa: F401
from awkward.nplikes import typetracer

to_list = ak.operations.to_list


def test_typetracer_shapes():
    shape = (1, 2, 3)
    assert typetracer.shapes_are_compatible(shape, (1, 2, 3))
    assert typetracer.shapes_are_compatible(shape, (1, typetracer.unknown_value, 3))
    assert typetracer.shapes_are_compatible(
        shape,
        (typetracer.unknown_value, typetracer.unknown_value, typetracer.unknown_value),
    )
    assert typetracer.shapes_are_compatible(
        shape,
        (typetracer.unknown_value, 2, typetracer.unknown_value),
    )
    assert not typetracer.shapes_are_compatible(
        shape,
        (typetracer.unknown_value, 1, typetracer.unknown_value),
    )
    assert not typetracer.shapes_are_compatible(
        shape,
        (typetracer.unknown_value, typetracer.unknown_value, 8),
    )
    assert not typetracer.shapes_are_compatible(
        shape,
        (typetracer.unknown_value, 3),
    )
