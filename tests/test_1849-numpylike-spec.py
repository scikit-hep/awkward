# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak  # noqa: F401
from awkward._nplikes import dtypes, typetracer

to_list = ak.operations.to_list


def unknown(dtype, traits=frozenset()):
    nplike = typetracer.TypeTracer.instance()
    return typetracer.TypeTracerArray._new(
        np.zeros(1, dtype=dtype), (), nplike=nplike, traits=traits
    )


def test_broadcast_shapes():
    assert typetracer.broadcast_shapes((1, 2, 3), (2, 3)) == (1, 2, 3)
    assert typetracer.shapes_are_compatible(
        typetracer.broadcast_shapes(
            (1, unknown(dtypes.int64), 3),
            (2, 3),
        ),
        (1, unknown(dtypes.int64), 3),
    )
    assert typetracer.shapes_are_compatible(
        typetracer.broadcast_shapes(
            (1, unknown(dtypes.int64), 3),
            (3,),
        ),
        (1, unknown(dtypes.int64), 3),
    )
    assert typetracer.shapes_are_compatible(
        typetracer.broadcast_shapes(
            (unknown(dtypes.int64), 2, 3),
            (2, 3),
        ),
        (unknown(dtypes.int64), 2, 3),
    )
    assert typetracer.shapes_are_compatible(
        typetracer.broadcast_shapes(
            (unknown(dtypes.int64),),
            (2,),
        ),
        (unknown(dtypes.int64),),
    )
    assert typetracer.shapes_are_compatible(
        typetracer.broadcast_shapes(
            (
                unknown(dtypes.int64),
                unknown(dtypes.int64),
            ),
            (1,),
        ),
        (
            unknown(dtypes.int64),
            unknown(dtypes.int64),
        ),
    )


def test_typetracer_shapes():
    shape = (1, 2, 3)
    assert typetracer.shapes_are_compatible(shape, (1, 2, 3))
    assert typetracer.shapes_are_compatible(shape, (1, unknown(dtypes.int64), 3))
    assert typetracer.shapes_are_compatible(
        shape,
        (unknown(dtypes.int64), unknown(dtypes.int64), unknown(dtypes.int64)),
    )
    assert typetracer.shapes_are_compatible(
        shape,
        (unknown(dtypes.int64), 2, unknown(dtypes.int64)),
    )
    assert not typetracer.shapes_are_compatible(
        shape,
        (unknown(dtypes.int64), 1, unknown(dtypes.int64)),
    )
    assert not typetracer.shapes_are_compatible(
        shape,
        (unknown(dtypes.int64), unknown(dtypes.int64), 8),
    )
    assert not typetracer.shapes_are_compatible(
        shape,
        (unknown(dtypes.int64), 3),
    )
