# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak  # noqa: F401
from awkward._nplikes import dtypes, typetracer


def is_true_or_unknown(x):
    if typetracer.is_unknown_scalar(x):
        return True
    else:
        return x


def test_broadcast_shapes():
    nplike: typetracer.TypeTracer = typetracer.TypeTracer.instance()

    assert is_true_or_unknown(
        nplike.shapes_are_compatible(
            nplike.broadcast_shapes((1, 2, 3), (2, 3)), (1, 2, 3)
        )
    )
    assert is_true_or_unknown(
        nplike.shapes_are_compatible(
            nplike.broadcast_shapes(
                (1, typetracer.unknown_scalar(dtypes.int64), 3),
                (2, 3),
            ),
            (1, typetracer.unknown_scalar(dtypes.int64), 3),
        )
    )
    assert is_true_or_unknown(
        nplike.shapes_are_compatible(
            nplike.broadcast_shapes(
                (1, typetracer.unknown_scalar(dtypes.int64), 3),
                (3,),
            ),
            (1, typetracer.unknown_scalar(dtypes.int64), 3),
        )
    )
    assert is_true_or_unknown(
        nplike.shapes_are_compatible(
            nplike.broadcast_shapes(
                (typetracer.unknown_scalar(dtypes.int64), 2, 3),
                (2, 3),
            ),
            (typetracer.unknown_scalar(dtypes.int64), 2, 3),
        )
    )
    assert is_true_or_unknown(
        nplike.shapes_are_compatible(
            nplike.broadcast_shapes(
                (typetracer.unknown_scalar(dtypes.int64),),
                (2,),
            ),
            (typetracer.unknown_scalar(dtypes.int64),),
        )
    )
    assert is_true_or_unknown(
        nplike.shapes_are_compatible(
            nplike.broadcast_shapes(
                (
                    typetracer.unknown_scalar(dtypes.int64),
                    typetracer.unknown_scalar(dtypes.int64),
                ),
                (1,),
            ),
            (
                typetracer.unknown_scalar(dtypes.int64),
                typetracer.unknown_scalar(dtypes.int64),
            ),
        )
    )


def test_typetracer_shapes():
    nplike: typetracer.TypeTracer = typetracer.TypeTracer.instance()

    shape = (1, 2, 3)
    assert is_true_or_unknown(nplike.shapes_are_compatible(shape, (1, 2, 3)))
    assert is_true_or_unknown(
        nplike.shapes_are_compatible(
            shape, (1, typetracer.unknown_scalar(dtypes.int64), 3)
        )
    )
    assert is_true_or_unknown(
        nplike.shapes_are_compatible(
            (1, typetracer.unknown_scalar(dtypes.int64), 3), shape
        )
    )
    assert is_true_or_unknown(
        nplike.shapes_are_compatible(
            shape,
            (
                typetracer.unknown_scalar(dtypes.int64),
                typetracer.unknown_scalar(dtypes.int64),
                typetracer.unknown_scalar(dtypes.int64),
            ),
        )
    )
    assert is_true_or_unknown(
        nplike.shapes_are_compatible(
            shape,
            (
                typetracer.unknown_scalar(dtypes.int64),
                2,
                typetracer.unknown_scalar(dtypes.int64),
            ),
        )
    )

    assert not (
        is_true_or_unknown(
            nplike.shapes_are_compatible(
                shape,
                (
                    typetracer.unknown_scalar(dtypes.int64),
                    1,
                    typetracer.unknown_scalar(dtypes.int64),
                ),
            )
        )
    )
    assert not (
        is_true_or_unknown(
            nplike.shapes_are_compatible(
                shape,
                (
                    typetracer.unknown_scalar(dtypes.int64),
                    typetracer.unknown_scalar(dtypes.int64),
                    8,
                ),
            )
        )
    )
    assert not (
        is_true_or_unknown(
            nplike.shapes_are_compatible(
                shape, (typetracer.unknown_scalar(dtypes.int64), 3)
            )
        )
    )
