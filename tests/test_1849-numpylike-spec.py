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
    assert is_true_or_unknown(
        typetracer.broadcast_shapes((1, 2, 3), (2, 3)) == (1, 2, 3)
    )
    assert is_true_or_unknown(
        typetracer.broadcast_shapes(
            (1, typetracer.unknown_value(dtypes.int64), 3),
            (2, 3),
        )
        == (1, typetracer.unknown_value(dtypes.int64), 3)
    )
    assert is_true_or_unknown(
        typetracer.broadcast_shapes(
            (1, typetracer.unknown_value(dtypes.int64), 3),
            (3,),
        )
        == (1, typetracer.unknown_value(dtypes.int64), 3)
    )
    assert is_true_or_unknown(
        typetracer.broadcast_shapes(
            (typetracer.unknown_value(dtypes.int64), 2, 3),
            (2, 3),
        )
        == (typetracer.unknown_value(dtypes.int64), 2, 3)
    )
    assert is_true_or_unknown(
        typetracer.broadcast_shapes(
            (typetracer.unknown_value(dtypes.int64),),
            (2,),
        )
        == (typetracer.unknown_value(dtypes.int64),)
    )
    assert is_true_or_unknown(
        typetracer.broadcast_shapes(
            (
                typetracer.unknown_value(dtypes.int64),
                typetracer.unknown_value(dtypes.int64),
            ),
            (1,),
        )
        == (
            typetracer.unknown_value(dtypes.int64),
            typetracer.unknown_value(dtypes.int64),
        )
    )


def test_typetracer_shapes():
    shape = typetracer.TypeTracerShape((1, 2, 3))
    assert is_true_or_unknown(shape == (1, 2, 3))
    assert is_true_or_unknown(shape == (1, typetracer.unknown_value(dtypes.int64), 3))
    assert is_true_or_unknown((1, typetracer.unknown_value(dtypes.int64), 3) == shape)
    assert is_true_or_unknown(
        shape
        == (
            typetracer.unknown_value(dtypes.int64),
            typetracer.unknown_value(dtypes.int64),
            typetracer.unknown_value(dtypes.int64),
        )
    )
    assert is_true_or_unknown(
        shape
        == (
            typetracer.unknown_value(dtypes.int64),
            2,
            typetracer.unknown_value(dtypes.int64),
        )
    )

    assert not (
        is_true_or_unknown(
            shape
            == (
                typetracer.unknown_value(dtypes.int64),
                1,
                typetracer.unknown_value(dtypes.int64),
            )
        )
    )
    assert not (
        is_true_or_unknown(
            shape
            == (
                typetracer.unknown_value(dtypes.int64),
                typetracer.unknown_value(dtypes.int64),
                8,
            )
        )
    )
    assert not (
        is_true_or_unknown(shape == (typetracer.unknown_value(dtypes.int64), 3))
    )
