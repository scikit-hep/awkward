# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

numba = pytest.importorskip("numba")


def test_field_name():
    builder = ak.ArrayBuilder()
    builder.begin_record("x")
    builder.field("time").real(0.0)
    builder.end_record()

    @numba.njit
    def func(builder):
        builder.begin_record("x")
        builder.field("time").real(2.0)
        builder.end_record()
        return builder

    func(builder)

    assert builder.type == ak.types.ArrayType(
        ak.types.RecordType(
            [ak.types.NumpyType("float64")], ["time"], parameters={"__record__": "x"}
        ),
        2,
    )


def test_no_field_name():
    builder = ak.ArrayBuilder()
    builder.begin_record()
    builder.field("time").real(0.0)
    builder.end_record()

    @numba.njit
    def func(builder):
        builder.begin_record()
        builder.field("time").real(2.0)
        builder.end_record()
        return builder

    func(builder)

    result = builder.snapshot()
    assert ak.almost_equal(
        result,
        ak.contents.RecordArray(
            fields=["time"],
            contents=[ak.contents.NumpyArray(np.array([0, 2], dtype=np.float64))],
        ),
    )
