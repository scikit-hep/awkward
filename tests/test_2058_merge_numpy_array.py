# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test_numpy():
    x = ak.from_numpy(
        np.arange(4 * 3, dtype=np.int64).reshape(4, 3), regulararray=False
    )
    y = ak.from_numpy(
        np.arange(4 * 2, dtype=np.int64).reshape(4, 2), regulararray=False
    )

    assert ak.concatenate((x, y)).type == ak.types.ArrayType(
        ak.types.ListType(ak.types.NumpyType("int64")), 8
    )


def test_regular():
    x = ak.from_numpy(np.arange(4 * 3, dtype=np.int64).reshape(4, 3), regulararray=True)
    y = ak.from_numpy(np.arange(4 * 2, dtype=np.int64).reshape(4, 2), regulararray=True)

    assert ak.concatenate((x, y)).type == ak.types.ArrayType(
        ak.types.ListType(ak.types.NumpyType("int64")), 8
    )


def test_regular_mergebool_false():
    x = ak.from_numpy(np.zeros((4, 3), dtype=np.bool_), regulararray=True)
    y = ak.from_numpy(np.ones((4, 2), dtype=np.int64), regulararray=True)

    assert ak.concatenate((x, y), mergebool=False).type == ak.types.ArrayType(
        ak.types.UnionType(
            [
                ak.types.RegularType(ak.types.NumpyType("bool"), 3),
                ak.types.RegularType(ak.types.NumpyType("int64"), 2),
            ]
        ),
        8,
    )


def test_regular_mergebool_true():
    x = ak.from_numpy(np.zeros((4, 3), dtype=np.bool_), regulararray=True)
    y = ak.from_numpy(np.ones((4, 2), dtype=np.int64), regulararray=True)

    assert ak.concatenate((x, y), mergebool=True).type == ak.types.ArrayType(
        ak.types.ListType(ak.types.NumpyType("int64")), 8
    )
