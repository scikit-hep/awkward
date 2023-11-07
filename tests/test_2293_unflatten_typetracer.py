# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest  # noqa: F401

import awkward as ak


def test_array_with_length():
    array = ak.Array(
        [[100.0, 200.0, 22.0], [4.0, 5.0], [8.0, 9.0, 10.0, 11.0]], backend="typetracer"
    )
    counts = ak.Array([2, 1, 1, 1, 2, 2])
    ak.unflatten(array, counts)


def test_array_without_length():
    array = ak.Array(
        ak.to_layout(
            [[100.0, 200.0, 22.0], [4.0, 5.0], [8.0, 9.0, 10.0, 11.0]]
        ).to_typetracer(forget_length=True)
    )
    counts = ak.Array(
        ak.to_layout([2, 1, 1, 1, 2, 2]).to_typetracer(forget_length=True)
    )
    ak.unflatten(array, counts)


def test_scalar():
    array = ak.Array(
        [[100.0, 200.0, 22.0, 8.0], [4.0, 5.0, 5.0, 8.0], [8.0, 9.0, 10.0, 8.0]],
        backend="typetracer",
    )
    ak.unflatten(array, 2)


def test_unknown_scalar():
    array = ak.Array(
        ak.to_layout([[100, 200, 22], [4, 5, 5], [8, 9, 10]]).to_typetracer(
            forget_length=True
        )
    )
    ak.unflatten(array, array[0, 0])


def test_unknown_length():
    array = ak.Array(
        ak.to_layout(
            [[100.0, 200.0, 22.0], [4.0, 5.0, 5.0], [8.0, 9.0, 10.0]]
        ).to_typetracer(forget_length=True)
    )
    ak.unflatten(array, array.layout.length)
