# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak


def test_typetracer():
    array = ak.Array([[0, 1, 2, 3], [8, 9, 10, 11]])
    typetracer = ak.Array(array.layout.to_typetracer())

    assert ak.backend(typetracer) == "typetracer"


def test_to_typetracer():
    array = ak.Array([[0, 1, 2, 3], [8, 9, 10, 11]])

    assert ak.backend(array) == "cpu"
    assert ak.backend(ak.to_backend(array, "typetracer")) == "typetracer"
