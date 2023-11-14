# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak


def test_where():
    result = ak.where(
        [True, False, False], ak.to_backend([1, 2, 3], "typetracer"), [4, 5, 6]
    )
    assert ak.backend(result) == "typetracer"


def test_almost_equal():
    assert not ak.almost_equal(
        [True, False, False], ak.to_backend([True, False, False], "typetracer")
    )
