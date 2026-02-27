# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import hypothesis_awkward.strategies as st_ak
from hypothesis import given, settings

import awkward as ak


@settings(max_examples=200)
@given(a=st_ak.constructors.arrays(allow_nan=True))
def test_reflexivity(a: ak.Array) -> None:
    """An array must be equal to itself."""
    assert ak.array_equal(a, a, equal_nan=True)


@settings(max_examples=200)
@given(
    a1=st_ak.constructors.arrays(allow_nan=True),
    a2=st_ak.constructors.arrays(allow_nan=True),
)
def test_symmetry(a1: ak.Array, a2: ak.Array) -> None:
    """Argument order must not affect the result."""
    result_12 = ak.array_equal(a1, a2, equal_nan=True)
    result_21 = ak.array_equal(a2, a1, equal_nan=True)
    assert result_12 == result_21
