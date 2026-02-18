from __future__ import annotations

import hypothesis_awkward.strategies as st_ak
from hypothesis import given, settings

import awkward as ak


@settings(max_examples=200)
@given(a=st_ak.constructors.arrays())
def test_roundtrip(a: ak.Array) -> None:
    """`to_buffers` followed by `from_buffers` reconstructs the array."""
    ak.from_buffers(*ak.to_buffers(a))
