# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import hypothesis_awkward.strategies as st_ak
from hypothesis import given, settings

import awkward as ak


@settings(max_examples=1000)
@given(a=st_ak.constructors.arrays())
def test_roundtrip(a: ak.Array) -> None:
    """`to_buffers` followed by `from_buffers` reconstructs the array."""
    sent = ak.to_buffers(a)
    returned = ak.from_buffers(*sent)
    assert ak.array_equal(a, returned)
