# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import hypothesis_awkward.strategies as st_ak
from hypothesis import given

import awkward as ak


@given(
    a=st_ak.constructors.arrays(
        # merging categorical arrays (generated since hypothesis-awkward
        # 0.22.0) raises NotImplementedError; not yet triaged
        allow_categorical=False,
    )
)
def test_roundtrip(a: ak.Array) -> None:
    """`to_buffers` followed by `from_buffers` reconstructs the array."""
    sent = ak.to_buffers(a)
    returned = ak.from_buffers(*sent)
    assert ak.array_equal(a, returned, equal_nan=True)
