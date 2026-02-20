# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import hypothesis_awkward.strategies as st_ak
from hypothesis import given, settings

import awkward as ak


@settings(max_examples=500)
@given(a=st_ak.constructors.arrays(allow_union=False))
def test_roundtrip(a: ak.Array) -> None:
    """`to_buffers` followed by `from_buffers` reconstructs the array.

    Union arrays are excluded because of known issues:

    - Union arrays: `ak.array_equal` returns `False` for some empty union
      arrays that are identical, e.g., `0 * union[unknown, (bool, bool)]`.

    See `test_roundtrip_no_equality_check` for a separate test that includes
    union arrays but only asserts that the roundtrip runs.
    """
    sent = ak.to_buffers(a)
    returned = ak.from_buffers(*sent)
    assert ak.array_equal(a, returned)


@settings(max_examples=500)
@given(a=st_ak.constructors.arrays())
def test_roundtrip_no_equality_check(a: ak.Array) -> None:
    """`to_buffers` followed by `from_buffers` does not error.

    This test includes all array types (including union arrays)
    but does not assert equality because `ak.array_equal`
    returns incorrect results for some union arrays.
    See `test_roundtrip` for equality assertions on the supported subset.
    """
    sent = ak.to_buffers(a)
    ak.from_buffers(*sent)
