# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import hypothesis_awkward.strategies as st_ak
from hypothesis import given, settings

import awkward as ak


@settings(max_examples=200)
@given(a=st_ak.constructors.arrays(allow_virtual=False, allow_union=False))
def test_roundtrip(a: ak.Array) -> None:
    """`to_buffers` followed by `from_buffers` reconstructs the array.

    Virtual and union arrays are excluded because of known issues:

    - Virtual arrays: `ak.materialize`, `ak.array_equal`, and `ak.to_list`
      all raise exceptions for some arrays with virtual buffers. The errors
      occur in `to_packed()` during materialization, e.g.,
      "placeholder arrays that are sliced should have known shapes" and
      "RecordArray length must be an integer for an array with concrete data".
      Even `ak.to_list` is affected because it calls `to_packed()` internally.

    - Union arrays: `ak.array_equal` returns `False` for some empty union
      arrays that are identical, e.g., `0 * union[unknown, (bool, bool)]`.

    See `test_roundtrip_no_error` for a separate test that includes virtual
    and union arrays but only asserts that the roundtrip runs without error.
    """
    sent = ak.to_buffers(a)
    returned = ak.from_buffers(*sent)
    assert ak.array_equal(a, returned)


@settings(max_examples=200)
@given(a=st_ak.constructors.arrays())
def test_roundtrip_no_error(a: ak.Array) -> None:
    """`to_buffers` followed by `from_buffers` does not raise.

    This test includes all array types (including virtual and union arrays)
    but does not assert equality because `ak.array_equal` raises exceptions
    for virtual arrays and returns incorrect results for some union arrays.
    See `test_roundtrip` for equality assertions on the supported subset.
    """
    sent = ak.to_buffers(a)
    ak.from_buffers(*sent)
