# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import hypothesis_awkward.strategies as st_ak
from hypothesis import given

import awkward as ak
from tests.properties.operations import known_issues


@given(
    a=st_ak.constructors.arrays().filter(lambda a: not known_issues.has_issue_2853(a))
)
def test_roundtrip(a: ak.Array) -> None:
    """`to_buffers` followed by `from_buffers` reconstructs the array."""
    sent = ak.to_buffers(a)
    returned = ak.from_buffers(*sent)
    assert ak.array_equal(a, returned, equal_nan=True)
