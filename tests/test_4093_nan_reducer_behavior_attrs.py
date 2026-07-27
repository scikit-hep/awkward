# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest

import awkward as ak

# Regression for: nan-variant reducers calling nan_to_none with highlevel=False and/or
# behavior=None/attrs=None.  Because ctx.unwrap on a raw layout does not collect the
# input array's behavior/attrs, they were silently dropped when the outer _impl
# received a layout instead of an ak.Array as the intermediate.


@pytest.mark.parametrize(
    "func",
    [ak.nanmin, ak.nanmax, ak.nanmean, ak.nanvar, ak.nanstd],
    ids=["nanmin", "nanmax", "nanmean", "nanvar", "nanstd"],
)
def test_nan_reducer_preserves_behavior_attrs(func):
    # Use axis=1 so the result is an array (not a scalar); the behavior/attrs drop
    # was only observable on array-valued results where wrap() is called.
    arr = ak.Array(
        [[1.0, float("nan"), 3.0], [4.0, float("nan"), 6.0]],
        behavior={"marker": "test"},
        attrs={"source": "data"},
    )
    result = func(arr, axis=1)
    assert result.behavior == {"marker": "test"}
    assert result.attrs == {"source": "data"}
