from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

pd = pytest.importorskip("pandas")


_data = {
    "x": ["abc", "FG_12345"],
    "y": [None, ["g1", "g2"]],
}

_expected_inner = pd.DataFrame(
    {
        "x": ["FG_12345", "FG_12345"],
        "y": ["g1", "g2"],
    },
    index=pd.MultiIndex.from_tuples(
        [(1, 0), (1, 1)],
        names=["entry", "subentry"],
    ),
)

_expected_outer = pd.DataFrame(
    {
        "x": ["abc", "FG_12345", "FG_12345"],
        "y": [np.nan, "g1", "g2"],
    },
    index=pd.MultiIndex.from_tuples(
        [(0, 0), (1, 0), (1, 1)],
        names=["entry", "subentry"],
    ),
)

_expected_left = pd.DataFrame(
    {
        "x": ["abc", "FG_12345", "FG_12345"],
        "y": [np.nan, "g1", "g2"],
    },
    index=pd.MultiIndex.from_tuples(
        [(0, 0), (1, 0), (1, 1)],
        names=["entry", "subentry"],
    ),
)

_expected_right = pd.DataFrame(
    {
        "x": ["FG_12345", "FG_12345"],
        "y": ["g1", "g2"],
    },
    index=pd.MultiIndex.from_tuples(
        [(1, 0), (1, 1)],
        names=["entry", "subentry"],
    ),
)

params = [
    ("inner", _expected_inner),
    ("outer", _expected_outer),
    ("left", _expected_left),
    ("right", _expected_right),
]


@pytest.mark.parametrize("how, expected", params)
def test_merge_option(how: str, expected: pd.DataFrame) -> None:
    a = ak.Array(_data)
    actual = ak.to_dataframe(a, how=how)
    pd.testing.assert_frame_equal(actual, expected)
