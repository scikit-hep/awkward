# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

pd = pytest.importorskip("pandas")

# Fields of depth 1, 2, and 3 (with empty and missing lists), so that merging
# has both shared and extra index levels
array = ak.Array(
    {"x": [1, 2, 3], "y": [[1.1, 2.2], [], [3.3]], "z": [[[1], [2, 3]], [[4]], None]}
)


@pytest.mark.parametrize("how", [None, "inner", "outer", "left", "right"])
def test_index_dtype(how):
    frames = ak.to_dataframe(array, how=how)
    for frame in frames if how is None else [frames]:
        assert (frame.index.to_frame().dtypes == np.int64).all()


def test_index_values():
    assert ak.to_dataframe(array, how="outer").index.tolist() == [
        (0, 0, 0),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (2, 0, 0),
    ]


@pytest.mark.parametrize("how", ["inner", "outer", "left", "right"])
@pytest.mark.parametrize("fields", [["x", "y", "z"], ["z", "y", "x"]])
def test_merge_extra_index_levels_in_both_field_orders(how, fields):
    frame = ak.to_dataframe(array[fields], how=how)
    assert frame.index.nlevels == 3
    assert all(dtype == np.dtype(np.int64) for dtype in frame.index.to_frame().dtypes)
    assert set(frame.columns) == set(fields)
