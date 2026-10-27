import numpy as np
import pytest

import awkward as ak

pd = pytest.importorskip("pandas")


@pytest.mark.parametrize("how", [None, "inner", "outer", "left", "right"])
@pytest.mark.parametrize("bits32", [False, True])
def test_index_width(monkeypatch, how, bits32):
    monkeypatch.setattr(ak._util, "win", True)
    monkeypatch.setattr(ak._util, "bits32", bits32)
    array = ak.Array({"x": ["abc", "def"], "y": [None, ["g1", "g2"]]})
    frames = ak.to_dataframe(array, how=how)
    if how is not None:
        frames = [frames]
    for frame in frames:
        for level in (
            frame.index.levels
            if isinstance(frame.index, pd.MultiIndex)
            else [frame.index]
        ):
            assert level.dtype == np.dtype("int64")
    if how == "right":
        assert frames[0].index.tolist() == [(1, 0), (1, 1)]
