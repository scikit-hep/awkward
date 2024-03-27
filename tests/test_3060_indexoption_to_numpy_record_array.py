# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

import awkward as ak


def test_to_numpy_record_array():
    test_array = ak.Array([{"x": 1, "y": 1.1}, None]).to_numpy()
    expected_numpy = np.ma.masked_array(
        data=[(1, 1.1), (np.iinfo(np.int64).max, np.nan)],
        mask=[(False, False), (True, True)],
        dtype=[("x", "<i8"), ("y", "<f8")],
    )
    assert np.array_equal(test_array["x"], expected_numpy["x"])
    # equal_nan not supported on integer arrays
    try:
        assert np.array_equal(test_array["y"], expected_numpy["y"], equal_nan=True)
    except TypeError:
        # older numpy versions do not support `equal_nan`
        assert np.array_equal(test_array["y"].filled(), expected_numpy["y"].filled())
