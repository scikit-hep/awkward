# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak


def test_binary():
    ak_array = ak.Array(np.arange(10, dtype="<u4"))
    np_array = np.arange(10, dtype=">u4")
    with pytest.raises(TypeError):
        # ak.array_equal now overrides np.array_equal, and requires
        # both arrays to be valid within awkward.
        assert np.array_equal(ak_array, np_array)
