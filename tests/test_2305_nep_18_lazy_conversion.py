# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak


def test_binary():
    ak_array = ak.Array(np.arange(10, dtype="<u4"))
    np_array = np.arange(10, dtype=">u4")
    assert np.array_equal(ak_array, np_array)


def test_different_backend():
    pytest.importorskip("jax")
    ak.jax.register_and_check()

    left = ak.Array([0, 1, 2], backend="jax")
    right = ak.Array([0, 1, 2], backend="cpu")
    with pytest.raises(
        ValueError, match="cannot operate on arrays with incompatible backends"
    ):
        assert np.array_equal(left, right)
