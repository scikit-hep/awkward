# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak


def test_akarray_from_zero_dim_nparray():
    np_scalar = np.array(2.7)  # A kind of scalar in numpy.
    assert np_scalar.ndim == 0 and np_scalar.shape == ()
    with pytest.raises(TypeError):
        # Conversion to ak.Array ought to throw here:
        b = ak.Array(np_scalar)  # (bugged) value: <Array [2.7] type='1 * int64'>
        # Now we're failing. Here's why.
        c = ak.to_numpy(b)  # value: array([2.7])
        assert np_scalar.shape == c.shape  # this fails

    with pytest.raises(TypeError):
        b = ak.from_numpy(np_scalar)
        c = ak.to_numpy(b)
        assert np_scalar.shape == c.shape
