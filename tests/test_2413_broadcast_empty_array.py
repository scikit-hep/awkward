# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest

import awkward as ak


def test_broadcast_arrays():
    x = ak.Array([])
    assert x.layout.is_unknown
    y = ak.Array([1])
    with pytest.raises(
        TypeError, match=r"EmptyArray.*convert these to arrays with known dtypes"
    ):
        ak.broadcast_arrays(x, y)
