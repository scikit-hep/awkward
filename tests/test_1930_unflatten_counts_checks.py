# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak
import awkward._connect.cling
import awkward._lookup


def test():
    array = ak.Array([[0, 1, 2, 3], [8, 9, 10, 11]])

    with pytest.raises(ValueError, match=r"one-dimensional"):
        ak.unflatten(array, [[4, 2, 2]], axis=-1)

    with pytest.raises(ValueError, match=r"negative counts"):
        ak.unflatten(array, -40, axis=-1)

    with pytest.raises(ValueError, match=r"does not fit"):
        ak.unflatten(array, [-40], axis=-1)

    with pytest.raises(ValueError, match=r"must be integers"):
        ak.unflatten(array, [4.0, 2.0, 2.0], axis=-1)

    with pytest.raises(ValueError, match=r"too large"):
        ak.unflatten(array, 100, axis=-1)

    with pytest.raises(ValueError, match=r"does not fit"):
        ak.unflatten(array, [100], axis=-1)
