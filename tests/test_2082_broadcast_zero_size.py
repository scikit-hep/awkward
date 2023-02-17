# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_axis(axis):
    this_slice = (slice(None),) * axis + (slice(0, 0),) + (...,)
    that_slice = (slice(None),) * axis + (slice(0, 1),) + (...,)
    x = np.arange(3 * 5 * 7).reshape(3, 5, 7)
    result = np.broadcast_arrays(x[this_slice], x[that_slice])

    y = ak.from_numpy(x)
    result_ak = ak.broadcast_arrays(y[this_slice], y[that_slice])

    assert ak.almost_equal(result, result_ak)
