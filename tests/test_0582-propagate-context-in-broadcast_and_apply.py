# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_toregular():
    array = ak.Array([{
        "x": np.arange(2*3*5).reshape(2, 3, 5).tolist(),
        "y": np.arange(2*3*5*7).reshape(2, 3, 5, 7)
    }])
    assert str(array.type) == (
        '1 * {"x": var * var * var * int64, '
        '"y": var * var * var * var * int64}'
    )
    assert str(ak.to_regular(array, axis=-1).type) == (
        '1 * {"x": var * var * 5 * int64, '
        '"y": var * var * var * 7 * int64}'
    )
    assert str(ak.to_regular(array, axis=-2).type) == (
        '1 * {"x": var * 3 * var * int64, '
        '"y": var * var * 5 * var * int64}'
    )
    assert str(ak.to_regular(array, axis=-3).type) == (
        '1 * {"x": 2 * var * var * int64, '
        '"y": var * 3 * var * var * int64}'
    )
