# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    a = ak.to_numpy(ak.Array({"A": [1, 2, 3], "B": [4, None, 5]}))
    assert a["A"].data.tolist() == [1, 2, 3]
    assert a["A"].mask.tolist() == [False, False, False]
    assert a["B"].data[0] == 4
    assert a["B"].data[2] == 5
    assert a["B"].mask.tolist() == [False, True, False]
