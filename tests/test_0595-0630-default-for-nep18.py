# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    out = np.digitize(ak.Array([1.1, 2.2, 3.3, 4.4, 5.5]), ak.Array([0, 1, 3, 6]))
    assert isinstance(out, ak.Array)
    assert out.tolist() == [2, 2, 3, 3, 3]

    assert np.median(ak.Array([1.1, 2.2, 3.3, 4.4, 5.5])) == 3.3
