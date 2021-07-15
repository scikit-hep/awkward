# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


@pytest.mark.skipif(
    ak._util.py27 or ak._util.py35,
    reason="SOMETIMES doesn't work under Python 2.7 and 3.5",
)
def test():
    out = np.digitize(ak.Array([1.1, 2.2, 3.3, 4.4, 5.5]), ak.Array([0, 1, 3, 6]))
    assert isinstance(out, ak.Array)
    assert out.tolist() == [2, 2, 3, 3, 3]

    assert np.median(ak.Array([1.1, 2.2, 3.3, 4.4, 5.5])) == 3.3
