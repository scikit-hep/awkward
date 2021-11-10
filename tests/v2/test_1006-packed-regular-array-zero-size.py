# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test():
    array = ak._v2.contents.RegularArray(
        ak._v2.contents.NumpyArray(np.empty(0, dtype=np.int32)),
        size=0,
        zeros_length=1,
    )
    packed = ak._v2.operations.structure.packed(array)
    assert ak.to_list(packed) == [[]]
