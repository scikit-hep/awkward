# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test():
    array = ak._v2.contents.NumpyArray(np.zeros((3, 0), dtype=np.int32))
    buffs = ak._v2.operations.convert.to_buffers(array)
    new_array = ak._v2.operations.convert.from_buffers(*buffs)

    assert ak.to_list(new_array) == [[], [], []]
