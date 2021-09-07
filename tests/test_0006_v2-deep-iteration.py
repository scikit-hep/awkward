# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test_iterator():
    content = ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3]))
    offsets = ak.layout.Index32(np.array([0, 2, 2, 3], "i4"))
    array = ak.layout.ListOffsetArray32(offsets, content)
    content = v1_to_v2(content)
    array = v1_to_v2(array)

    assert list(content) == [1.1, 2.2, 3.3]
    assert [np.asarray(x).tolist() for x in array] == [[1.1, 2.2], [], [3.3]]
