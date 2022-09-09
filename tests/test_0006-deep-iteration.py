# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_iterator():
    content = ak._v2.contents.NumpyArray(np.array([1.1, 2.2, 3.3]))
    offsets = ak._v2.index.Index32(np.array([0, 2, 2, 3], "i4"))
    array = ak._v2.contents.ListOffsetArray(offsets, content)

    assert list(content) == [1.1, 2.2, 3.3]
    assert [np.asarray(x).tolist() for x in array] == [[1.1, 2.2], [], [3.3]]
