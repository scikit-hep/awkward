# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test_iterator():
    content = ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3]))
    offsets = ak.index.Index32(np.array([0, 2, 2, 3], "i4"))
    array = ak.contents.ListOffsetArray(offsets, content)

    assert list(content) == [1.1, 2.2, 3.3]
    assert [np.asarray(x).tolist() for x in array] == [[1.1, 2.2], [], [3.3]]
