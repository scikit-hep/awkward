# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    array = ak.Array([[1, 2, 3], [], [4, 5]])
    assert array[2].tolist() == [4, 5]
    assert array[2,].tolist() == [4, 5]
    assert array[2, 1] == 5

    assert array[np.int64(2)].tolist() == [4, 5]
    assert array[np.int64(2),].tolist() == [4, 5]
    assert array[np.int64(2), 1] == 5
    assert array[2, np.int64(1)] == 5
    assert array[np.int64(2), np.int64(1)] == 5

    assert array[np.int64([2]).reshape(())].tolist() == [4, 5]
    assert array[np.int64([2]).reshape(()),].tolist() == [4, 5]
    assert array[np.int64([2]).reshape(()), 1] == 5
    assert array[2, np.int64([1]).reshape(())] == 5
    assert array[np.int64([2]).reshape(()), np.int64([1]).reshape(())] == 5
