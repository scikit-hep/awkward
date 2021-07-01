# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    array = ak.from_numpy(np.zeros((3, 3, 5)))
    flattened = ak.flatten(array, axis=None)
    assert flattened.ndim == 1
