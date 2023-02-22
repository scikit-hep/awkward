# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak
from awkward._nplikes.shape import unknown_length


def test():
    array = ak.Array([2, 1, 1, 1, 2, 2, 3, 3, 4], backend="typetracer")
    result = ak.run_lengths(array)
    assert isinstance(result.layout, ak.contents.NumpyArray)
    assert result.layout.length is unknown_length
    assert result.layout.dtype == np.dtype("int64")
