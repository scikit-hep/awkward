# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak
from awkward._nplikes.array_like import MaterializableArray
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.placeholder import PlaceholderArray
from awkward._nplikes.virtual import VirtualNDArray


def test():
    numpy = Numpy.instance()
    a = ak.contents.NumpyArray(np.arange(10, dtype=np.float32))
    b = ak.contents.NumpyArray(PlaceholderArray(numpy, (10,), np.float32))
    layout = ak.contents.RecordArray([a, b], None, 5)
    layout.materialize(VirtualNDArray)
    layout.to_packed()
    with pytest.raises(RuntimeError, match="should never have been encountered"):
        layout.materialize(MaterializableArray)

    a = ak.contents.NumpyArray(
        VirtualNDArray(
            numpy, (10,), np.float32, lambda: np.arange(10, dtype=np.float32)
        )
    )
    b = ak.contents.NumpyArray(PlaceholderArray(Numpy.instance(), (10,), np.float32))
    layout = ak.contents.RecordArray([a, b], None, 5)
    layout.materialize(VirtualNDArray)
    layout.to_packed()
    with pytest.raises(RuntimeError, match="should never have been encountered"):
        layout.materialize(MaterializableArray)
