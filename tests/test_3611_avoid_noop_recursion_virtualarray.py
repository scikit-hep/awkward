# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
#
from __future__ import annotations

import sys

import numpy as np

import awkward as ak
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.virtual import VirtualArray


def test():
    numpy_like = Numpy.instance()
    vc = VirtualArray(
        numpy_like,
        shape=(1,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([1], dtype=np.int64),
    )
    v = ak.contents.NumpyArray(vc)

    for _ in range(sys.getrecursionlimit() + 1):
        v = v[:]

    assert ak.materialize(v)
