# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
#
from __future__ import annotations

import numpy as np

from awkward._nplikes.numpy import Numpy
from awkward._nplikes.virtual import VirtualNDArray


def test():
    numpy_like = Numpy.instance()
    v1 = VirtualNDArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.float32),
        generator=lambda: np.arange(10, dtype=np.float32),
    )
    v2 = v1.copy()
    np.testing.assert_array_equal(v1.materialize(), np.arange(10, dtype=np.float32))
    np.testing.assert_array_equal(v2.materialize(), np.arange(10, dtype=np.float32))

    v1 = VirtualNDArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.float32),
        generator=lambda: np.arange(10, dtype=np.float32),
    )
    v2 = v1.copy()
    np.testing.assert_array_equal(v2.materialize(), np.arange(10, dtype=np.float32))
    np.testing.assert_array_equal(v1.materialize(), np.arange(10, dtype=np.float32))
