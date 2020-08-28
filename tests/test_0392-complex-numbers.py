# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import numpy as np
import awkward1


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test(dtype):
    builder = awkward1.ArrayBuilder()

    with builder.list():
        for size in np.random.randint(4, 10, 5):
            np_array = np.random.random(size).astype(dtype)
            ak_array = awkward1.Array(np_array)
            builder.append(ak_array)

    R = builder.snapshot()
    print(R)
