# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest
import numpy as np
import cupy as cp
import awkward1 as ak

def test_cupy_interop():
    c = cupy.arange(10)
    n = np.arange(10)
    cupy_index_arr = ak.layout.Index64.from_cupy(c)
    np_index_arr = ak.layout.Index64(n)

    # GPU->CPU
    assert ak.to_list(np.asarray(cupy_index_arr.copy_to("cpu"))) == ak.to_list(np.asarray(np_index_arr))
    # CPU->CPU
    assert ak.to_list(np.asarray(np_index_arr.copy_to("cpu"))) == ak.to_list(np.asarray(np_index_arr))
    #CPU->GPU->CPU
    assert ak.to_list(np.asarray(np_index_arr)) == ak.to_list(np.asarray(np_index_arr.copy_to("cuda").copy_to("cpu")))
