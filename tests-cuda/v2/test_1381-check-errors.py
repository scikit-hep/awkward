# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import pytest  # noqa: F401
import numpy as np  # noqa: F401
import cupy as cp  # noqa: F401
import awkward as ak  # noqa: F401
import awkward._v2._connect.cuda


def test():
    v2_array = ak._v2.Array([[1, 2, 3], [], [4, 5]]).layout
    starts, stops = v2_array.starts, v2_array.stops

    err_array = ak._v2.contents.ListArray(stops, starts, v2_array.content)
    cuda_array_err = err_array.to_backend("cuda")

    stream = cp.cuda.Stream(non_blocking=True)

    with stream:
        cuda_array_err._compact_offsets64(True)

    with pytest.raises(ValueError):
        awkward._v2._connect.cuda.synchronize_cuda(stream)
