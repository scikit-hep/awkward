# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

import awkward._v2._lookup  # noqa: E402
import awkward._v2._connect.cling  # noqa: E402

ROOT = pytest.importorskip("ROOT")


compiler = ROOT.gInterpreter.Declare


def test_to_from_data_frame_large():
    n = 6
    assert 2 * (n // 2) == n
    rows = 3 ** (n // 2)
    cols = n

    arr = np.zeros((rows, cols), dtype=int)
    shape = (rows,)

    source = np.array([-1, 0, 1], dtype=np.int64)[:, None]

    for col in range(n // 2):
        shape = (
            -1,
            3,
            shape[-1] // 3,
        )
        col_view = arr[:, col]
        col_view.shape = shape
        col_view[:] = source

    ak_array_in = ak._v2.from_numpy(arr, regulararray=True)

    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})
    assert data_frame.GetColumnType("x") == "ROOT::VecOps::RVec<int64_t>"
