# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak
import awkward._connect.cling
import awkward._lookup

ROOT = pytest.importorskip("ROOT")

ROOT.ROOT.EnableImplicitMT(1)

compiler = ROOT.gInterpreter.Declare


def test_to_from_data_frame_large():
    n = 6
    assert 2 * (n // 2) == n
    rows = 3 ** (n // 2)
    cols = n

    arr = np.zeros((rows, cols), dtype=np.int64)
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

    ak_array_in = ak.from_numpy(arr, regulararray=True)

    data_frame = ak.to_rdataframe({"x": ak_array_in})

    ak_array_out = ak.from_rdataframe(
        data_frame,
        columns=("x",),
        highlevel=False,
    )
    assert isinstance(ak_array_out, ak.contents.RecordArray)
    assert len(ak_array_in) == len(ak_array_out)


def test_data_frame_integers():
    ak_array_in = ak.Array([1, 2, 3, 4, 5])

    data_frame = ak.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x") == "int64_t"

    def overload_add(left, right):
        return ak.Array({"x": left.x + right.x})

    behavior = {}
    behavior[np.add, "Overload", "Overload"] = overload_add

    ak_array_out = ak.from_rdataframe(
        data_frame,
        columns=("x",),
        keep_order=True,
        behavior=behavior,
        with_name="Overload",
    )
    assert ak_array_in.to_list() == ak_array_out["x"].to_list()
    assert (ak_array_out + ak_array_out).to_list() == [
        {"x": 2},
        {"x": 4},
        {"x": 6},
        {"x": 8},
        {"x": 10},
    ]
