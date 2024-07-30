# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import os

import pytest

import awkward as ak

pytest.importorskip("pyarrow.parquet")


def test(tmp_path):
    filename = os.path.join(tmp_path, "whatever.parquet")

    px = ak.Array([[1.0, 2.0], [3.0], [], [4.0, 5.0, 6.0]])
    py = ak.Array([[0.5, 1.5], [2.5], [], [3.5, 4.5, 5.5]])
    pz = ak.Array([[0.2, 1.2], [2.2], [], [3.2, 4.2, 5.2]])
    e = ak.Array([[0.1, 1.1], [2.1], [], [3.1, 4.1, 5.1]])
    data = {"px": px, "py": py, "pz": pz, "e": e}  # not quite a record

    ak.to_parquet(data, filename)

    assert ak.from_parquet(filename).tolist() == ak.to_list(ak.to_layout(data))
