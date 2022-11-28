# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import os

import numpy as np  # noqa: F401
import pytest

import awkward as ak

pyarrow = pytest.importorskip("pyarrow")


def test_issue1():
    null_array = ak.highlevel.Array({"null_col": [None]}).layout
    tpe = null_array.to_arrow().storage.field("null_col").type
    assert tpe.storage_type == pyarrow.null()


pyarrow_parquet = pytest.importorskip("pyarrow.parquet")


def test_issue2(tmp_path):
    import awkward._connect.pyarrow

    filename = os.path.join(tmp_path, "whatever.parquet")

    null_table = pyarrow.Table.from_pydict({"null_col": pyarrow.array([None])})
    pyarrow_parquet.write_table(null_table, filename)

    assert (
        str(
            awkward._connect.pyarrow.handle_arrow(
                pyarrow_parquet.read_table(filename)
            ).form.type
        )
        == "{null_col: ?unknown}"
    )
