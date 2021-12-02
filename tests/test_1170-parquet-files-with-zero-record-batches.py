# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

pytest.importorskip("pyarrow.parquet")


def test_parquet():
    empty = ak.from_parquet("tests/samples/zero-record-batches.parquet")
    assert isinstance(empty, ak.Array)
    assert len(empty) == 0
    assert str(empty.type) == "0 * {}"


def test_concatenate():
    one = ak.Array(ak.layout.RecordArray([], [], length=0))
    two = ak.Array(ak.layout.RecordArray([], [], length=0))
    assert len(ak.concatenate([one, two])) == 0
    assert len(ak.concatenate([one])) == 0

    one = ak.Array(ak.layout.RecordArray([], [], length=3))
    two = ak.Array(ak.layout.RecordArray([], [], length=5))
    assert len(ak.concatenate([one, two])) == 8
    assert len(ak.concatenate([one])) == 3
