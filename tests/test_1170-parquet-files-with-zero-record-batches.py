# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

pytest.importorskip("pyarrow.parquet")


def test():
    empty = ak.from_parquet("tests/samples/zero-record-batches.parquet")
    assert isinstance(empty, ak.Array)
    assert len(empty) == 0
    assert str(empty.type) == "0 * {}"
