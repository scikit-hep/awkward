# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import os

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


pytest.importorskip("pyarrow.parquet")


def test(tmp_path):
    filename = os.path.join(tmp_path, "what-ever.parquet")
    fish = ak.Array([True, True])[np.newaxis]
    clob = ak.Array([2, 3, 7])[np.newaxis]
    frog = ak.zip({"c": clob, "f": fish}, depth_limit=1)
    ak.to_parquet(frog, filename)
    assert ak.from_parquet(filename).tolist() == frog.tolist()
