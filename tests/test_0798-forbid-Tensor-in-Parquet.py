# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import os

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


pytest.importorskip("pyarrow.parquet")


def test(tmp_path):
    filename = os.path.join(tmp_path, "test.parquet")
    dog = ak.from_iter([1, 2, 5])
    cat = ak.from_iter([4])
    pets = ak.zip({"dog": dog[np.newaxis], "cat": cat[np.newaxis]}, depth_limit=1)
    ak.to_parquet(pets, filename)
    assert ak.from_parquet(filename).tolist() == pets.tolist()
