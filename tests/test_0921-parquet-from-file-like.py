# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import io
import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


pytest.importorskip("pyarrow.parquet")


def test():
    array = ak.Array([1, 2, 3])
    file_ = io.BytesIO()
    ak.to_parquet(array, file_)
    file_.seek(0)

    array_from_file = ak.from_parquet(file_)
    assert ak.to_list(array) == ak.to_list(array_from_file)
