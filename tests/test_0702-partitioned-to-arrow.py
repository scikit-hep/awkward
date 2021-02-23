# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


pyarrow = pytest.importorskip("pyarrow")


def test():
    array = ak.repartition(ak.Array(range(10)), [3, 3, 3, 1])
    arrow = ak.to_arrow(array)
    assert isinstance(arrow, pyarrow.ChunkedArray)
    assert arrow.to_pylist() == list(range(10))
