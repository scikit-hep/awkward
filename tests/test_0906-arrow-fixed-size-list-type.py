# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

pyarrow = pytest.importorskip("pyarrow")


def test():
    struct = pyarrow.struct(
        [
            pyarrow.field("x", pyarrow.list_(pyarrow.float64(), 2)),
            pyarrow.field("y", pyarrow.list_(pyarrow.float64(), 2)),
        ]
    )
    array = pyarrow.array([{"x": [1.0, 2.0], "y": [3.0, 4.0]}], type=struct)
    as_awkward = ak.from_arrow(array)

    assert ak.to_list(as_awkward) == [{"x": [1.0, 2.0], "y": [3.0, 4.0]}]
