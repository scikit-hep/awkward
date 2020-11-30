# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward1 as ak  # noqa: F401


def test():
    a = ak.partition.IrregularlyPartitionedArray(
        [
            ak.Array(
                [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}]
            ).layout,
            ak.Array(
                [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}, {"x": 6, "y": 6.6}]
            ).layout,
        ],
        [3, 6],
    )
    assert [a["y", i] for i in range(6)] == [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
    assert [a[i, "y"] for i in range(6)] == [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
