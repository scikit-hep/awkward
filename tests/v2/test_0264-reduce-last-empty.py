# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test():
    assert ak.to_list(
        ak._v2.operations.reducers.prod(
            ak._v2.highlevel.Array([[[2, 3, 5]], [[7], [11]], [[]]]), axis=-1
        )
    ) == [
        [30],
        [7, 11],
        [1],
    ]

    assert ak.to_list(
        ak._v2.operations.reducers.prod(
            ak._v2.highlevel.Array([[[2, 3, 5]], [[7], [11]], []]), axis=-1
        )
    ) == [
        [30],
        [7, 11],
        [],
    ]
