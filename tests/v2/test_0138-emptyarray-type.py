# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test():
    empty1 = ak._v2.highlevel.Array(ak._v2.contents.EmptyArray(), check_valid=True)
    empty2 = ak._v2.highlevel.Array(
        ak._v2.contents.ListOffsetArray(
            ak._v2.index.Index64(np.array([0, 0, 0, 0], dtype=np.int64)),
            ak._v2.contents.EmptyArray(),
        ),
        check_valid=False,
    )
    array = ak._v2.highlevel.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], check_valid=True)

    assert ak._v2.operations.convert.to_numpy(empty1).dtype.type is np.float64

    assert ak.to_list(array[empty1]) == [[], [], []]
    assert (
        ak.to_list(
            array[
                empty1,
            ]
        )
        == [[], [], []]
    )
    assert ak.to_list(array[empty2]) == [[], [], []]
    assert (
        ak.to_list(
            array[
                empty2,
            ]
        )
        == [[], [], []]
    )
