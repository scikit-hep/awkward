# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    empty1 = ak.Array(ak.layout.EmptyArray(), check_valid=True)
    empty2 = ak.Array(
        ak.layout.ListOffsetArray64(
            ak.layout.Index64(np.array([0, 0, 0, 0], dtype=np.int64)),
            ak.layout.EmptyArray(),
        ),
        check_valid=True,
    )
    array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], check_valid=True)

    assert ak.to_numpy(empty1).dtype.type is np.float64

    assert ak.to_list(array[empty1]) == []
    assert (
        ak.to_list(
            array[
                empty1,
            ]
        )
        == []
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
