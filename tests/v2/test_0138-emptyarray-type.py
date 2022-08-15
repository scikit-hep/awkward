# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

to_list = ak._v2.operations.to_list


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

    assert ak._v2.operations.to_numpy(empty1).dtype.type is np.float64

    assert to_list(array[empty1]) == []
    assert (
        to_list(
            array[
                empty1,
            ]
        )
        == []
    )
    assert to_list(array[empty2]) == [[], [], []]
    assert to_list(
        array[
            empty2,
        ]
    ) == [[], [], []]
