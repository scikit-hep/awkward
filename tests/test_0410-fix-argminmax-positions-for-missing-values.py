# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

to_list = ak._v2.operations.to_list


def test_jagged_axis0():
    assert ak._v2.operations.min(
        ak._v2.highlevel.Array([[1.1, 5.5], [4.4], [2.2, 3.3, 0.0, -10]]), axis=0
    ).tolist() == [1.1, 3.3, 0, -10]
    assert ak._v2.operations.argmin(
        ak._v2.highlevel.Array([[1.1, 5.5], [4.4], [2.2, 3.3, 0.0, -10]]), axis=0
    ).tolist() == [0, 2, 2, 2]


def test_jagged_axis1():
    # first is [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]]

    array = ak._v2.highlevel.Array(
        [
            [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
            [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
        ]
    )
    assert ak._v2.operations.min(array, axis=1).tolist() == [
        [1, 2, 3.3],
        [1, 2, 3.3],
    ]
    assert ak._v2.operations.argmin(array, axis=1).tolist() == [
        [4, 3, 2],
        [4, 3, 2],
    ]

    array = ak._v2.highlevel.Array(
        [
            [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
            [[], [1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
        ]
    )
    assert ak._v2.operations.min(array, axis=1).tolist() == [
        [1, 2, 3.3],
        [1, 2, 3.3],
    ]
    assert ak._v2.operations.argmin(array, axis=1).tolist() == [
        [4, 3, 2],
        [5, 4, 3],
    ]

    array = ak._v2.highlevel.Array(
        [
            [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
            [[], [], [1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
        ]
    )
    assert ak._v2.operations.min(array, axis=1).tolist() == [
        [1, 2, 3.3],
        [1, 2, 3.3],
    ]
    assert ak._v2.operations.argmin(array, axis=1).tolist() == [
        [4, 3, 2],
        [6, 5, 4],
    ]

    array = ak._v2.highlevel.Array(
        [
            [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
            [[1.1], [1.1, 2.2], [], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
        ]
    )
    assert ak._v2.operations.min(array, axis=1).tolist() == [
        [1, 2, 3.3],
        [1, 2, 3.3],
    ]
    assert ak._v2.operations.argmin(array, axis=1).tolist() == [
        [4, 3, 2],
        [5, 4, 3],
    ]

    array = ak._v2.highlevel.Array(
        [
            [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
            [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [], [999, 2.0], [1.0]],
        ]
    )
    assert ak._v2.operations.min(array, axis=1).tolist() == [
        [1, 2, 3.3],
        [1, 2, 3.3],
    ]
    assert ak._v2.operations.argmin(array, axis=1).tolist() == [
        [4, 3, 2],
        [5, 4, 2],
    ]

    array = ak._v2.highlevel.Array(
        [
            [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
            [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [], [1.0]],
        ]
    )
    assert ak._v2.operations.min(array, axis=1).tolist() == [
        [1, 2, 3.3],
        [1, 2, 3.3],
    ]
    assert ak._v2.operations.argmin(array, axis=1).tolist() == [
        [4, 3, 2],
        [5, 3, 2],
    ]

    array = ak._v2.highlevel.Array(
        [
            [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
            [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0], []],
        ]
    )
    assert ak._v2.operations.min(array, axis=1).tolist() == [
        [1, 2, 3.3],
        [1, 2, 3.3],
    ]
    assert ak._v2.operations.argmin(array, axis=1).tolist() == [
        [4, 3, 2],
        [4, 3, 2],
    ]

    array = ak._v2.highlevel.Array(
        [
            [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
            [[1.1, 999, 999], [1.1, 2.2, 999], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
        ]
    )
    assert ak._v2.operations.min(array, axis=1).tolist() == [
        [1, 2, 3.3],
        [1, 2, 3.3],
    ]
    assert ak._v2.operations.argmin(array, axis=1).tolist() == [
        [4, 3, 2],
        [4, 3, 2],
    ]

    array = ak._v2.highlevel.Array(
        [
            [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
            [[1.1, 999, 999, 999], [1.1, 2.2, 999], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
        ]
    )
    assert ak._v2.operations.min(array, axis=1).tolist() == [
        [1, 2, 3.3],
        [1, 2, 3.3, 999],
    ]
    assert ak._v2.operations.argmin(array, axis=1).tolist() == [
        [4, 3, 2],
        [4, 3, 2, 0],
    ]

    # first is [[], [1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]]

    array = ak._v2.highlevel.Array(
        [
            [[], [1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
            [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
        ]
    )
    assert ak._v2.operations.min(array, axis=1).tolist() == [
        [1, 2, 3.3],
        [1, 2, 3.3],
    ]
    assert ak._v2.operations.argmin(array, axis=1).tolist() == [
        [5, 4, 3],
        [4, 3, 2],
    ]

    array = ak._v2.highlevel.Array(
        [
            [[], [1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
            [[], [1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
        ]
    )
    assert ak._v2.operations.min(array, axis=1).tolist() == [
        [1, 2, 3.3],
        [1, 2, 3.3],
    ]
    assert ak._v2.operations.argmin(array, axis=1).tolist() == [
        [5, 4, 3],
        [5, 4, 3],
    ]

    array = ak._v2.highlevel.Array(
        [
            [[], [1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
            [[], [], [1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
        ]
    )
    assert ak._v2.operations.min(array, axis=1).tolist() == [
        [1, 2, 3.3],
        [1, 2, 3.3],
    ]
    assert ak._v2.operations.argmin(array, axis=1).tolist() == [
        [5, 4, 3],
        [6, 5, 4],
    ]

    array = ak._v2.highlevel.Array(
        [
            [[], [1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
            [[1.1], [1.1, 2.2], [], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
        ]
    )
    assert ak._v2.operations.min(array, axis=1).tolist() == [
        [1, 2, 3.3],
        [1, 2, 3.3],
    ]
    assert ak._v2.operations.argmin(array, axis=1).tolist() == [
        [5, 4, 3],
        [5, 4, 3],
    ]

    array = ak._v2.highlevel.Array(
        [
            [[], [1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
            [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [], [999, 2.0], [1.0]],
        ]
    )
    assert ak._v2.operations.min(array, axis=1).tolist() == [
        [1, 2, 3.3],
        [1, 2, 3.3],
    ]
    assert ak._v2.operations.argmin(array, axis=1).tolist() == [
        [5, 4, 3],
        [5, 4, 2],
    ]

    array = ak._v2.highlevel.Array(
        [
            [[], [1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
            [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [], [1.0]],
        ]
    )
    assert ak._v2.operations.min(array, axis=1).tolist() == [
        [1, 2, 3.3],
        [1, 2, 3.3],
    ]
    assert ak._v2.operations.argmin(array, axis=1).tolist() == [
        [5, 4, 3],
        [5, 3, 2],
    ]

    array = ak._v2.highlevel.Array(
        [
            [[], [1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
            [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0], []],
        ]
    )
    assert ak._v2.operations.min(array, axis=1).tolist() == [
        [1, 2, 3.3],
        [1, 2, 3.3],
    ]
    assert ak._v2.operations.argmin(array, axis=1).tolist() == [
        [5, 4, 3],
        [4, 3, 2],
    ]

    array = ak._v2.highlevel.Array(
        [
            [[], [1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
            [[1.1, 999, 999], [1.1, 2.2, 999], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
        ]
    )
    assert ak._v2.operations.min(array, axis=1).tolist() == [
        [1, 2, 3.3],
        [1, 2, 3.3],
    ]
    assert ak._v2.operations.argmin(array, axis=1).tolist() == [
        [5, 4, 3],
        [4, 3, 2],
    ]

    array = ak._v2.highlevel.Array(
        [
            [[], [1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
            [[1.1, 999, 999, 999], [1.1, 2.2, 999], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
        ]
    )
    assert ak._v2.operations.min(array, axis=1).tolist() == [
        [1, 2, 3.3],
        [1, 2, 3.3, 999],
    ]
    assert ak._v2.operations.argmin(array, axis=1).tolist() == [
        [5, 4, 3],
        [4, 3, 2, 0],
    ]

    # first is [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [], [999, 2.0], [1.0]]

    array = ak._v2.highlevel.Array(
        [
            [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [], [999, 2.0], [1.0]],
            [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
        ]
    )
    assert ak._v2.operations.min(array, axis=1).tolist() == [
        [1, 2, 3.3],
        [1, 2, 3.3],
    ]
    assert ak._v2.operations.argmin(array, axis=1).tolist() == [
        [5, 4, 2],
        [4, 3, 2],
    ]

    array = ak._v2.highlevel.Array(
        [
            [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [], [999, 2.0], [1.0]],
            [[], [1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
        ]
    )
    assert ak._v2.operations.min(array, axis=1).tolist() == [
        [1, 2, 3.3],
        [1, 2, 3.3],
    ]
    assert ak._v2.operations.argmin(array, axis=1).tolist() == [
        [5, 4, 2],
        [5, 4, 3],
    ]

    array = ak._v2.highlevel.Array(
        [
            [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [], [999, 2.0], [1.0]],
            [[], [], [1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
        ]
    )
    assert ak._v2.operations.min(array, axis=1).tolist() == [
        [1, 2, 3.3],
        [1, 2, 3.3],
    ]
    assert ak._v2.operations.argmin(array, axis=1).tolist() == [
        [5, 4, 2],
        [6, 5, 4],
    ]

    array = ak._v2.highlevel.Array(
        [
            [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [], [999, 2.0], [1.0]],
            [[1.1], [1.1, 2.2], [], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
        ]
    )
    assert ak._v2.operations.min(array, axis=1).tolist() == [
        [1, 2, 3.3],
        [1, 2, 3.3],
    ]
    assert ak._v2.operations.argmin(array, axis=1).tolist() == [
        [5, 4, 2],
        [5, 4, 3],
    ]

    array = ak._v2.highlevel.Array(
        [
            [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [], [999, 2.0], [1.0]],
            [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [], [999, 2.0], [1.0]],
        ]
    )
    assert ak._v2.operations.min(array, axis=1).tolist() == [
        [1, 2, 3.3],
        [1, 2, 3.3],
    ]
    assert ak._v2.operations.argmin(array, axis=1).tolist() == [
        [5, 4, 2],
        [5, 4, 2],
    ]

    array = ak._v2.highlevel.Array(
        [
            [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [], [999, 2.0], [1.0]],
            [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [], [1.0]],
        ]
    )
    assert ak._v2.operations.min(array, axis=1).tolist() == [
        [1, 2, 3.3],
        [1, 2, 3.3],
    ]
    assert ak._v2.operations.argmin(array, axis=1).tolist() == [
        [5, 4, 2],
        [5, 3, 2],
    ]

    array = ak._v2.highlevel.Array(
        [
            [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [], [999, 2.0], [1.0]],
            [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0], []],
        ]
    )
    assert ak._v2.operations.min(array, axis=1).tolist() == [
        [1, 2, 3.3],
        [1, 2, 3.3],
    ]
    assert ak._v2.operations.argmin(array, axis=1).tolist() == [
        [5, 4, 2],
        [4, 3, 2],
    ]

    array = ak._v2.highlevel.Array(
        [
            [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [], [999, 2.0], [1.0]],
            [[1.1, 999, 999], [1.1, 2.2, 999], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
        ]
    )
    assert ak._v2.operations.min(array, axis=1).tolist() == [
        [1, 2, 3.3],
        [1, 2, 3.3],
    ]
    assert ak._v2.operations.argmin(array, axis=1).tolist() == [
        [5, 4, 2],
        [4, 3, 2],
    ]

    array = ak._v2.highlevel.Array(
        [
            [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [], [999, 2.0], [1.0]],
            [[1.1, 999, 999, 999], [1.1, 2.2, 999], [1.1, 2.2, 3.3], [999, 2.0], [1.0]],
        ]
    )
    assert ak._v2.operations.min(array, axis=1).tolist() == [
        [1, 2, 3.3],
        [1, 2, 3.3, 999],
    ]
    assert ak._v2.operations.argmin(array, axis=1).tolist() == [
        [5, 4, 2],
        [4, 3, 2, 0],
    ]


def test_IndexedOptionArray():
    content = ak._v2.highlevel.Array([1.1, 2.2, 3.3, 4.4, 5.5]).layout
    index = ak._v2.index.Index64(np.array([4, 2, -1, -1, 1, 0, 1]))
    array = ak._v2.highlevel.Array(ak._v2.contents.IndexedOptionArray(index, content))
    assert array.tolist() == [5.5, 3.3, None, None, 2.2, 1.1, 2.2]
    assert ak._v2.operations.min(array, axis=0) == 1.1
    assert ak._v2.operations.argmin(array, axis=0) == 5

    assert ak._v2.operations.argmin(
        ak._v2.highlevel.Array([[2.2, 1.1], [None, 3.3], [2.2, 1.1]]), axis=-1
    ).tolist() == [1, 1, 1]
    assert ak._v2.operations.argmin(
        ak._v2.highlevel.Array([[2.2, 1.1], [None, 3.3], [2.2, None, 1.1]]), axis=-1
    ).tolist() == [1, 1, 2]
    assert ak._v2.operations.argmin(
        ak._v2.highlevel.Array([[2.2, 1.1], [3.3, None], [2.2, None, 1.1]]), axis=-1
    ).tolist() == [1, 0, 2]

    assert ak._v2.operations.argmin(
        ak._v2.highlevel.Array([[2.2, 1.1, 0.0], [], [None, 0.5], [2, 1]]), axis=0
    ).tolist() == [3, 2, 0]
    assert ak._v2.operations.argmin(
        ak._v2.highlevel.Array([[2.2, 1.1, 0.0], [], [0.5, None], [2, 1]]), axis=0
    ).tolist() == [2, 3, 0]
    assert ak._v2.operations.argmin(
        ak._v2.highlevel.Array([[2.2, 1.1, 0.0], [0.5, None], [], [2, 1]]), axis=0
    ).tolist() == [1, 3, 0]


def test_ByteMaskedArray():
    content = ak._v2.highlevel.Array([1.1, 2.2, 3.3, 999, 999, 4.4, 5.5]).layout
    mask = ak._v2.index.Index8(
        np.array([False, False, False, True, True, False, False])
    )
    bytemaskedarray = ak._v2.contents.ByteMaskedArray(mask, content, valid_when=False)
    array = ak._v2.highlevel.Array(bytemaskedarray)
    assert array.tolist() == [1.1, 2.2, 3.3, None, None, 4.4, 5.5]
    assert ak._v2.operations.max(array, axis=0) == 5.5
    assert ak._v2.operations.argmax(array, axis=0) == 6

    offsets = ak._v2.index.Index64(np.array([0, 2, 4, 7], dtype=np.int64))
    listoffsetarray = ak._v2.contents.ListOffsetArray(offsets, bytemaskedarray)
    array = ak._v2.highlevel.Array(listoffsetarray)
    assert array.tolist() == [[1.1, 2.2], [3.3, None], [None, 4.4, 5.5]]
    assert ak._v2.operations.max(array, axis=1).tolist() == [2.2, 3.3, 5.5]
    assert ak._v2.operations.argmax(array, axis=1).tolist() == [1, 0, 2]
