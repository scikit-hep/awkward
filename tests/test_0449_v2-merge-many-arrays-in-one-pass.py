# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2  # noqa: F401

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test_numpyarray():
    for dtype1 in ("i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f4", "f8", "?"):
        for dtype2 in ("i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f4", "f8", "?"):
            for dtype3 in (
                "i1",
                "i2",
                "i4",
                "i8",
                "u1",
                "u2",
                "u4",
                "u8",
                "f4",
                "f8",
                "?",
            ):
                for dtype4 in (
                    "i1",
                    "i2",
                    "i4",
                    "i8",
                    "u1",
                    "u2",
                    "u4",
                    "u8",
                    "f4",
                    "f8",
                    "?",
                ):
                    one = np.array([0, 1, 2], dtype=dtype1)
                    two = np.array([3, 0], dtype=dtype2)
                    three = np.array([], dtype=dtype3)
                    four = np.array([4, 5, 0, 6, 7], dtype=dtype4)
                    combined = np.concatenate([one, two, three, four])

                    ak_combined = v1_to_v2(ak.layout.NumpyArray(one)).mergemany(
                        [
                            v1_to_v2(ak.layout.NumpyArray(two)),
                            v1_to_v2(ak.layout.NumpyArray(three)),
                            v1_to_v2(ak.layout.NumpyArray(four)),
                        ]
                    )

                    assert ak.to_list(ak_combined) == combined.tolist()
                    assert ak_combined.dtype == combined.dtype

                    ak_combined = v1_to_v2(ak.layout.NumpyArray(one)).mergemany(
                        [
                            v1_to_v2(ak.layout.NumpyArray(two)),
                            v1_to_v2(ak.layout.EmptyArray()),
                            v1_to_v2(ak.layout.NumpyArray(four)),
                        ]
                    )

                    assert ak.to_list(ak_combined) == combined.tolist()
                    assert ak_combined.dtype == np.concatenate([one, two, four]).dtype


def test_lists():
    one = v1_to_v2(ak.Array([[1, 2, 3], [], [4, 5]]).layout)
    two = v1_to_v2(ak.Array([[1.1, 2.2], [3.3, 4.4]]).layout)
    three = v1_to_v2(ak.layout.EmptyArray())
    four = v1_to_v2(
        ak.from_numpy(np.array([[10], [20]]), regulararray=True, highlevel=False)
    )
    assert ak.to_list(one.mergemany([two, three, four])) == [
        [1.0, 2.0, 3.0],
        [],
        [4.0, 5.0],
        [1.1, 2.2],
        [3.3, 4.4],
        [10.0],
        [20.0],
    ]
    assert ak.to_list(four.mergemany([three, two, one])) == [
        [10.0],
        [20.0],
        [1.1, 2.2],
        [3.3, 4.4],
        [1.0, 2.0, 3.0],
        [],
        [4.0, 5.0],
    ]

    one = ak.Array([[1, 2, 3], [], [4, 5]]).layout
    two = ak.Array([[1.1, 2.2], [3.3, 4.4]]).layout
    one = v1_to_v2(ak.layout.ListArray64(one.starts, one.stops, one.content))
    two = v1_to_v2(ak.layout.ListArray64(two.starts, two.stops, two.content))
    assert ak.to_list(one.mergemany([two, three, four])) == [
        [1.0, 2.0, 3.0],
        [],
        [4.0, 5.0],
        [1.1, 2.2],
        [3.3, 4.4],
        [10.0],
        [20.0],
    ]
    assert ak.to_list(four.mergemany([three, two, one])) == [
        [10.0],
        [20.0],
        [1.1, 2.2],
        [3.3, 4.4],
        [1.0, 2.0, 3.0],
        [],
        [4.0, 5.0],
    ]


@pytest.mark.skip(reason="cannot merge NumpyArray with ListOffsetArray")
def test_records():
    one = v1_to_v2(
        ak.Array(
            [{"x": 1, "y": [1]}, {"x": 2, "y": [1, 2]}, {"x": 3, "y": [1, 2, 3]}]
        ).layout
    )
    two = v1_to_v2(ak.Array([{"y": [], "x": 4}, {"y": [3, 2, 1], "x": 5}]).layout)
    three = two[0:0]
    four = v1_to_v2(ak.Array([{"x": 6, "y": [1]}, {"x": 7, "y": [1, 2]}]).layout)
    ak.Array([{"x": 6, "y": [1]}, {"x": 7, "y": [1, 2]}]).layout
    assert ak.to_list(one.mergemany([two, three, four])) == [
        {"x": 1, "y": [1]},
        {"x": 2, "y": [1, 2]},
        {"x": 3, "y": [1, 2, 3]},
        {"y": [], "x": 4},
        {"y": [3, 2, 1], "x": 5},
        {"x": 6, "y": [1]},
        {"x": 7, "y": [1, 2]},
    ]
    assert 1 == 2

    three = ak.layout.EmptyArray()
    assert ak.to_list(one.mergemany([two, three, four])) == [
        {"x": 1, "y": [1]},
        {"x": 2, "y": [1, 2]},
        {"x": 3, "y": [1, 2, 3]},
        {"y": [], "x": 4},
        {"y": [3, 2, 1], "x": 5},
        {"x": 6, "y": [1]},
        {"x": 7, "y": [1, 2]},
    ]


@pytest.mark.skip(reason="cannot merge NumpyArray with ListOffsetArray")
def test_tuples():
    one = v1_to_v2(ak.Array([(1, [1]), (2, [1, 2]), (3, [1, 2, 3])]).layout)
    two = v1_to_v2(ak.Array([(4, []), (5, [3, 2, 1])]).layout)
    three = two[0:0]
    four = v1_to_v2(ak.Array([(6, [1]), (7, [1, 2])]).layout)
    assert ak.to_list(one.mergemany([two, three, four])) == [
        (1, [1]),
        (2, [1, 2]),
        (3, [1, 2, 3]),
        (4, []),
        (5, [3, 2, 1]),
        (6, [1]),
        (7, [1, 2]),
    ]

    three = ak.layout.EmptyArray()
    assert ak.to_list(one.mergemany([two, three, four])) == [
        (1, [1]),
        (2, [1, 2]),
        (3, [1, 2, 3]),
        (4, []),
        (5, [3, 2, 1]),
        (6, [1]),
        (7, [1, 2]),
    ]


def test_indexed():
    one = v1_to_v2(ak.Array([1, 2, 3, None, 4, None, None, 5]).layout)
    two = v1_to_v2(ak.Array([6, 7, 8]).layout)
    three = v1_to_v2(ak.layout.EmptyArray())
    four = v1_to_v2(ak.Array([9, None, None]).layout)
    assert ak.to_list(one.mergemany([two, three, four])) == [
        1,
        2,
        3,
        None,
        4,
        None,
        None,
        5,
        6,
        7,
        8,
        9,
        None,
        None,
    ]


def test_reverse_indexed():
    one = v1_to_v2(ak.Array([1, 2, 3]).layout)
    two = v1_to_v2(ak.Array([4, 5]).layout)
    three = v1_to_v2(ak.Array([None, 6, None]).layout)
    assert ak.to_list(one.mergemany([two, three])) == [1, 2, 3, 4, 5, None, 6, None]

    four = v1_to_v2(ak.Array([7, 8, None, None, 9]).layout)
    assert ak.to_list(one.mergemany([two, three, four])) == [
        1,
        2,
        3,
        4,
        5,
        None,
        6,
        None,
        7,
        8,
        None,
        None,
        9,
    ]


def test_bytemasked():
    one = v1_to_v2(
        ak.Array([1, 2, 3, 4, 5, 6]).mask[[True, True, False, True, False, True]].layout
    )
    two = v1_to_v2(
        ak.Array([7, 99, 999, 8, 9]).mask[[True, False, False, True, True]].layout
    )
    three = v1_to_v2(ak.Array([100, 200, 300]).layout)
    four = v1_to_v2(ak.Array([None, None, 123, None]).layout)
    assert ak.to_list(one.mergemany([two, three, four])) == [
        1,
        2,
        None,
        4,
        None,
        6,
        7,
        None,
        None,
        8,
        9,
        100,
        200,
        300,
        None,
        None,
        123,
        None,
    ]
    assert ak.to_list(four.mergemany([three, two, one])) == [
        None,
        None,
        123,
        None,
        100,
        200,
        300,
        7,
        None,
        None,
        8,
        9,
        1,
        2,
        None,
        4,
        None,
        6,
    ]
    assert ak.to_list(three.mergemany([four, one])) == [
        100,
        200,
        300,
        None,
        None,
        123,
        None,
        1,
        2,
        None,
        4,
        None,
        6,
    ]
    assert ak.to_list(three.mergemany([four, one, two])) == [
        100,
        200,
        300,
        None,
        None,
        123,
        None,
        1,
        2,
        None,
        4,
        None,
        6,
        7,
        None,
        None,
        8,
        9,
    ]
    assert ak.to_list(three.mergemany([two, one])) == [
        100,
        200,
        300,
        7,
        None,
        None,
        8,
        9,
        1,
        2,
        None,
        4,
        None,
        6,
    ]
    assert ak.to_list(three.mergemany([two, one, four])) == [
        100,
        200,
        300,
        7,
        None,
        None,
        8,
        9,
        1,
        2,
        None,
        4,
        None,
        6,
        None,
        None,
        123,
        None,
    ]


def test_empty():
    one = v1_to_v2(ak.layout.EmptyArray())
    two = v1_to_v2(ak.layout.EmptyArray())
    three = v1_to_v2(ak.Array([1, 2, 3]).layout)
    four = v1_to_v2(ak.Array([4, 5]).layout)
    assert ak.to_list(one.mergemany([two])) == []
    assert ak.to_list(one.mergemany([two, one, two, one, two])) == []
    assert ak.to_list(one.mergemany([two, three])) == [1, 2, 3]
    assert ak.to_list(one.mergemany([two, three, four])) == [1, 2, 3, 4, 5]
    assert ak.to_list(one.mergemany([three])) == [1, 2, 3]
    assert ak.to_list(one.mergemany([three, four])) == [1, 2, 3, 4, 5]
    assert ak.to_list(one.mergemany([three, two])) == [1, 2, 3]
    assert ak.to_list(one.mergemany([three, two, four])) == [1, 2, 3, 4, 5]
    assert ak.to_list(one.mergemany([three, four, two])) == [1, 2, 3, 4, 5]


def test_union():
    one = v1_to_v2(ak.Array([1, 2, [], [3, 4]]).layout)
    two = v1_to_v2(ak.Array([100, 200, 300]).layout)
    three = v1_to_v2(ak.Array([{"x": 1}, {"x": 2}, 5, 6, 7]).layout)

    assert ak.to_list(one.mergemany([two, three])) == [
        1,
        2,
        [],
        [3, 4],
        100,
        200,
        300,
        {"x": 1},
        {"x": 2},
        5,
        6,
        7,
    ]
    assert ak.to_list(one.mergemany([three, two])) == [
        1,
        2,
        [],
        [3, 4],
        {"x": 1},
        {"x": 2},
        5,
        6,
        7,
        100,
        200,
        300,
    ]
    assert ak.to_list(two.mergemany([one, three])) == [
        100,
        200,
        300,
        1,
        2,
        [],
        [3, 4],
        {"x": 1},
        {"x": 2},
        5,
        6,
        7,
    ]
    assert ak.to_list(two.mergemany([three, one])) == [
        100,
        200,
        300,
        {"x": 1},
        {"x": 2},
        5,
        6,
        7,
        1,
        2,
        [],
        [3, 4],
    ]
    assert ak.to_list(three.mergemany([one, two])) == [
        {"x": 1},
        {"x": 2},
        5,
        6,
        7,
        1,
        2,
        [],
        [3, 4],
        100,
        200,
        300,
    ]
    assert ak.to_list(three.mergemany([two, one])) == [
        {"x": 1},
        {"x": 2},
        5,
        6,
        7,
        100,
        200,
        300,
        1,
        2,
        [],
        [3, 4],
    ]


def test_union_option():
    one = v1_to_v2(ak.Array([1, 2, [], [3, 4]]).layout)
    two = v1_to_v2(ak.Array([100, None, 300]).layout)
    three = v1_to_v2(ak.Array([{"x": 1}, {"x": 2}, 5, 6, 7]).layout)

    assert ak.to_list(one.mergemany([two, three])) == [
        1,
        2,
        [],
        [3, 4],
        100,
        None,
        300,
        {"x": 1},
        {"x": 2},
        5,
        6,
        7,
    ]
    assert ak.to_list(one.mergemany([three, two])) == [
        1,
        2,
        [],
        [3, 4],
        {"x": 1},
        {"x": 2},
        5,
        6,
        7,
        100,
        None,
        300,
    ]
    assert ak.to_list(two.mergemany([one, three])) == [
        100,
        None,
        300,
        1,
        2,
        [],
        [3, 4],
        {"x": 1},
        {"x": 2},
        5,
        6,
        7,
    ]
    assert ak.to_list(two.mergemany([three, one])) == [
        100,
        None,
        300,
        {"x": 1},
        {"x": 2},
        5,
        6,
        7,
        1,
        2,
        [],
        [3, 4],
    ]
    assert ak.to_list(three.mergemany([one, two])) == [
        {"x": 1},
        {"x": 2},
        5,
        6,
        7,
        1,
        2,
        [],
        [3, 4],
        100,
        None,
        300,
    ]
    assert ak.to_list(three.mergemany([two, one])) == [
        {"x": 1},
        {"x": 2},
        5,
        6,
        7,
        100,
        None,
        300,
        1,
        2,
        [],
        [3, 4],
    ]

    one = v1_to_v2(ak.Array([1, 2, [], [3, 4]]).layout)
    two = v1_to_v2(ak.Array([100, None, 300]).layout)
    three = v1_to_v2(ak.Array([{"x": 1}, {"x": 2}, 5, None, 7]).layout)

    assert ak.to_list(one.mergemany([two, three])) == [
        1,
        2,
        [],
        [3, 4],
        100,
        None,
        300,
        {"x": 1},
        {"x": 2},
        5,
        None,
        7,
    ]
    assert ak.to_list(one.mergemany([three, two])) == [
        1,
        2,
        [],
        [3, 4],
        {"x": 1},
        {"x": 2},
        5,
        None,
        7,
        100,
        None,
        300,
    ]
    assert ak.to_list(two.mergemany([one, three])) == [
        100,
        None,
        300,
        1,
        2,
        [],
        [3, 4],
        {"x": 1},
        {"x": 2},
        5,
        None,
        7,
    ]
    assert ak.to_list(two.mergemany([three, one])) == [
        100,
        None,
        300,
        {"x": 1},
        {"x": 2},
        5,
        None,
        7,
        1,
        2,
        [],
        [3, 4],
    ]
    assert ak.to_list(three.mergemany([one, two])) == [
        {"x": 1},
        {"x": 2},
        5,
        None,
        7,
        1,
        2,
        [],
        [3, 4],
        100,
        None,
        300,
    ]
    assert ak.to_list(three.mergemany([two, one])) == [
        {"x": 1},
        {"x": 2},
        5,
        None,
        7,
        100,
        None,
        300,
        1,
        2,
        [],
        [3, 4],
    ]

    one = v1_to_v2(ak.Array([1, 2, [], [3, 4]]).layout)
    two = v1_to_v2(ak.Array([100, 200, 300]).layout)
    three = v1_to_v2(ak.Array([{"x": 1}, {"x": 2}, 5, None, 7]).layout)

    assert ak.to_list(one.mergemany([two, three])) == [
        1,
        2,
        [],
        [3, 4],
        100,
        200,
        300,
        {"x": 1},
        {"x": 2},
        5,
        None,
        7,
    ]
    assert ak.to_list(one.mergemany([three, two])) == [
        1,
        2,
        [],
        [3, 4],
        {"x": 1},
        {"x": 2},
        5,
        None,
        7,
        100,
        200,
        300,
    ]
    assert ak.to_list(two.mergemany([one, three])) == [
        100,
        200,
        300,
        1,
        2,
        [],
        [3, 4],
        {"x": 1},
        {"x": 2},
        5,
        None,
        7,
    ]
    assert ak.to_list(two.mergemany([three, one])) == [
        100,
        200,
        300,
        {"x": 1},
        {"x": 2},
        5,
        None,
        7,
        1,
        2,
        [],
        [3, 4],
    ]
    assert ak.to_list(three.mergemany([one, two])) == [
        {"x": 1},
        {"x": 2},
        5,
        None,
        7,
        1,
        2,
        [],
        [3, 4],
        100,
        200,
        300,
    ]
    assert ak.to_list(three.mergemany([two, one])) == [
        {"x": 1},
        {"x": 2},
        5,
        None,
        7,
        100,
        200,
        300,
        1,
        2,
        [],
        [3, 4],
    ]


def test_strings():
    one = ak.Array(["uno", "dos", "tres"]).layout
    two = ak.Array(["un", "deux", "trois", "quatre"]).layout
    three = ak.Array(["onay", "ootay", "eethray"]).layout
    assert ak.to_list(one.mergemany([two, three])) == [
        "uno",
        "dos",
        "tres",
        "un",
        "deux",
        "trois",
        "quatre",
        "onay",
        "ootay",
        "eethray",
    ]


def test_concatenate():
    one = v1_to_v2(ak.Array([1, 2, 3]).layout)
    two = v1_to_v2(ak.Array([4.4, 5.5]).layout)
    three = v1_to_v2(ak.Array([6, 7, 8]).layout)
    four = v1_to_v2(ak.Array([[9, 9, 9], [10, 10, 10]]).layout)

    assert ak.to_list(one.concatenate([two, three, four])) == [
        1,
        2,
        3,
        4.4,
        5.5,
        6,
        7,
        8,
        [9, 9, 9],
        [10, 10, 10],
    ]
    assert ak.to_list(four.concatenate([one, two, three])) == [
        [9, 9, 9],
        [10, 10, 10],
        1,
        2,
        3,
        4.4,
        5.5,
        6,
        7,
        8,
    ]
    assert ak.to_list(one.concatenate([two, four, three])) == [
        1,
        2,
        3,
        4.4,
        5.5,
        [9, 9, 9],
        [10, 10, 10],
        6,
        7,
        8,
    ]

    five = v1_to_v2(ak.Array(["nine", "ten"]).layout)
    assert ak.to_list(one.concatenate([two, three, five])) == [
        1,
        2,
        3,
        4.4,
        5.5,
        6,
        7,
        8,
        "nine",
        "ten",
    ]
    assert ak.to_list(five.concatenate([one, two, three])) == [
        "nine",
        "ten",
        1,
        2,
        3,
        4.4,
        5.5,
        6,
        7,
        8,
    ]
    assert ak.to_list(one.concatenate([two, five, three])) == [
        1,
        2,
        3,
        4.4,
        5.5,
        "nine",
        "ten",
        6,
        7,
        8,
    ]
