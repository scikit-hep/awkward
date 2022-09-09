# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401
import itertools

to_list = ak._v2.operations.to_list


def test_numpyarray():
    for dtype1, dtype2, dtype3, dtype4 in itertools.product(
        ("i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f4", "f8", "?"), repeat=4
    ):
        one = np.array([0, 1, 2], dtype=dtype1)
        two = np.array([3, 0], dtype=dtype2)
        three = np.array([], dtype=dtype3)
        four = np.array([4, 5, 0, 6, 7], dtype=dtype4)
        combined = np.concatenate([one, two, three, four])

        ak_combined = ak._v2.contents.NumpyArray(one).mergemany(
            [
                ak._v2.contents.NumpyArray(two),
                ak._v2.contents.NumpyArray(three),
                ak._v2.contents.NumpyArray(four),
            ]
        )

        assert to_list(ak_combined) == combined.tolist()
        assert ak_combined.dtype == combined.dtype

        assert (
            ak._v2.contents.NumpyArray(one)
            .typetracer.mergemany(
                [
                    ak._v2.contents.NumpyArray(two),
                    ak._v2.contents.NumpyArray(three),
                    ak._v2.contents.NumpyArray(four),
                ]
            )
            .form
            == ak._v2.contents.NumpyArray(one)
            .mergemany(
                [
                    ak._v2.contents.NumpyArray(two),
                    ak._v2.contents.NumpyArray(three),
                    ak._v2.contents.NumpyArray(four),
                ]
            )
            .form
        )

        ak_combined = ak._v2.contents.NumpyArray(one).mergemany(
            [
                ak._v2.contents.NumpyArray(two),
                ak._v2.contents.EmptyArray(),
                ak._v2.contents.NumpyArray(four),
            ]
        )

        assert to_list(ak_combined) == combined.tolist()
        assert ak_combined.dtype == np.concatenate([one, two, four]).dtype

        assert (
            ak._v2.contents.NumpyArray(one)
            .typetracer.mergemany(
                [
                    ak._v2.contents.NumpyArray(two),
                    ak._v2.contents.EmptyArray(),
                    ak._v2.contents.NumpyArray(four),
                ]
            )
            .form
            == ak._v2.contents.NumpyArray(one)
            .mergemany(
                [
                    ak._v2.contents.NumpyArray(two),
                    ak._v2.contents.EmptyArray(),
                    ak._v2.contents.NumpyArray(four),
                ]
            )
            .form
        )


def test_lists():
    one = ak._v2.highlevel.Array([[1, 2, 3], [], [4, 5]]).layout
    two = ak._v2.highlevel.Array([[1.1, 2.2], [3.3, 4.4]]).layout
    three = ak._v2.contents.EmptyArray()
    four = ak._v2.operations.from_numpy(
        np.array([[10], [20]]), regulararray=True, highlevel=False
    )
    assert to_list(one.mergemany([two, three, four])) == [
        [1.0, 2.0, 3.0],
        [],
        [4.0, 5.0],
        [1.1, 2.2],
        [3.3, 4.4],
        [10.0],
        [20.0],
    ]
    assert to_list(four.mergemany([three, two, one])) == [
        [10.0],
        [20.0],
        [1.1, 2.2],
        [3.3, 4.4],
        [1.0, 2.0, 3.0],
        [],
        [4.0, 5.0],
    ]
    assert (
        one.typetracer.mergemany([two, three, four]).form
        == one.mergemany([two, three, four]).form
    )
    assert (
        four.typetracer.mergemany([three, four, one]).form
        == four.mergemany([three, four, one]).form
    )

    one = ak._v2.highlevel.Array([[1, 2, 3], [], [4, 5]]).layout
    two = ak._v2.highlevel.Array([[1.1, 2.2], [3.3, 4.4]]).layout
    one = ak._v2.contents.ListArray(one.starts, one.stops, one.content)
    two = ak._v2.contents.ListArray(two.starts, two.stops, two.content)
    assert to_list(one.mergemany([two, three, four])) == [
        [1.0, 2.0, 3.0],
        [],
        [4.0, 5.0],
        [1.1, 2.2],
        [3.3, 4.4],
        [10.0],
        [20.0],
    ]
    assert to_list(four.mergemany([three, two, one])) == [
        [10.0],
        [20.0],
        [1.1, 2.2],
        [3.3, 4.4],
        [1.0, 2.0, 3.0],
        [],
        [4.0, 5.0],
    ]

    assert (
        one.typetracer.mergemany([two, three, four]).form
        == one.mergemany([two, three, four]).form
    )

    assert (
        four.typetracer.mergemany([three, four, one]).form
        == four.mergemany([three, four, one]).form
    )


def test_records():
    one = ak._v2.highlevel.Array(
        [{"x": 1, "y": [1]}, {"x": 2, "y": [1, 2]}, {"x": 3, "y": [1, 2, 3]}]
    ).layout
    two = ak._v2.highlevel.Array([{"y": [], "x": 4}, {"y": [3, 2, 1], "x": 5}]).layout
    three = two[0:0]
    four = ak._v2.highlevel.Array([{"x": 6, "y": [1]}, {"x": 7, "y": [1, 2]}]).layout
    # ak._v2.highlevel.Array([{"x": 6, "y": [1]}, {"x": 7, "y": [1, 2]}]).layout
    assert to_list(one.mergemany([two, three, four])) == [
        {"x": 1, "y": [1]},
        {"x": 2, "y": [1, 2]},
        {"x": 3, "y": [1, 2, 3]},
        {"y": [], "x": 4},
        {"y": [3, 2, 1], "x": 5},
        {"x": 6, "y": [1]},
        {"x": 7, "y": [1, 2]},
    ]
    assert (
        one.typetracer.mergemany([two, three, four]).form
        == one.mergemany([two, three, four]).form
    )

    three = ak._v2.contents.EmptyArray()
    assert to_list(one.mergemany([two, three, four])) == [
        {"x": 1, "y": [1]},
        {"x": 2, "y": [1, 2]},
        {"x": 3, "y": [1, 2, 3]},
        {"y": [], "x": 4},
        {"y": [3, 2, 1], "x": 5},
        {"x": 6, "y": [1]},
        {"x": 7, "y": [1, 2]},
    ]
    assert (
        one.typetracer.mergemany([two, three, four]).form
        == one.mergemany([two, three, four]).form
    )


def test_tuples():
    one = ak._v2.highlevel.Array([(1, [1]), (2, [1, 2]), (3, [1, 2, 3])]).layout
    two = ak._v2.highlevel.Array([(4, []), (5, [3, 2, 1])]).layout
    three = two[0:0]
    four = ak._v2.highlevel.Array([(6, [1]), (7, [1, 2])]).layout
    assert to_list(one.mergemany([two, three, four])) == [
        (1, [1]),
        (2, [1, 2]),
        (3, [1, 2, 3]),
        (4, []),
        (5, [3, 2, 1]),
        (6, [1]),
        (7, [1, 2]),
    ]
    assert (
        one.typetracer.mergemany([two, three, four]).form
        == one.mergemany([two, three, four]).form
    )

    three = ak._v2.contents.EmptyArray()
    assert to_list(one.mergemany([two, three, four])) == [
        (1, [1]),
        (2, [1, 2]),
        (3, [1, 2, 3]),
        (4, []),
        (5, [3, 2, 1]),
        (6, [1]),
        (7, [1, 2]),
    ]
    assert (
        one.typetracer.mergemany([two, three, four]).form
        == one.mergemany([two, three, four]).form
    )


def test_indexed():
    one = ak._v2.highlevel.Array([1, 2, 3, None, 4, None, None, 5]).layout
    two = ak._v2.highlevel.Array([6, 7, 8]).layout
    three = ak._v2.contents.EmptyArray()
    four = ak._v2.highlevel.Array([9, None, None]).layout
    assert to_list(one.mergemany([two, three, four])) == [
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
    assert (
        one.typetracer.mergemany([two, three, four]).form
        == one.mergemany([two, three, four]).form
    )


def test_reverse_indexed():
    one = ak._v2.highlevel.Array([1, 2, 3]).layout
    two = ak._v2.highlevel.Array([4, 5]).layout
    three = ak._v2.highlevel.Array([None, 6, None]).layout
    assert to_list(one.mergemany([two, three])) == [1, 2, 3, 4, 5, None, 6, None]
    assert (
        one.typetracer.mergemany([two, three]).form == one.mergemany([two, three]).form
    )

    four = ak._v2.highlevel.Array([7, 8, None, None, 9]).layout
    assert to_list(one.mergemany([two, three, four])) == [
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
    assert (
        one.typetracer.mergemany([two, three, four]).form
        == one.mergemany([two, three, four]).form
    )


def test_bytemasked():
    one = ak._v2.contents.ByteMaskedArray(
        ak._v2.index.Index8([True, True, False, True, False, True]),
        ak._v2.highlevel.Array([1, 2, 3, 4, 5, 6]).layout,
        valid_when=True,
    )
    two = ak._v2.contents.ByteMaskedArray(
        ak._v2.index.Index8([True, False, False, True, True]),
        ak._v2.highlevel.Array([7, 99, 999, 8, 9]).layout,
        valid_when=True,
    )
    three = ak._v2.highlevel.Array([100, 200, 300]).layout
    four = ak._v2.highlevel.Array([None, None, 123, None]).layout
    assert to_list(one.mergemany([two, three, four])) == [
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
    assert to_list(four.mergemany([three, two, one])) == [
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
    assert to_list(three.mergemany([four, one])) == [
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
    assert to_list(three.mergemany([four, one, two])) == [
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
    assert to_list(three.mergemany([two, one])) == [
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
    assert to_list(three.mergemany([two, one, four])) == [
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

    assert (
        one.typetracer.mergemany([two, three, four]).form
        == one.mergemany([two, three, four]).form
    )
    assert (
        four.typetracer.mergemany([three, two, one]).form
        == four.mergemany([three, two, one]).form
    )
    assert (
        three.typetracer.mergemany([four, one]).form
        == three.mergemany([four, one]).form
    )
    assert (
        three.typetracer.mergemany([four, one, two]).form
        == three.mergemany([four, one, two]).form
    )
    assert (
        three.typetracer.mergemany([two, one]).form == three.mergemany([two, one]).form
    )
    assert (
        three.typetracer.mergemany([two, one, four]).form
        == three.mergemany([two, one, four]).form
    )


def test_empty():
    one = ak._v2.contents.EmptyArray()
    two = ak._v2.contents.EmptyArray()
    three = ak._v2.highlevel.Array([1, 2, 3]).layout
    four = ak._v2.highlevel.Array([4, 5]).layout
    assert to_list(one.mergemany([two])) == []
    assert to_list(one.mergemany([two, one, two, one, two])) == []
    assert to_list(one.mergemany([two, three])) == [1, 2, 3]
    assert to_list(one.mergemany([two, three, four])) == [1, 2, 3, 4, 5]
    assert to_list(one.mergemany([three])) == [1, 2, 3]
    assert to_list(one.mergemany([three, four])) == [1, 2, 3, 4, 5]
    assert to_list(one.mergemany([three, two])) == [1, 2, 3]
    assert to_list(one.mergemany([three, two, four])) == [1, 2, 3, 4, 5]
    assert to_list(one.mergemany([three, four, two])) == [1, 2, 3, 4, 5]

    assert one.typetracer.mergemany([two]).form == one.mergemany([two]).form
    assert (
        one.typetracer.mergemany([two, one, two, one, two]).form
        == one.mergemany([two, one, two, one, two]).form
    )
    assert (
        one.typetracer.mergemany([two, three]).form == one.mergemany([two, three]).form
    )
    assert (
        one.typetracer.mergemany([two, three, four]).form
        == one.mergemany([two, three, four]).form
    )
    assert one.typetracer.mergemany([three]).form == one.mergemany([three]).form
    assert (
        one.typetracer.mergemany([three, four]).form
        == one.mergemany([three, four]).form
    )
    assert (
        one.typetracer.mergemany([three, two]).form == one.mergemany([three, two]).form
    )
    assert (
        one.typetracer.mergemany([three, two, four]).form
        == one.mergemany([three, two, four]).form
    )
    assert (
        one.typetracer.mergemany([three, four, two]).form
        == one.mergemany([three, four, two]).form
    )


def test_union():
    one = ak._v2.highlevel.Array([1, 2, [], [3, 4]]).layout
    two = ak._v2.highlevel.Array([100, 200, 300]).layout
    three = ak._v2.highlevel.Array([{"x": 1}, {"x": 2}, 5, 6, 7]).layout

    assert to_list(one.mergemany([two, three])) == [
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
    assert to_list(one.mergemany([three, two])) == [
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
    assert to_list(two.mergemany([one, three])) == [
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
    assert to_list(two.mergemany([three, one])) == [
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
    assert to_list(three.mergemany([one, two])) == [
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
    assert to_list(three.mergemany([two, one])) == [
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

    assert (
        one.typetracer.mergemany([two, three]).form == one.mergemany([two, three]).form
    )
    assert (
        one.typetracer.mergemany([three, two]).form == one.mergemany([three, two]).form
    )
    assert (
        two.typetracer.mergemany([one, three]).form == two.mergemany([one, three]).form
    )
    assert (
        two.typetracer.mergemany([three, one]).form == two.mergemany([three, one]).form
    )
    assert (
        three.typetracer.mergemany([one, two]).form == three.mergemany([one, two]).form
    )
    assert (
        three.typetracer.mergemany([two, one]).form == three.mergemany([two, one]).form
    )


def test_union_option():
    one = ak._v2.highlevel.Array([1, 2, [], [3, 4]]).layout
    two = ak._v2.highlevel.Array([100, None, 300]).layout
    three = ak._v2.highlevel.Array([{"x": 1}, {"x": 2}, 5, 6, 7]).layout

    assert to_list(one.mergemany([two, three])) == [
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
    assert to_list(one.mergemany([three, two])) == [
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
    assert to_list(two.mergemany([one, three])) == [
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
    assert to_list(two.mergemany([three, one])) == [
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
    assert to_list(three.mergemany([one, two])) == [
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
    assert to_list(three.mergemany([two, one])) == [
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

    assert (
        one.typetracer.mergemany([two, three]).form == one.mergemany([two, three]).form
    )
    assert (
        one.typetracer.mergemany([three, two]).form == one.mergemany([three, two]).form
    )
    assert (
        two.typetracer.mergemany([one, three]).form == two.mergemany([one, three]).form
    )
    assert (
        two.typetracer.mergemany([three, one]).form == two.mergemany([three, one]).form
    )
    assert (
        three.typetracer.mergemany([one, two]).form == three.mergemany([one, two]).form
    )
    assert (
        three.typetracer.mergemany([two, one]).form == three.mergemany([two, one]).form
    )

    one = ak._v2.highlevel.Array([1, 2, [], [3, 4]]).layout
    two = ak._v2.highlevel.Array([100, None, 300]).layout
    three = ak._v2.highlevel.Array([{"x": 1}, {"x": 2}, 5, None, 7]).layout

    assert to_list(one.mergemany([two, three])) == [
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
    assert to_list(one.mergemany([three, two])) == [
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
    assert to_list(two.mergemany([one, three])) == [
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
    assert to_list(two.mergemany([three, one])) == [
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
    assert to_list(three.mergemany([one, two])) == [
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
    assert to_list(three.mergemany([two, one])) == [
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

    assert (
        one.typetracer.mergemany([two, three]).form == one.mergemany([two, three]).form
    )
    assert (
        one.typetracer.mergemany([three, two]).form == one.mergemany([three, two]).form
    )
    assert (
        two.typetracer.mergemany([one, three]).form == two.mergemany([one, three]).form
    )
    assert (
        two.typetracer.mergemany([three, one]).form == two.mergemany([three, one]).form
    )
    assert (
        three.typetracer.mergemany([one, two]).form == three.mergemany([one, two]).form
    )
    assert (
        three.typetracer.mergemany([two, one]).form == three.mergemany([two, one]).form
    )

    one = ak._v2.highlevel.Array([1, 2, [], [3, 4]]).layout
    two = ak._v2.highlevel.Array([100, 200, 300]).layout
    three = ak._v2.highlevel.Array([{"x": 1}, {"x": 2}, 5, None, 7]).layout

    assert to_list(one.mergemany([two, three])) == [
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
    assert to_list(one.mergemany([three, two])) == [
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
    assert to_list(two.mergemany([one, three])) == [
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
    assert to_list(two.mergemany([three, one])) == [
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
    assert to_list(three.mergemany([one, two])) == [
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
    assert to_list(three.mergemany([two, one])) == [
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

    assert (
        one.typetracer.mergemany([two, three]).form == one.mergemany([two, three]).form
    )
    assert (
        one.typetracer.mergemany([three, two]).form == one.mergemany([three, two]).form
    )
    assert (
        two.typetracer.mergemany([one, three]).form == two.mergemany([one, three]).form
    )
    assert (
        two.typetracer.mergemany([three, one]).form == two.mergemany([three, one]).form
    )
    assert (
        three.typetracer.mergemany([one, two]).form == three.mergemany([one, two]).form
    )
    assert (
        three.typetracer.mergemany([two, one]).form == three.mergemany([two, one]).form
    )


def test_strings():
    one = ak._v2.highlevel.Array(["uno", "dos", "tres"]).layout
    two = ak._v2.highlevel.Array(["un", "deux", "trois", "quatre"]).layout
    three = ak._v2.highlevel.Array(["onay", "ootay", "eethray"]).layout
    assert to_list(one.mergemany([two, three])) == [
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

    assert (
        one.typetracer.mergemany([two, three]).form == one.mergemany([two, three]).form
    )


def test_concatenate():
    one = ak._v2.highlevel.Array([1, 2, 3]).layout
    two = ak._v2.highlevel.Array([4.4, 5.5]).layout
    three = ak._v2.highlevel.Array([6, 7, 8]).layout
    four = ak._v2.highlevel.Array([[9, 9, 9], [10, 10, 10]]).layout

    assert to_list(ak._v2.operations.concatenate([one, two, three, four])) == [
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
    assert to_list(ak._v2.operations.concatenate([four, one, two, three])) == [
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
    assert to_list(ak._v2.operations.concatenate([one, two, four, three])) == [
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

    five = ak._v2.highlevel.Array(["nine", "ten"]).layout
    assert to_list(ak._v2.operations.concatenate([one, two, three, five])) == [
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
    assert to_list(ak._v2.operations.concatenate([five, one, two, three])) == [
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
    assert to_list(ak._v2.operations.concatenate([one, two, five, three])) == [
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
