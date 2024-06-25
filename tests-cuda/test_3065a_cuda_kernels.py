from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

import awkward as ak
from awkward.operations.ak_concatenate import enforce_concatenated_form

to_list = ak.operations.to_list


@pytest.fixture(scope="function", autouse=True)
def cleanup_cuda():
    yield
    cp._default_memory_pool.free_all_blocks()
    cp.cuda.Device().synchronize()


def test_0449_merge_many_arrays_in_one_pass_concatenate():
    one = ak.highlevel.Array([1, 2, 3]).layout
    two = ak.highlevel.Array([4.4, 5.5]).layout
    three = ak.highlevel.Array([6, 7, 8]).layout
    four = ak.highlevel.Array([[9, 9, 9], [10, 10, 10]]).layout

    cuda_one = ak.to_backend(one, "cuda")
    cuda_two = ak.to_backend(two, "cuda")
    cuda_three = ak.to_backend(three, "cuda")
    cuda_four = ak.to_backend(four, "cuda")

    assert to_list(
        ak.operations.concatenate([cuda_one, cuda_two, cuda_three, cuda_four])
    ) == [
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
    assert to_list(
        ak.operations.concatenate([cuda_four, cuda_one, cuda_two, cuda_three])
    ) == [
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
    assert to_list(
        ak.operations.concatenate([cuda_one, cuda_two, cuda_four, cuda_three])
    ) == [
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

    five = ak.highlevel.Array(["nine", "ten"]).layout
    cuda_five = ak.to_backend(five, "cuda")

    assert to_list(
        ak.operations.concatenate([cuda_one, cuda_two, cuda_three, cuda_five])
    ) == [
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
    assert to_list(
        ak.operations.concatenate([cuda_five, cuda_one, cuda_two, cuda_three])
    ) == [
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
    assert to_list(
        ak.operations.concatenate([cuda_one, cuda_two, cuda_five, cuda_three])
    ) == [
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


def test_0449_merge_many_arrays_in_one_pass_lists():
    one = ak.highlevel.Array([[1, 2, 3], [], [4, 5]]).layout
    two = ak.highlevel.Array([[1.1, 2.2], [3.3, 4.4]]).layout
    three = ak.contents.EmptyArray()
    four = ak.operations.from_numpy(
        np.array([[10], [20]]), regulararray=True, highlevel=False
    )

    cuda_one = ak.to_backend(one, "cuda", highlevel=False)
    cuda_two = ak.to_backend(two, "cuda", highlevel=False)
    cuda_three = ak.to_backend(three, "cuda", highlevel=False)
    cuda_four = ak.to_backend(four, "cuda", highlevel=False)

    assert to_list(cuda_one._mergemany([cuda_two, cuda_three, cuda_four])) == [
        [1.0, 2.0, 3.0],
        [],
        [4.0, 5.0],
        [1.1, 2.2],
        [3.3, 4.4],
        [10.0],
        [20.0],
    ]
    assert to_list(cuda_four._mergemany([cuda_three, cuda_two, cuda_one])) == [
        [10.0],
        [20.0],
        [1.1, 2.2],
        [3.3, 4.4],
        [1.0, 2.0, 3.0],
        [],
        [4.0, 5.0],
    ]
    assert (
        cuda_one.to_typetracer()
        ._mergemany(
            [
                cuda_two.to_typetracer(),
                cuda_three.to_typetracer(),
                cuda_four.to_typetracer(),
            ]
        )
        .form
        == cuda_one._mergemany([cuda_two, cuda_three, cuda_four]).form
    )
    assert (
        cuda_four.to_typetracer()
        ._mergemany(
            [
                cuda_three.to_typetracer(),
                cuda_four.to_typetracer(),
                cuda_one.to_typetracer(),
            ]
        )
        .form
        == cuda_four._mergemany([cuda_three, cuda_four, cuda_one]).form
    )

    one = ak.highlevel.Array([[1, 2, 3], [], [4, 5]]).layout
    two = ak.highlevel.Array([[1.1, 2.2], [3.3, 4.4]]).layout
    one = ak.contents.ListArray(one.starts, one.stops, one.content)
    two = ak.contents.ListArray(two.starts, two.stops, two.content)
    cuda_one = ak.to_backend(one, "cuda", highlevel=False)
    cuda_two = ak.to_backend(two, "cuda", highlevel=False)

    assert to_list(cuda_one._mergemany([cuda_two, cuda_three, cuda_four])) == [
        [1.0, 2.0, 3.0],
        [],
        [4.0, 5.0],
        [1.1, 2.2],
        [3.3, 4.4],
        [10.0],
        [20.0],
    ]
    assert to_list(cuda_four._mergemany([cuda_three, cuda_two, cuda_one])) == [
        [10.0],
        [20.0],
        [1.1, 2.2],
        [3.3, 4.4],
        [1.0, 2.0, 3.0],
        [],
        [4.0, 5.0],
    ]

    assert (
        cuda_one.to_typetracer()
        ._mergemany(
            [
                cuda_two.to_typetracer(),
                cuda_three.to_typetracer(),
                cuda_four.to_typetracer(),
            ]
        )
        .form
        == cuda_one._mergemany([cuda_two, cuda_three, cuda_four]).form
    )

    assert (
        cuda_four.to_typetracer()
        ._mergemany(
            [
                cuda_three.to_typetracer(),
                cuda_four.to_typetracer(),
                cuda_one.to_typetracer(),
            ]
        )
        .form
        == cuda_four._mergemany([cuda_three, cuda_four, cuda_one]).form
    )


def test_0449_merge_many_arrays_in_one_pass_records():
    one = ak.highlevel.Array(
        [{"x": 1, "y": [1]}, {"x": 2, "y": [1, 2]}, {"x": 3, "y": [1, 2, 3]}]
    ).layout
    two = ak.highlevel.Array([{"y": [], "x": 4}, {"y": [3, 2, 1], "x": 5}]).layout
    four = ak.highlevel.Array([{"x": 6, "y": [1]}, {"x": 7, "y": [1, 2]}]).layout

    cuda_one = ak.to_backend(one, "cuda", highlevel=False)
    cuda_two = ak.to_backend(two, "cuda", highlevel=False)
    cuda_three = cuda_two[0:0]
    cuda_four = ak.to_backend(four, "cuda", highlevel=False)

    assert to_list(cuda_one._mergemany([cuda_two, cuda_three, cuda_four])) == [
        {"x": 1, "y": [1]},
        {"x": 2, "y": [1, 2]},
        {"x": 3, "y": [1, 2, 3]},
        {"y": [], "x": 4},
        {"y": [3, 2, 1], "x": 5},
        {"x": 6, "y": [1]},
        {"x": 7, "y": [1, 2]},
    ]
    assert (
        cuda_one.to_typetracer()
        ._mergemany(
            [
                cuda_two.to_typetracer(),
                cuda_three.to_typetracer(),
                cuda_four.to_typetracer(),
            ]
        )
        .form
        == cuda_one._mergemany([cuda_two, cuda_three, cuda_four]).form
    )

    three = ak.contents.EmptyArray()
    cuda_three = ak.to_backend(three, "cuda", highlevel=False)

    assert to_list(cuda_one._mergemany([cuda_two, cuda_three, cuda_four])) == [
        {"x": 1, "y": [1]},
        {"x": 2, "y": [1, 2]},
        {"x": 3, "y": [1, 2, 3]},
        {"y": [], "x": 4},
        {"y": [3, 2, 1], "x": 5},
        {"x": 6, "y": [1]},
        {"x": 7, "y": [1, 2]},
    ]
    assert (
        cuda_one.to_typetracer()
        ._mergemany(
            [
                cuda_two.to_typetracer(),
                cuda_three.to_typetracer(),
                cuda_four.to_typetracer(),
            ]
        )
        .form
        == cuda_one._mergemany([cuda_two, cuda_three, cuda_four]).form
    )


def test_0449_merge_many_arrays_in_one_pass_tuples():
    one = ak.highlevel.Array([(1, [1]), (2, [1, 2]), (3, [1, 2, 3])]).layout
    two = ak.highlevel.Array([(4, []), (5, [3, 2, 1])]).layout
    four = ak.highlevel.Array([(6, [1]), (7, [1, 2])]).layout

    cuda_one = ak.to_backend(one, "cuda", highlevel=False)
    cuda_two = ak.to_backend(two, "cuda", highlevel=False)
    cuda_three = cuda_two[0:0]
    cuda_four = ak.to_backend(four, "cuda", highlevel=False)

    assert to_list(cuda_one._mergemany([cuda_two, cuda_three, cuda_four])) == [
        (1, [1]),
        (2, [1, 2]),
        (3, [1, 2, 3]),
        (4, []),
        (5, [3, 2, 1]),
        (6, [1]),
        (7, [1, 2]),
    ]
    assert (
        cuda_one.to_typetracer()
        ._mergemany(
            [
                cuda_two.to_typetracer(),
                cuda_three.to_typetracer(),
                cuda_four.to_typetracer(),
            ]
        )
        .form
        == cuda_one._mergemany([cuda_two, cuda_three, cuda_four]).form
    )

    three = ak.contents.EmptyArray()
    cuda_three = ak.to_backend(three, "cuda", highlevel=False)

    assert to_list(cuda_one._mergemany([cuda_two, cuda_three, cuda_four])) == [
        (1, [1]),
        (2, [1, 2]),
        (3, [1, 2, 3]),
        (4, []),
        (5, [3, 2, 1]),
        (6, [1]),
        (7, [1, 2]),
    ]
    assert (
        cuda_one.to_typetracer()
        ._mergemany(
            [
                cuda_two.to_typetracer(),
                cuda_three.to_typetracer(),
                cuda_four.to_typetracer(),
            ]
        )
        .form
        == cuda_one._mergemany([cuda_two, cuda_three, cuda_four]).form
    )


def test_0449_merge_many_arrays_in_one_pass_indexed():
    one = ak.highlevel.Array([1, 2, 3, None, 4, None, None, 5]).layout
    two = ak.highlevel.Array([6, 7, 8]).layout
    three = ak.contents.EmptyArray()
    four = ak.highlevel.Array([9, None, None]).layout

    cuda_one = ak.to_backend(one, "cuda", highlevel=False)
    cuda_two = ak.to_backend(two, "cuda", highlevel=False)
    cuda_three = ak.to_backend(three, "cuda", highlevel=False)
    cuda_four = ak.to_backend(four, "cuda", highlevel=False)

    assert to_list(cuda_one._mergemany([cuda_two, cuda_three, cuda_four])) == [
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
        cuda_one.to_typetracer()
        ._mergemany(
            [
                cuda_two.to_typetracer(),
                cuda_three.to_typetracer(),
                cuda_four.to_typetracer(),
            ]
        )
        .form
        == cuda_one._mergemany([cuda_two, cuda_three, cuda_four]).form
    )


def test_0449_merge_many_arrays_in_one_pass_reverse_indexed():
    one = ak.highlevel.Array([1, 2, 3]).layout
    two = ak.highlevel.Array([4, 5]).layout
    three = ak.highlevel.Array([None, 6, None]).layout

    cuda_one = ak.to_backend(one, "cuda", highlevel=False)
    cuda_two = ak.to_backend(two, "cuda", highlevel=False)
    cuda_three = ak.to_backend(three, "cuda", highlevel=False)

    assert to_list(cuda_one._mergemany([cuda_two, cuda_three])) == [
        1,
        2,
        3,
        4,
        5,
        None,
        6,
        None,
    ]
    assert (
        cuda_one.to_typetracer()
        ._mergemany([cuda_two.to_typetracer(), cuda_three.to_typetracer()])
        .form
        == cuda_one._mergemany([cuda_two, cuda_three]).form
    )

    four = ak.highlevel.Array([7, 8, None, None, 9]).layout
    cuda_four = ak.to_backend(four, "cuda", highlevel=False)

    assert to_list(cuda_one._mergemany([cuda_two, cuda_three, cuda_four])) == [
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
        cuda_one.to_typetracer()
        ._mergemany(
            [
                cuda_two.to_typetracer(),
                cuda_three.to_typetracer(),
                cuda_four.to_typetracer(),
            ]
        )
        .form
        == cuda_one._mergemany([cuda_two, cuda_three, cuda_four]).form
    )


def test_0449_merge_many_arrays_in_one_pass_bytemasked():
    one = ak.contents.ByteMaskedArray(
        ak.index.Index8([True, True, False, True, False, True]),
        ak.highlevel.Array([1, 2, 3, 4, 5, 6]).layout,
        valid_when=True,
    )
    two = ak.contents.ByteMaskedArray(
        ak.index.Index8([True, False, False, True, True]),
        ak.highlevel.Array([7, 99, 999, 8, 9]).layout,
        valid_when=True,
    )
    three = ak.highlevel.Array([100, 200, 300]).layout
    four = ak.highlevel.Array([None, None, 123, None]).layout

    cuda_one = ak.to_backend(one, "cuda", highlevel=False)
    cuda_two = ak.to_backend(two, "cuda", highlevel=False)
    cuda_three = ak.to_backend(three, "cuda", highlevel=False)
    cuda_four = ak.to_backend(four, "cuda", highlevel=False)

    assert to_list(cuda_one._mergemany([cuda_two, cuda_three, cuda_four])) == [
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
    assert to_list(cuda_four._mergemany([cuda_three, cuda_two, cuda_one])) == [
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
    assert to_list(cuda_three._mergemany([cuda_four, cuda_one])) == [
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
    assert to_list(cuda_three._mergemany([cuda_four, cuda_one, cuda_two])) == [
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
    assert to_list(cuda_three._mergemany([cuda_two, cuda_one])) == [
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
    assert to_list(cuda_three._mergemany([cuda_two, cuda_one, cuda_four])) == [
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
        cuda_one.to_typetracer()
        ._mergemany(
            [
                cuda_two.to_typetracer(),
                cuda_three.to_typetracer(),
                cuda_four.to_typetracer(),
            ]
        )
        .form
        == cuda_one._mergemany([cuda_two, cuda_three, cuda_four]).form
    )
    assert (
        cuda_four.to_typetracer()
        ._mergemany(
            [
                cuda_three.to_typetracer(),
                cuda_two.to_typetracer(),
                cuda_one.to_typetracer(),
            ]
        )
        .form
        == cuda_four._mergemany([cuda_three, cuda_two, cuda_one]).form
    )
    assert (
        cuda_three.to_typetracer()
        ._mergemany([cuda_four.to_typetracer(), cuda_one.to_typetracer()])
        .form
        == cuda_three._mergemany([cuda_four, cuda_one]).form
    )
    assert (
        cuda_three.to_typetracer()
        ._mergemany(
            [
                cuda_four.to_typetracer(),
                cuda_one.to_typetracer(),
                cuda_two.to_typetracer(),
            ]
        )
        .form
        == cuda_three._mergemany([cuda_four, cuda_one, cuda_two]).form
    )
    assert (
        cuda_three.to_typetracer()
        ._mergemany([cuda_two.to_typetracer(), cuda_one.to_typetracer()])
        .form
        == cuda_three._mergemany([cuda_two, cuda_one]).form
    )
    assert (
        cuda_three.to_typetracer()
        ._mergemany(
            [
                cuda_two.to_typetracer(),
                cuda_one.to_typetracer(),
                cuda_four.to_typetracer(),
            ]
        )
        .form
        == cuda_three._mergemany([cuda_two, cuda_one, cuda_four]).form
    )


def test_0449_merge_many_arrays_in_one_pass_union():
    one = ak.highlevel.Array([1, 2, [], [3, 4]]).layout
    two = ak.highlevel.Array([100, 200, 300]).layout
    three = ak.highlevel.Array([{"x": 1}, {"x": 2}, 5, 6, 7]).layout

    cuda_one = ak.to_backend(one, "cuda", highlevel=False)
    cuda_two = ak.to_backend(two, "cuda", highlevel=False)
    cuda_three = ak.to_backend(three, "cuda", highlevel=False)

    assert to_list(cuda_one._mergemany([cuda_two, cuda_three])) == [
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
    assert to_list(cuda_one._mergemany([cuda_three, cuda_two])) == [
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
    assert to_list(cuda_two._mergemany([cuda_one, cuda_three])) == [
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
    assert to_list(cuda_two._mergemany([cuda_three, cuda_one])) == [
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
    assert to_list(cuda_three._mergemany([cuda_one, cuda_two])) == [
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
    assert to_list(cuda_three._mergemany([cuda_two, cuda_one])) == [
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
        cuda_one.to_typetracer()
        ._mergemany([cuda_two.to_typetracer(), cuda_three.to_typetracer()])
        .form
        == cuda_one._mergemany([cuda_two, cuda_three]).form
    )
    assert (
        cuda_one.to_typetracer()
        ._mergemany([cuda_three.to_typetracer(), cuda_two.to_typetracer()])
        .form
        == cuda_one._mergemany([cuda_three, cuda_two]).form
    )
    assert (
        cuda_two.to_typetracer()
        ._mergemany([cuda_one.to_typetracer(), cuda_three.to_typetracer()])
        .form
        == cuda_two._mergemany([cuda_one, cuda_three]).form
    )
    assert (
        cuda_two.to_typetracer()
        ._mergemany([cuda_three.to_typetracer(), cuda_one.to_typetracer()])
        .form
        == cuda_two._mergemany([cuda_three, cuda_one]).form
    )
    assert (
        cuda_three.to_typetracer()
        ._mergemany([cuda_one.to_typetracer(), cuda_two.to_typetracer()])
        .form
        == cuda_three._mergemany([cuda_one, cuda_two]).form
    )
    assert (
        cuda_three.to_typetracer()
        ._mergemany([cuda_two.to_typetracer(), cuda_one.to_typetracer()])
        .form
        == cuda_three._mergemany([cuda_two, cuda_one]).form
    )


def test_0449_merge_many_arrays_in_one_pass_union_option():
    one = ak.highlevel.Array([1, 2, [], [3, 4]]).layout
    two = ak.highlevel.Array([100, None, 300]).layout
    three = ak.highlevel.Array([{"x": 1}, {"x": 2}, 5, 6, 7]).layout

    cuda_one = ak.to_backend(one, "cuda", highlevel=False)
    cuda_two = ak.to_backend(two, "cuda", highlevel=False)
    cuda_three = ak.to_backend(three, "cuda", highlevel=False)

    assert to_list(cuda_one._mergemany([cuda_two, cuda_three])) == [
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
    assert to_list(cuda_one._mergemany([cuda_three, cuda_two])) == [
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
    assert to_list(cuda_two._mergemany([cuda_one, cuda_three])) == [
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
    assert to_list(cuda_two._mergemany([cuda_three, cuda_one])) == [
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
    assert to_list(cuda_three._mergemany([cuda_one, cuda_two])) == [
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
    assert to_list(cuda_three._mergemany([cuda_two, cuda_one])) == [
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
        cuda_one.to_typetracer()
        ._mergemany([cuda_two.to_typetracer(), cuda_three.to_typetracer()])
        .form
        == cuda_one._mergemany([cuda_two, cuda_three]).form
    )
    assert (
        cuda_one.to_typetracer()
        ._mergemany([cuda_three.to_typetracer(), cuda_two.to_typetracer()])
        .form
        == cuda_one._mergemany([cuda_three, cuda_two]).form
    )
    assert (
        cuda_two.to_typetracer()
        ._mergemany([cuda_one.to_typetracer(), cuda_three.to_typetracer()])
        .form
        == cuda_two._mergemany([cuda_one, cuda_three]).form
    )
    assert (
        cuda_two.to_typetracer()
        ._mergemany([cuda_three.to_typetracer(), cuda_one.to_typetracer()])
        .form
        == cuda_two._mergemany([cuda_three, cuda_one]).form
    )
    assert (
        cuda_three.to_typetracer()
        ._mergemany([cuda_one.to_typetracer(), cuda_two.to_typetracer()])
        .form
        == cuda_three._mergemany([cuda_one, cuda_two]).form
    )
    assert (
        cuda_three.to_typetracer()
        ._mergemany([cuda_two.to_typetracer(), cuda_one.to_typetracer()])
        .form
        == cuda_three._mergemany([cuda_two, cuda_one]).form
    )

    one = ak.highlevel.Array([1, 2, [], [3, 4]]).layout
    two = ak.highlevel.Array([100, None, 300]).layout
    three = ak.highlevel.Array([{"x": 1}, {"x": 2}, 5, None, 7]).layout

    cuda_one = ak.to_backend(one, "cuda", highlevel=False)
    cuda_two = ak.to_backend(two, "cuda", highlevel=False)
    cuda_three = ak.to_backend(three, "cuda", highlevel=False)

    assert to_list(cuda_one._mergemany([cuda_two, cuda_three])) == [
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
    assert to_list(cuda_one._mergemany([cuda_three, cuda_two])) == [
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
    assert to_list(cuda_two._mergemany([cuda_one, cuda_three])) == [
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
    assert to_list(cuda_two._mergemany([cuda_three, cuda_one])) == [
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
    assert to_list(cuda_three._mergemany([cuda_one, cuda_two])) == [
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
    assert to_list(cuda_three._mergemany([cuda_two, cuda_one])) == [
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
        cuda_one.to_typetracer()
        ._mergemany([cuda_two.to_typetracer(), cuda_three.to_typetracer()])
        .form
        == cuda_one._mergemany([cuda_two, cuda_three]).form
    )
    assert (
        cuda_one.to_typetracer()
        ._mergemany([cuda_three.to_typetracer(), cuda_two.to_typetracer()])
        .form
        == cuda_one._mergemany([cuda_three, cuda_two]).form
    )
    assert (
        cuda_two.to_typetracer()
        ._mergemany([cuda_one.to_typetracer(), cuda_three.to_typetracer()])
        .form
        == cuda_two._mergemany([cuda_one, cuda_three]).form
    )
    assert (
        cuda_two.to_typetracer()
        ._mergemany([cuda_three.to_typetracer(), cuda_one.to_typetracer()])
        .form
        == cuda_two._mergemany([cuda_three, cuda_one]).form
    )
    assert (
        cuda_three.to_typetracer()
        ._mergemany([cuda_one.to_typetracer(), cuda_two.to_typetracer()])
        .form
        == cuda_three._mergemany([cuda_one, cuda_two]).form
    )
    assert (
        cuda_three.to_typetracer()
        ._mergemany([cuda_two.to_typetracer(), cuda_one.to_typetracer()])
        .form
        == cuda_three._mergemany([cuda_two, cuda_one]).form
    )

    one = ak.highlevel.Array([1, 2, [], [3, 4]]).layout
    two = ak.highlevel.Array([100, 200, 300]).layout
    three = ak.highlevel.Array([{"x": 1}, {"x": 2}, 5, None, 7]).layout

    cuda_one = ak.to_backend(one, "cuda", highlevel=False)
    cuda_two = ak.to_backend(two, "cuda", highlevel=False)
    cuda_three = ak.to_backend(three, "cuda", highlevel=False)

    assert to_list(cuda_one._mergemany([cuda_two, cuda_three])) == [
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
    assert to_list(cuda_one._mergemany([cuda_three, cuda_two])) == [
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
    assert to_list(cuda_two._mergemany([cuda_one, cuda_three])) == [
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
    assert to_list(cuda_two._mergemany([cuda_three, cuda_one])) == [
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
    assert to_list(cuda_three._mergemany([cuda_one, cuda_two])) == [
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
    assert to_list(cuda_three._mergemany([cuda_two, cuda_one])) == [
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
        cuda_one.to_typetracer()
        ._mergemany([cuda_two.to_typetracer(), cuda_three.to_typetracer()])
        .form
        == cuda_one._mergemany([cuda_two, cuda_three]).form
    )
    assert (
        cuda_one.to_typetracer()
        ._mergemany([cuda_three.to_typetracer(), cuda_two.to_typetracer()])
        .form
        == cuda_one._mergemany([cuda_three, cuda_two]).form
    )
    assert (
        cuda_two.to_typetracer()
        ._mergemany([cuda_one.to_typetracer(), cuda_three.to_typetracer()])
        .form
        == cuda_two._mergemany([cuda_one, cuda_three]).form
    )
    assert (
        cuda_two.to_typetracer()
        ._mergemany([cuda_three.to_typetracer(), cuda_one.to_typetracer()])
        .form
        == cuda_two._mergemany([cuda_three, cuda_one]).form
    )
    assert (
        cuda_three.to_typetracer()
        ._mergemany([cuda_one.to_typetracer(), cuda_two.to_typetracer()])
        .form
        == cuda_three._mergemany([cuda_one, cuda_two]).form
    )
    assert (
        cuda_three.to_typetracer()
        ._mergemany([cuda_two.to_typetracer(), cuda_one.to_typetracer()])
        .form
        == cuda_three._mergemany([cuda_two, cuda_one]).form
    )


def test_0449_merge_many_arrays_in_one_pass_strings():
    one = ak.highlevel.Array(["uno", "dos", "tres"]).layout
    two = ak.highlevel.Array(["un", "deux", "trois", "quatre"]).layout
    three = ak.highlevel.Array(["onay", "ootay", "eethray"]).layout

    cuda_one = ak.to_backend(one, "cuda", highlevel=False)
    cuda_two = ak.to_backend(two, "cuda", highlevel=False)
    cuda_three = ak.to_backend(three, "cuda", highlevel=False)

    assert to_list(cuda_one._mergemany([cuda_two, cuda_three])) == [
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
        cuda_one.to_typetracer()
        ._mergemany([cuda_two.to_typetracer(), cuda_three.to_typetracer()])
        .form
        == cuda_one._mergemany([cuda_two, cuda_three]).form
    )


layouts = [
    # ListArray
    ak.contents.ListArray(
        ak.index.Index64([0, 3]),
        ak.index.Index64([3, 6]),
        ak.contents.NumpyArray(np.arange(6, dtype=np.int64)),
    ),
    # ListArray
    ak.contents.ListOffsetArray(
        ak.index.Index64([0, 3, 6]),
        ak.contents.NumpyArray(np.arange(6, dtype=np.int64)),
    ),
    # RegularArray
    ak.contents.RegularArray(ak.contents.NumpyArray(np.arange(6, dtype=np.int64)), 3),
    ak.contents.RegularArray(ak.contents.NumpyArray(np.arange(6, dtype=np.int64)), 2),
    # ByteMaskedArray
    ak.contents.ByteMaskedArray(
        ak.index.Index8([True, False, False, True]),
        ak.contents.NumpyArray(np.arange(6, dtype=np.int32)),
        valid_when=True,
    ),
    # ByteMaskedArray
    ak.contents.BitMaskedArray(
        ak.index.IndexU8([1 << 0 | 1 << 1 | 0 << 2 | 0 << 3 | 1 << 4 | 0 << 5]),
        ak.contents.NumpyArray(np.arange(6, dtype=np.int32)),
        valid_when=True,
        lsb_order=True,
        length=6,
    ),
    # UnmaskedArray
    ak.contents.UnmaskedArray(ak.contents.NumpyArray(np.arange(6, dtype=np.int32))),
    # IndexedOptionArray
    ak.contents.IndexedOptionArray(
        ak.index.Index64([3, 1, -1, -1, 2, 0, -1]),
        ak.contents.NumpyArray(np.arange(6, dtype=np.int32)),
    ),
    # NumpyArray
    ak.contents.NumpyArray(np.arange(6, dtype=np.int16)),
    ak.contents.NumpyArray(np.arange(6 * 4, dtype=np.float32).reshape(6, 4)),
    # IndexedArray
    ak.contents.IndexedArray(
        ak.index.Index64([3, 1, 1, 0, 2, 0, 0]),
        ak.contents.NumpyArray(np.arange(6, dtype=np.int32)),
    ),
    # RecordArray
    ak.contents.RecordArray(
        [ak.contents.NumpyArray(np.arange(6, dtype=np.int16))], ["x"]
    ),
    ak.contents.RecordArray(
        [ak.contents.NumpyArray(np.arange(6, dtype=np.float64))], ["y"]
    ),
    ak.contents.RecordArray(
        [ak.contents.NumpyArray(np.arange(6, dtype=np.float32))], None
    ),
    # UnionArray
    ak.contents.UnionArray(
        ak.index.Index8([0, 0, 1]),
        ak.index.Index64([0, 1, 0]),
        [
            ak.contents.NumpyArray(np.arange(6, dtype=np.int16)),
            ak.contents.RecordArray(
                [ak.contents.NumpyArray(np.arange(6, dtype=np.float32))], None
            ),
        ],
    ),
]
cuda_layouts = ak.to_backend(layouts, "cuda", highlevel=False)


@pytest.mark.parametrize("left", cuda_layouts)
@pytest.mark.parametrize("right", cuda_layouts)
def test_2860_enforce_concatenated_form_symmetric(left, right):
    result = ak.concatenate([left, right], axis=0, highlevel=False)
    part_0_result = enforce_concatenated_form(left, result.form)
    assert part_0_result.form == result.form

    part_1_result = enforce_concatenated_form(right, result.form)
    assert part_1_result.form == result.form

    assert part_0_result.to_list() == result[: part_0_result.length].to_list()
    assert part_1_result.to_list() == result[part_0_result.length :].to_list()
    assert (
        ak.backend(result)
        == ak.backend(part_0_result)
        == ak.backend(part_1_result)
        == "cuda"
    )
