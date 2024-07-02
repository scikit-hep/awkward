from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

import awkward as ak
from awkward.types import ArrayType, ListType, NumpyType, OptionType, RegularType

to_list = ak.operations.to_list


@pytest.fixture(scope="function", autouse=True)
def cleanup_cuda():
    yield
    cp._default_memory_pool.free_all_blocks()
    cp.cuda.Device().synchronize()


def test_0184_concatenate_number():
    a1 = ak.highlevel.Array([[1, 2, 3], [], [4, 5]]).layout
    a2 = ak.highlevel.Array([[[1.1], [2.2, 3.3]], [[]], [[4.4], [5.5]]]).layout
    a3 = ak.highlevel.Array([[123], [223], [323]]).layout

    cuda_a1 = ak.to_backend(a1, "cuda")
    cuda_a2 = ak.to_backend(a2, "cuda")
    cuda_a3 = ak.to_backend(a3, "cuda")

    assert to_list(ak.operations.concatenate([cuda_a1, 999], axis=1)) == [
        [1, 2, 3, 999],
        [999],
        [4, 5, 999],
    ]

    assert to_list(ak.operations.concatenate([cuda_a2, 999], axis=2)) == [
        [[1.1, 999.0], [2.2, 3.3, 999.0]],
        [[999.0]],
        [[4.4, 999.0], [5.5, 999.0]],
    ]

    assert (
        str(ak.operations.type(ak.operations.concatenate([cuda_a1, cuda_a3], axis=1)))
        == "3 * var * int64"
    )

    assert to_list(ak.operations.concatenate([cuda_a1, cuda_a3], axis=1)) == [
        [1, 2, 3, 123],
        [223],
        [4, 5, 323],
    ]


def test_0184_negative_axis_concatenate():
    one = ak.highlevel.Array(
        [[[0.0, 1.1, 2.2], []], [[3.3, 4.4]], [[5.5], [6.6, 7.7, 8.8, 9.9]]]
    ).layout
    two = ak.highlevel.Array(
        [[[10, 20], [30]], [[40]], [[50, 60, 70], [80, 90]]]
    ).layout
    arrays = [one, two]

    cuda_arrays = ak.to_backend(arrays, "cuda")

    assert ak.operations.concatenate(cuda_arrays, axis=-1).to_list() == [
        [[0.0, 1.1, 2.2, 10, 20], [30]],
        [[3.3, 4.4, 40]],
        [[5.5, 50, 60, 70], [6.6, 7.7, 8.8, 9.9, 80, 90]],
    ]

    assert ak.operations.concatenate(cuda_arrays, axis=-2).to_list() == [
        [[0.0, 1.1, 2.2], [], [10, 20], [30]],
        [[3.3, 4.4], [40]],
        [[5.5], [6.6, 7.7, 8.8, 9.9], [50, 60, 70], [80, 90]],
    ]

    assert ak.operations.concatenate(cuda_arrays, axis=-3).to_list() == [
        [[0.0, 1.1, 2.2], []],
        [[3.3, 4.4]],
        [[5.5], [6.6, 7.7, 8.8, 9.9]],
        [[10, 20], [30]],
        [[40]],
        [[50, 60, 70], [80, 90]],
    ]


def test_0184_broadcast_and_apply_levels_concatenate():
    arrays = [
        ak.highlevel.Array(
            [[[0.0, 1.1, 2.2], []], [[3.3, 4.4]], [[5.5], [6.6, 7.7, 8.8, 9.9]]]
        ).layout,
        ak.highlevel.Array([[[10, 20], [30]], [[40]], [[50, 60, 70], [80, 90]]]).layout,
    ]

    cuda_arrays = ak.to_backend(arrays, "cuda")

    # nothing is required to have the same length
    assert ak.operations.concatenate(cuda_arrays, axis=0).to_list() == [
        [[0.0, 1.1, 2.2], []],
        [[3.3, 4.4]],
        [[5.5], [6.6, 7.7, 8.8, 9.9]],
        [[10, 20], [30]],
        [[40]],
        [[50, 60, 70], [80, 90]],
    ]
    # the outermost arrays are required to have the same length, but nothing deeper than that
    assert ak.operations.concatenate(cuda_arrays, axis=1).to_list() == [
        [[0.0, 1.1, 2.2], [], [10, 20], [30]],
        [[3.3, 4.4], [40]],
        [[5.5], [6.6, 7.7, 8.8, 9.9], [50, 60, 70], [80, 90]],
    ]
    # the outermost arrays and the first level are required to have the same length, but nothing deeper
    assert ak.operations.concatenate(cuda_arrays, axis=2).to_list() == [
        [[0.0, 1.1, 2.2, 10, 20], [30]],
        [[3.3, 4.4, 40]],
        [[5.5, 50, 60, 70], [6.6, 7.7, 8.8, 9.9, 80, 90]],
    ]


def test_0184_list_array_concatenate():
    one = ak.highlevel.Array([[1, 2, 3], [], [4, 5]]).layout
    two = ak.highlevel.Array([[1.1, 2.2], [3.3, 4.4], [5.5]]).layout

    one = ak.contents.ListArray(one.starts, one.stops, one.content)
    two = ak.contents.ListArray(two.starts, two.stops, two.content)

    cuda_one = ak.to_backend(one, "cuda")
    cuda_two = ak.to_backend(two, "cuda")

    assert to_list(ak.operations.concatenate([cuda_one, cuda_two], 0)) == [
        [1, 2, 3],
        [],
        [4, 5],
        [1.1, 2.2],
        [3.3, 4.4],
        [5.5],
    ]
    assert to_list(ak.operations.concatenate([cuda_one, cuda_two], 1)) == [
        [1, 2, 3, 1.1, 2.2],
        [3.3, 4.4],
        [4, 5, 5.5],
    ]


def test_0184_indexed_array_concatenate():
    one = ak.highlevel.Array([[1, 2, 3], [None, 4], None, [None, 5]]).layout
    two = ak.highlevel.Array([6, 7, 8]).layout
    three = ak.highlevel.Array([[6.6], [7.7, 8.8]]).layout
    four = ak.highlevel.Array([[6.6], [7.7, 8.8], None, [9.9]]).layout

    cuda_one = ak.to_backend(one, "cuda")
    cuda_two = ak.to_backend(two, "cuda")
    cuda_three = ak.to_backend(three, "cuda")
    cuda_four = ak.to_backend(four, "cuda")

    assert to_list(ak.operations.concatenate([cuda_one, cuda_two], 0)) == [
        [1, 2, 3],
        [None, 4],
        None,
        [None, 5],
        6,
        7,
        8,
    ]

    with pytest.raises(ValueError):
        to_list(ak.operations.concatenate([cuda_one, cuda_three], 1))

    assert to_list(ak.operations.concatenate([cuda_one, cuda_four], 1)) == [
        [1, 2, 3, 6.6],
        [None, 4, 7.7, 8.8],
        [],
        [None, 5, 9.9],
    ]


def test_0184_listoffsetarray_concatenate():
    content_one = ak.contents.NumpyArray(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    offsets_one = ak.index.Index64(np.array([0, 3, 3, 5, 9]))
    one = ak.contents.ListOffsetArray(offsets_one, content_one)

    cuda_one = ak.to_backend(one, "cuda")

    assert to_list(cuda_one) == [[1, 2, 3], [], [4, 5], [6, 7, 8, 9]]

    content_two = ak.contents.NumpyArray(np.array([100, 200, 300, 400, 500]))
    offsets_two = ak.index.Index64(np.array([0, 2, 4, 4, 5]))
    two = ak.contents.ListOffsetArray(offsets_two, content_two)

    cuda_two = ak.to_backend(two, "cuda")

    assert to_list(cuda_two) == [[100, 200], [300, 400], [], [500]]
    assert to_list(ak.operations.concatenate([cuda_one, cuda_two], 0)) == [
        [1, 2, 3],
        [],
        [4, 5],
        [6, 7, 8, 9],
        [100, 200],
        [300, 400],
        [],
        [500],
    ]
    assert to_list(ak.operations.concatenate([cuda_one, cuda_two], 1)) == [
        [1, 2, 3, 100, 200],
        [300, 400],
        [4, 5],
        [6, 7, 8, 9, 500],
    ]


def test_0184_even_more():
    dim1 = ak.highlevel.Array([1.1, 2.2, 3.3, 4.4, 5.5]).layout
    dim1a = ak.highlevel.Array([[1.1], [2.2], [3.3], [4.4], [5.5]]).layout
    dim1b = ak.highlevel.Array(np.array([[1.1], [2.2], [3.3], [4.4], [5.5]])).layout
    dim2 = ak.highlevel.Array([[0, 1, 2], [], [3, 4], [5], [6, 7, 8, 9]]).layout
    dim3 = ak.highlevel.Array(
        [[[0, 1, 2], []], [[3, 4]], [], [[5], [6, 7, 8, 9]], []]
    ).layout

    cuda_dim1 = ak.to_backend(dim1, "cuda")
    cuda_dim1a = ak.to_backend(dim1a, "cuda")
    cuda_dim1b = ak.to_backend(dim1b, "cuda")
    cuda_dim2 = ak.to_backend(dim2, "cuda")
    cuda_dim3 = ak.to_backend(dim3, "cuda")

    num = cp.array([999])

    assert ak.operations.concatenate([cuda_dim1, num]).to_list() == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
        999,
    ]
    assert ak.operations.concatenate([num, cuda_dim1]).to_list() == [
        999,
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]
    assert ak.operations.concatenate([cuda_dim1, cuda_dim2]).to_list() == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
        [0, 1, 2],
        [],
        [3, 4],
        [5],
        [6, 7, 8, 9],
    ]
    assert ak.operations.concatenate([cuda_dim2, cuda_dim1]).to_list() == [
        [0, 1, 2],
        [],
        [3, 4],
        [5],
        [6, 7, 8, 9],
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]

    with pytest.raises(ValueError):
        ak.operations.concatenate([cuda_dim1, 999], axis=1)

    assert ak.operations.concatenate([cuda_dim2, 999], axis=1).to_list() == [
        [0, 1, 2, 999],
        [999],
        [3, 4, 999],
        [5, 999],
        [6, 7, 8, 9, 999],
    ]

    assert ak.operations.concatenate([999, cuda_dim2], axis=1).to_list() == [
        [999, 0, 1, 2],
        [999],
        [999, 3, 4],
        [999, 5],
        [999, 6, 7, 8, 9],
    ]

    with pytest.raises(ValueError):
        ak.operations.concatenate([cuda_dim1, cuda_dim2], axis=1)
    with pytest.raises(ValueError):
        ak.operations.concatenate([cuda_dim2, cuda_dim1], axis=1)
    assert ak.operations.concatenate([cuda_dim1a, cuda_dim2], axis=1).to_list() == [
        [1.1, 0, 1, 2],
        [2.2],
        [3.3, 3, 4],
        [4.4, 5],
        [5.5, 6, 7, 8, 9],
    ]
    assert ak.operations.concatenate([cuda_dim2, cuda_dim1a], axis=1).to_list() == [
        [0, 1, 2, 1.1],
        [2.2],
        [3, 4, 3.3],
        [5, 4.4],
        [6, 7, 8, 9, 5.5],
    ]
    assert ak.operations.concatenate([cuda_dim1b, cuda_dim2], axis=1).to_list() == [
        [1.1, 0, 1, 2],
        [2.2],
        [3.3, 3, 4],
        [4.4, 5],
        [5.5, 6, 7, 8, 9],
    ]
    assert ak.operations.concatenate([cuda_dim2, cuda_dim1b], axis=1).to_list() == [
        [0, 1, 2, 1.1],
        [2.2],
        [3, 4, 3.3],
        [5, 4.4],
        [6, 7, 8, 9, 5.5],
    ]
    num = cp.array([123])

    assert ak.operations.concatenate([num, cuda_dim1, cuda_dim2]).to_list() == [
        123,
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
        [0, 1, 2],
        [],
        [3, 4],
        [5],
        [6, 7, 8, 9],
    ]
    assert ak.operations.concatenate([num, cuda_dim2, cuda_dim1]).to_list() == [
        123,
        [0, 1, 2],
        [],
        [3, 4],
        [5],
        [6, 7, 8, 9],
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]
    assert ak.operations.concatenate([cuda_dim1, num, cuda_dim2]).to_list() == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
        123,
        [0, 1, 2],
        [],
        [3, 4],
        [5],
        [6, 7, 8, 9],
    ]
    assert ak.operations.concatenate([cuda_dim1, cuda_dim2, num]).to_list() == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
        [0, 1, 2],
        [],
        [3, 4],
        [5],
        [6, 7, 8, 9],
        123,
    ]
    assert ak.operations.concatenate([cuda_dim2, num, cuda_dim1]).to_list() == [
        [0, 1, 2],
        [],
        [3, 4],
        [5],
        [6, 7, 8, 9],
        123,
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]
    assert ak.operations.concatenate([cuda_dim2, cuda_dim1, num]).to_list() == [
        [0, 1, 2],
        [],
        [3, 4],
        [5],
        [6, 7, 8, 9],
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
        123,
    ]

    with pytest.raises(ValueError):
        ak.operations.concatenate([123, cuda_dim1, cuda_dim2], axis=1)
    with pytest.raises(ValueError):
        ak.operations.concatenate([123, cuda_dim2, cuda_dim1], axis=1)
    with pytest.raises(ValueError):
        ak.operations.concatenate([cuda_dim1, 123, cuda_dim2], axis=1)
    with pytest.raises(ValueError):
        ak.operations.concatenate([cuda_dim1, cuda_dim2, 123], axis=1)
    with pytest.raises(ValueError):
        ak.operations.concatenate([cuda_dim2, 123, cuda_dim1], axis=1)
    with pytest.raises(ValueError):
        ak.operations.concatenate([cuda_dim2, cuda_dim1, 123], axis=1)

    assert ak.operations.concatenate(
        [123, cuda_dim1a, cuda_dim2], axis=1
    ).to_list() == [
        [123, 1.1, 0, 1, 2],
        [123, 2.2],
        [123, 3.3, 3, 4],
        [123, 4.4, 5],
        [123, 5.5, 6, 7, 8, 9],
    ]
    assert ak.operations.concatenate(
        [123, cuda_dim2, cuda_dim1a], axis=1
    ).to_list() == [
        [123, 0, 1, 2, 1.1],
        [123, 2.2],
        [123, 3, 4, 3.3],
        [123, 5, 4.4],
        [123, 6, 7, 8, 9, 5.5],
    ]
    assert ak.operations.concatenate(
        [cuda_dim1a, 123, cuda_dim2], axis=1
    ).to_list() == [
        [1.1, 123, 0, 1, 2],
        [2.2, 123],
        [3.3, 123, 3, 4],
        [4.4, 123, 5],
        [5.5, 123, 6, 7, 8, 9],
    ]
    assert ak.operations.concatenate(
        [cuda_dim1a, cuda_dim2, 123], axis=1
    ).to_list() == [
        [1.1, 0, 1, 2, 123],
        [2.2, 123],
        [3.3, 3, 4, 123],
        [4.4, 5, 123],
        [5.5, 6, 7, 8, 9, 123],
    ]
    assert ak.operations.concatenate(
        [cuda_dim2, 123, cuda_dim1a], axis=1
    ).to_list() == [
        [0, 1, 2, 123, 1.1],
        [123, 2.2],
        [3, 4, 123, 3.3],
        [5, 123, 4.4],
        [6, 7, 8, 9, 123, 5.5],
    ]
    assert ak.operations.concatenate(
        [cuda_dim2, cuda_dim1a, 123], axis=1
    ).to_list() == [
        [0, 1, 2, 1.1, 123],
        [2.2, 123],
        [3, 4, 3.3, 123],
        [5, 4.4, 123],
        [6, 7, 8, 9, 5.5, 123],
    ]

    assert ak.operations.concatenate(
        [123, cuda_dim1b, cuda_dim2], axis=1
    ).to_list() == [
        [123, 1.1, 0, 1, 2],
        [123, 2.2],
        [123, 3.3, 3, 4],
        [123, 4.4, 5],
        [123, 5.5, 6, 7, 8, 9],
    ]
    assert ak.operations.concatenate(
        [123, cuda_dim2, cuda_dim1b], axis=1
    ).to_list() == [
        [123, 0, 1, 2, 1.1],
        [123, 2.2],
        [123, 3, 4, 3.3],
        [123, 5, 4.4],
        [123, 6, 7, 8, 9, 5.5],
    ]
    assert ak.operations.concatenate(
        [cuda_dim1b, 123, cuda_dim2], axis=1
    ).to_list() == [
        [1.1, 123, 0, 1, 2],
        [2.2, 123],
        [3.3, 123, 3, 4],
        [4.4, 123, 5],
        [5.5, 123, 6, 7, 8, 9],
    ]
    assert ak.operations.concatenate(
        [cuda_dim1b, cuda_dim2, 123], axis=1
    ).to_list() == [
        [1.1, 0, 1, 2, 123],
        [2.2, 123],
        [3.3, 3, 4, 123],
        [4.4, 5, 123],
        [5.5, 6, 7, 8, 9, 123],
    ]
    assert ak.operations.concatenate(
        [cuda_dim2, 123, cuda_dim1b], axis=1
    ).to_list() == [
        [0, 1, 2, 123, 1.1],
        [123, 2.2],
        [3, 4, 123, 3.3],
        [5, 123, 4.4],
        [6, 7, 8, 9, 123, 5.5],
    ]
    assert ak.operations.concatenate(
        [cuda_dim2, cuda_dim1b, 123], axis=1
    ).to_list() == [
        [0, 1, 2, 1.1, 123],
        [2.2, 123],
        [3, 4, 3.3, 123],
        [5, 4.4, 123],
        [6, 7, 8, 9, 5.5, 123],
    ]

    # FIX ME
    # assert ak.operations.concatenate([cuda_dim3, 123]).to_list() == [
    #     [[0, 1, 2], []],
    #     [[3, 4]],
    #     [],
    #     [[5], [6, 7, 8, 9]],
    #     [],
    #     123,
    # ]

    assert ak.operations.concatenate([cuda_dim3, num]).to_list() == [
        [[0, 1, 2], []],
        [[3, 4]],
        [],
        [[5], [6, 7, 8, 9]],
        [],
        123,
    ]
    assert ak.operations.concatenate([num, cuda_dim3]).to_list() == [
        123,
        [[0, 1, 2], []],
        [[3, 4]],
        [],
        [[5], [6, 7, 8, 9]],
        [],
    ]

    assert ak.operations.concatenate([cuda_dim3, 123], axis=1).to_list() == [
        [[0, 1, 2], [], 123],
        [[3, 4], 123],
        [123],
        [[5], [6, 7, 8, 9], 123],
        [123],
    ]
    assert ak.operations.concatenate([123, cuda_dim3], axis=1).to_list() == [
        [123, [0, 1, 2], []],
        [123, [3, 4]],
        [123],
        [123, [5], [6, 7, 8, 9]],
        [123],
    ]

    with pytest.raises(ValueError):
        ak.operations.concatenate([cuda_dim3, cuda_dim1], axis=1)
    with pytest.raises(ValueError):
        ak.operations.concatenate([cuda_dim1, cuda_dim3], axis=1)

    assert ak.operations.concatenate([cuda_dim3, cuda_dim2], axis=1).to_list() == [
        [[0, 1, 2], [], 0, 1, 2],
        [[3, 4]],
        [3, 4],
        [[5], [6, 7, 8, 9], 5],
        [6, 7, 8, 9],
    ]
    assert ak.operations.concatenate([cuda_dim2, cuda_dim3], axis=1).to_list() == [
        [0, 1, 2, [0, 1, 2], []],
        [[3, 4]],
        [3, 4],
        [5, [5], [6, 7, 8, 9]],
        [6, 7, 8, 9],
    ]

    assert ak.operations.concatenate([cuda_dim3, 123], axis=2).to_list() == [
        [[0, 1, 2, 123], [123]],
        [[3, 4, 123]],
        [],
        [[5, 123], [6, 7, 8, 9, 123]],
        [],
    ]
    assert ak.operations.concatenate([123, cuda_dim3], axis=2).to_list() == [
        [[123, 0, 1, 2], [123]],
        [[123, 3, 4]],
        [],
        [[123, 5], [123, 6, 7, 8, 9]],
        [],
    ]

    assert ak.operations.concatenate([cuda_dim3, cuda_dim3], axis=2).to_list() == [
        [[0, 1, 2, 0, 1, 2], []],
        [[3, 4, 3, 4]],
        [],
        [[5, 5], [6, 7, 8, 9, 6, 7, 8, 9]],
        [],
    ]

    rec1 = ak.highlevel.Array(
        [
            {"x": [1, 2], "y": [1.1]},
            {"x": [], "y": [2.2, 3.3]},
            {"x": [3], "y": []},
            {"x": [5, 6, 7], "y": []},
            {"x": [8, 9], "y": [4.4, 5.5]},
        ]
    ).layout
    rec2 = ak.highlevel.Array(
        [
            {"x": [100], "y": [10, 20]},
            {"x": [200], "y": []},
            {"x": [300, 400], "y": [30]},
            {"x": [], "y": [40, 50]},
            {"x": [400, 500], "y": [60]},
        ]
    ).layout

    assert ak.operations.concatenate([rec1, rec2]).to_list() == [
        {"x": [1, 2], "y": [1.1]},
        {"x": [], "y": [2.2, 3.3]},
        {"x": [3], "y": []},
        {"x": [5, 6, 7], "y": []},
        {"x": [8, 9], "y": [4.4, 5.5]},
        {"x": [100], "y": [10, 20]},
        {"x": [200], "y": []},
        {"x": [300, 400], "y": [30]},
        {"x": [], "y": [40, 50]},
        {"x": [400, 500], "y": [60]},
    ]

    assert ak.operations.concatenate([rec1, rec2], axis=1).to_list() == [
        {"x": [1, 2, 100], "y": [1.1, 10, 20]},
        {"x": [200], "y": [2.2, 3.3]},
        {"x": [3, 300, 400], "y": [30]},
        {"x": [5, 6, 7], "y": [40, 50]},
        {"x": [8, 9, 400, 500], "y": [4.4, 5.5, 60]},
    ]


def test_1586_numpy_regular():
    a1 = ak.Array(np.array([[0.0, 1.1], [2.2, 3.3]]))
    a2 = ak.from_json("[[4.4, 5.5], [6.6, 7.7], [8.8, 9.9]]")
    cuda_a1 = ak.to_backend(a1, "cuda")
    cuda_a2 = ak.to_backend(a2, "cuda")

    assert isinstance(cuda_a1.layout, ak.contents.NumpyArray)

    cuda_a2 = ak.to_regular(cuda_a2, axis=1)

    cuda_c = ak.concatenate([cuda_a1, cuda_a2])

    assert cuda_c.to_list() == [
        [0.0, 1.1],
        [2.2, 3.3],
        [4.4, 5.5],
        [6.6, 7.7],
        [8.8, 9.9],
    ]
    assert cuda_c.type == ArrayType(RegularType(NumpyType("float64"), 2), 5)


def test_1586_regular_option():
    a1 = ak.from_json("[[0.0, 1.1], [2.2, 3.3]]")
    a2 = ak.from_json("[[4.4, 5.5], [6.6, 7.7], null, [8.8, 9.9]]")
    cuda_a1 = ak.to_backend(a1, "cuda")
    cuda_a2 = ak.to_backend(a2, "cuda")

    cuda_a1 = ak.to_regular(cuda_a1, axis=1)
    cuda_a2 = ak.to_regular(cuda_a2, axis=1)
    cuda_c = ak.concatenate([cuda_a1, cuda_a2])

    assert cuda_c.to_list() == [
        [0.0, 1.1],
        [2.2, 3.3],
        [4.4, 5.5],
        [6.6, 7.7],
        None,
        [8.8, 9.9],
    ]
    assert cuda_c.type == ArrayType(OptionType(RegularType(NumpyType("float64"), 2)), 6)


def test_1586_regular_regular():
    a1 = ak.from_json("[[0.0, 1.1], [2.2, 3.3]]")
    a2 = ak.from_json("[[4.4, 5.5], [6.6, 7.7], [8.8, 9.9]]")
    cuda_a1 = ak.to_backend(a1, "cuda")
    cuda_a2 = ak.to_backend(a2, "cuda")

    cuda_a1 = ak.to_regular(cuda_a1, axis=1)
    cuda_a2 = ak.to_regular(cuda_a2, axis=1)

    cuda_c = ak.concatenate([cuda_a1, cuda_a2])

    assert cuda_c.to_list() == [
        [0.0, 1.1],
        [2.2, 3.3],
        [4.4, 5.5],
        [6.6, 7.7],
        [8.8, 9.9],
    ]
    assert cuda_c.type == ArrayType(RegularType(NumpyType("float64"), 2), 5)


def test_1586_option_option():
    a1 = ak.from_json("[[0.0, 1.1], null, [2.2, 3.3]]")
    a2 = ak.from_json("[[4.4, 5.5], [6.6, 7.7], null, [8.8, 9.9]]")
    cuda_a1 = ak.to_backend(a1, "cuda")
    cuda_a2 = ak.to_backend(a2, "cuda")

    cuda_a1 = ak.to_regular(cuda_a1, axis=1)
    cuda_a2 = ak.to_regular(cuda_a2, axis=1)
    cuda_c = ak.concatenate([cuda_a1, cuda_a2])

    assert cuda_c.to_list() == [
        [0.0, 1.1],
        None,
        [2.2, 3.3],
        [4.4, 5.5],
        [6.6, 7.7],
        None,
        [8.8, 9.9],
    ]
    assert cuda_c.type == ArrayType(OptionType(RegularType(NumpyType("float64"), 2)), 7)


def test_1586_option_option_axis1():
    a1 = ak.from_json("[[0.0, 1.1], null, [2.2, 3.3]]")
    a2 = ak.from_json("[[4.4, 5.5, 6.6], null, [7.7, 8.8, 9.9]]")
    cuda_a1 = ak.to_backend(a1, "cuda")
    cuda_a2 = ak.to_backend(a2, "cuda")

    cuda_a1 = ak.to_regular(cuda_a1, axis=1)
    cuda_a2 = ak.to_regular(cuda_a2, axis=1)
    cuda_c = ak.concatenate([cuda_a1, cuda_a2], axis=1)
    assert cuda_c.to_list() == [
        [0.0, 1.1, 4.4, 5.5, 6.6],
        [],
        [2.2, 3.3, 7.7, 8.8, 9.9],
    ]
    assert cuda_c.type == ArrayType(ListType(NumpyType("float64")), 3)


def test_1586_regular_option_axis1():
    a1 = ak.from_json("[[0.0, 1.1], [7, 8], [2.2, 3.3]]")
    a2 = ak.from_json("[[4.4, 5.5, 6.6], null, [7.7, 8.8, 9.9]]")
    cuda_a1 = ak.to_backend(a1, "cuda")
    cuda_a2 = ak.to_backend(a2, "cuda")

    cuda_a1 = ak.to_regular(cuda_a1, axis=1)
    cuda_a2 = ak.to_regular(cuda_a2, axis=1)
    cuda_c = ak.concatenate([cuda_a1, cuda_a2], axis=1)
    assert cuda_c.to_list() == [
        [0.0, 1.1, 4.4, 5.5, 6.6],
        [7, 8],
        [2.2, 3.3, 7.7, 8.8, 9.9],
    ]
    assert cuda_c.type == ArrayType(ListType(NumpyType("float64")), 3)


def test_1586_option_regular():
    a1 = ak.from_json("[[0.0, 1.1], null, [2.2, 3.3]]")
    a2 = ak.from_json("[[4.4, 5.5], [6.6, 7.7], [8.8, 9.9]]")
    cuda_a1 = ak.to_backend(a1, "cuda")
    cuda_a2 = ak.to_backend(a2, "cuda")

    cuda_a1 = ak.to_regular(cuda_a1, axis=1)
    cuda_a2 = ak.to_regular(cuda_a2, axis=1)
    cuda_c = ak.concatenate([cuda_a1, cuda_a2])
    assert cuda_c.to_list() == [
        [0.0, 1.1],
        None,
        [2.2, 3.3],
        [4.4, 5.5],
        [6.6, 7.7],
        [8.8, 9.9],
    ]
    assert cuda_c.type == ArrayType(OptionType(RegularType(NumpyType("float64"), 2)), 6)


def test_1586_option_regular_axis1():
    a1 = ak.from_json("[[0.0, 1.1], null, [2.2, 3.3]]")
    a2 = ak.from_json("[[4.4, 5.5, 6.6], [7, 8, 9], [7.7, 8.8, 9.9]]")
    cuda_a1 = ak.to_backend(a1, "cuda")
    cuda_a2 = ak.to_backend(a2, "cuda")

    cuda_a1 = ak.to_regular(cuda_a1, axis=1)
    cuda_a2 = ak.to_regular(cuda_a2, axis=1)
    cuda_c = ak.concatenate([cuda_a1, cuda_a2], axis=1)
    assert cuda_c.to_list() == [
        [0.0, 1.1, 4.4, 5.5, 6.6],
        [7, 8, 9],
        [2.2, 3.3, 7.7, 8.8, 9.9],
    ]
    assert cuda_c.type == ArrayType(ListType(NumpyType("float64")), 3)


def test_1586_regular_numpy():
    a1 = ak.from_json("[[0.0, 1.1], [2.2, 3.3]]")
    a2 = ak.Array(np.array([[4.4, 5.5], [6.6, 7.7], [8.8, 9.9]]))
    cuda_a1 = ak.to_backend(a1, "cuda")
    cuda_a2 = ak.to_backend(a2, "cuda")

    cuda_a1 = ak.to_regular(cuda_a1, axis=1)
    assert isinstance(cuda_a2.layout, ak.contents.NumpyArray)
    cuda_c = ak.concatenate([cuda_a1, cuda_a2])
    assert cuda_c.to_list() == [
        [0.0, 1.1],
        [2.2, 3.3],
        [4.4, 5.5],
        [6.6, 7.7],
        [8.8, 9.9],
    ]
    assert cuda_c.type == ArrayType(RegularType(NumpyType("float64"), 2), 5)


def test_2663_broadcast_tuples_record_record():
    record_1 = ak.Array(
        [
            [
                {"0": [1, 5, 1], "1": [2, 5, 1]},
                {"0": [3, 5, 1], "1": [4, 5, 1]},
            ]
        ]
    )
    record_2 = ak.Array(
        [
            [
                {"0": [1, 5, 1], "1": [9, 10, 11]},
                {"0": [6, 7, 8], "1": [4, 5, 1]},
            ]
        ]
    )
    cuda_record_1 = ak.to_backend(record_1, "cuda")
    cuda_record_2 = ak.to_backend(record_2, "cuda")
    cuda_result = ak.concatenate([cuda_record_1, cuda_record_2], axis=-1)

    assert ak.almost_equal(
        to_list(cuda_result),
        [
            [
                {"0": [1, 5, 1, 1, 5, 1], "1": [2, 5, 1, 9, 10, 11]},
                {"0": [3, 5, 1, 6, 7, 8], "1": [4, 5, 1, 4, 5, 1]},
            ]
        ],
    )


def test_2663_broadcast_tuples_tuple_tuple():
    tuple_1 = ak.Array(
        [
            [
                ([1, 5, 1], [2, 5, 1]),
                ([3, 5, 1], [4, 5, 1]),
            ]
        ]
    )
    tuple_2 = ak.Array(
        [
            [
                ([1, 5, 1], [9, 10, 11]),
                ([6, 7, 8], [4, 5, 1]),
            ]
        ]
    )

    result = ak.concatenate([tuple_1, tuple_2], axis=-1)
    assert ak.almost_equal(
        result,
        [
            [
                ([1, 5, 1, 1, 5, 1], [2, 5, 1, 9, 10, 11]),
                ([3, 5, 1, 6, 7, 8], [4, 5, 1, 4, 5, 1]),
            ]
        ],
    )


def test_flatten_UnionArray():
    content1 = ak.operations.from_iter(
        [[1.1], [2.2, 2.2], [3.3, 3.3, 3.3]], highlevel=False
    )
    content2 = ak.operations.from_iter(
        [[[3, 3, 3], [3, 3, 3], [3, 3, 3]], [[2, 2], [2, 2]], [[1]]], highlevel=False
    )
    content3 = ak.operations.from_iter(
        [
            [["3", "3", "3"], ["3", "3", "3"], ["3", "3", "3"]],
            [["2", "2"], ["2", "2"]],
            [["1"]],
        ],
        highlevel=False,
    )
    tags = ak.index.Index8(np.array([0, 1, 0, 1, 0, 1], dtype=np.int8))
    index = ak.index.Index64(np.array([0, 0, 1, 1, 2, 2], dtype=np.int64))
    array = ak.contents.UnionArray(tags, index, [content1, content2])

    assert to_list(array) == [
        [1.1],
        [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
        [2.2, 2.2],
        [[2, 2], [2, 2]],
        [3.3, 3.3, 3.3],
        [[1]],
    ]
    assert to_list(array[1:]) == [
        [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
        [2.2, 2.2],
        [[2, 2], [2, 2]],
        [3.3, 3.3, 3.3],
        [[1]],
    ]
    assert ak.operations.to_list(ak.operations.flatten(array)) == [
        1.1,
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        2.2,
        2.2,
        [2, 2],
        [2, 2],
        3.3,
        3.3,
        3.3,
        [1],
    ]
    assert ak.operations.to_list(ak.operations.flatten(array[1:])) == [
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        2.2,
        2.2,
        [2, 2],
        [2, 2],
        3.3,
        3.3,
        3.3,
        [1],
    ]

    array = ak.contents.UnionArray(tags, index, [content2, content3])

    assert to_list(array) == [
        [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
        [["3", "3", "3"], ["3", "3", "3"], ["3", "3", "3"]],
        [[2, 2], [2, 2]],
        [["2", "2"], ["2", "2"]],
        [[1]],
        [["1"]],
    ]
    assert ak.operations.to_list(ak.operations.flatten(array, axis=2)) == [
        [3, 3, 3, 3, 3, 3, 3, 3, 3],
        ["3", "3", "3", "3", "3", "3", "3", "3", "3"],
        [2, 2, 2, 2],
        ["2", "2", "2", "2"],
        [1],
        ["1"],
    ]
    assert ak.operations.to_list(ak.operations.flatten(array[1:], axis=2)) == [
        ["3", "3", "3", "3", "3", "3", "3", "3", "3"],
        [2, 2, 2, 2],
        ["2", "2", "2", "2"],
        [1],
        ["1"],
    ]
    assert ak.operations.to_list(ak.operations.flatten(array[:, 1:], axis=2)) == [
        [3, 3, 3, 3, 3, 3],
        ["3", "3", "3", "3", "3", "3"],
        [2, 2],
        ["2", "2"],
        [],
        [],
    ]


def test_0150_flatten_UnionArray():
    content1 = ak.operations.from_iter(
        [[1.1], [2.2, 2.2], [3.3, 3.3, 3.3]], highlevel=False
    )
    content2 = ak.operations.from_iter(
        [[[3, 3, 3], [3, 3, 3], [3, 3, 3]], [[2, 2], [2, 2]], [[1]]], highlevel=False
    )
    content3 = ak.operations.from_iter(
        [
            [["3", "3", "3"], ["3", "3", "3"], ["3", "3", "3"]],
            [["2", "2"], ["2", "2"]],
            [["1"]],
        ],
        highlevel=False,
    )
    tags = ak.index.Index8(np.array([0, 1, 0, 1, 0, 1], dtype=np.int8))
    index = ak.index.Index64(np.array([0, 0, 1, 1, 2, 2], dtype=np.int64))
    array = ak.contents.UnionArray(tags, index, [content1, content2])
    cuda_array = ak.to_backend(array, "cuda")

    assert to_list(cuda_array) == [
        [1.1],
        [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
        [2.2, 2.2],
        [[2, 2], [2, 2]],
        [3.3, 3.3, 3.3],
        [[1]],
    ]
    assert to_list(cuda_array[1:]) == [
        [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
        [2.2, 2.2],
        [[2, 2], [2, 2]],
        [3.3, 3.3, 3.3],
        [[1]],
    ]
    assert ak.operations.to_list(ak.operations.flatten(cuda_array)) == [
        1.1,
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        2.2,
        2.2,
        [2, 2],
        [2, 2],
        3.3,
        3.3,
        3.3,
        [1],
    ]
    assert ak.operations.to_list(ak.operations.flatten(cuda_array[1:])) == [
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        2.2,
        2.2,
        [2, 2],
        [2, 2],
        3.3,
        3.3,
        3.3,
        [1],
    ]

    array = ak.contents.UnionArray(tags, index, [content2, content3])
    cuda_array = ak.to_backend(array, "cuda")

    assert to_list(cuda_array) == [
        [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
        [["3", "3", "3"], ["3", "3", "3"], ["3", "3", "3"]],
        [[2, 2], [2, 2]],
        [["2", "2"], ["2", "2"]],
        [[1]],
        [["1"]],
    ]
    assert ak.operations.to_list(ak.operations.flatten(cuda_array, axis=2)) == [
        [3, 3, 3, 3, 3, 3, 3, 3, 3],
        ["3", "3", "3", "3", "3", "3", "3", "3", "3"],
        [2, 2, 2, 2],
        ["2", "2", "2", "2"],
        [1],
        ["1"],
    ]
    assert ak.operations.to_list(ak.operations.flatten(cuda_array[1:], axis=2)) == [
        ["3", "3", "3", "3", "3", "3", "3", "3", "3"],
        [2, 2, 2, 2],
        ["2", "2", "2", "2"],
        [1],
        ["1"],
    ]
    assert ak.operations.to_list(ak.operations.flatten(cuda_array[:, 1:], axis=2)) == [
        [3, 3, 3, 3, 3, 3],
        ["3", "3", "3", "3", "3", "3"],
        [2, 2],
        ["2", "2"],
        [],
        [],
    ]
