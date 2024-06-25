from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

import awkward as ak
from awkward.types import ArrayType, NumpyType, RegularType

to_list = ak.operations.to_list


@pytest.fixture(scope="function", autouse=True)
def cleanup_cuda():
    yield
    cp._default_memory_pool.free_all_blocks()
    cp.cuda.Device().synchronize()


def test_0150_ByteMaskedArray_flatten():
    content = ak.operations.from_iter(
        [
            [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
            [],
            [[5.5]],
            [[6.6, 7.7, 8.8, 9.9]],
            [[], [10.0, 11.1, 12.2]],
        ],
        highlevel=False,
    )
    mask = ak.index.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    array = ak.contents.ByteMaskedArray(mask, content, valid_when=False)
    cuda_array = ak.to_backend(array, "cuda")

    assert to_list(cuda_array) == [
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
        [],
        None,
        None,
        [[], [10.0, 11.1, 12.2]],
    ]
    assert ak.operations.to_list(ak.operations.flatten(cuda_array, axis=1)) == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
        [],
        [10.0, 11.1, 12.2],
    ]
    assert ak.operations.to_list(ak.operations.flatten(cuda_array, axis=-2)) == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
        [],
        [10.0, 11.1, 12.2],
    ]
    assert ak.operations.to_list(ak.operations.flatten(cuda_array, axis=2)) == [
        [0.0, 1.1, 2.2, 3.3, 4.4],
        [],
        None,
        None,
        [10.0, 11.1, 12.2],
    ]
    assert ak.operations.to_list(ak.operations.flatten(cuda_array, axis=-1)) == [
        [0.0, 1.1, 2.2, 3.3, 4.4],
        [],
        None,
        None,
        [10.0, 11.1, 12.2],
    ]


def test_1586_should_preserve_regulararray_numpy_regular_axis1():
    a1 = ak.Array(np.array([[0.0, 1.1], [2.2, 3.3]]))
    a2 = ak.from_json("[[4.4, 5.5, 6.6], [7.7, 8.8, 9.9]]")
    cuda_a1 = ak.to_backend(a1, "cuda")
    assert isinstance(cuda_a1.layout, ak.contents.NumpyArray)

    a2 = ak.to_regular(a2, axis=1)
    cuda_a2 = ak.to_backend(a2, "cuda")

    c = ak.concatenate([cuda_a1, cuda_a2], axis=1)
    assert c.to_list() == [[0.0, 1.1, 4.4, 5.5, 6.6], [2.2, 3.3, 7.7, 8.8, 9.9]]
    assert c.type == ArrayType(RegularType(NumpyType("float64"), 5), 2)


def test_1586_should_preserve_regulararray_regular_numpy_axis1():
    a1 = ak.from_json("[[0.0, 1.1], [2.2, 3.3]]")
    a2 = ak.Array(np.array([[4.4, 5.5, 6.6], [7.7, 8.8, 9.9]]))

    cuda_a1 = ak.to_backend(a1, "cuda")
    cuda_a2 = ak.to_backend(a2, "cuda")

    cuda_a1 = ak.to_regular(cuda_a1, axis=1)

    assert isinstance(cuda_a2.layout, ak.contents.NumpyArray)
    c = ak.concatenate([cuda_a1, cuda_a2], axis=1)
    assert c.to_list() == [[0.0, 1.1, 4.4, 5.5, 6.6], [2.2, 3.3, 7.7, 8.8, 9.9]]
    assert c.type == ArrayType(RegularType(NumpyType("float64"), 5), 2)


def test_1586_should_preserve_regulararray_regular_regular_axis1():
    a1 = ak.from_json("[[0.0, 1.1], [2.2, 3.3]]")
    a2 = ak.from_json("[[4.4, 5.5, 6.6], [7.7, 8.8, 9.9]]")

    cuda_a1 = ak.to_backend(a1, "cuda")
    cuda_a2 = ak.to_backend(a2, "cuda")

    cuda_a1 = ak.to_regular(cuda_a1, axis=1)
    cuda_a2 = ak.to_regular(cuda_a2, axis=1)
    c = ak.concatenate([cuda_a1, cuda_a2], axis=1)
    assert c.to_list() == [[0.0, 1.1, 4.4, 5.5, 6.6], [2.2, 3.3, 7.7, 8.8, 9.9]]
    assert c.type == ArrayType(RegularType(NumpyType("float64"), 5), 2)


def test_0072_regulararray_fillna_unionarray():
    content1 = ak.operations.from_iter([[], [1.1], [2.2, 2.2]], highlevel=False)
    content2 = ak.operations.from_iter([["two", "two"], ["one"], []], highlevel=False)
    tags = ak.index.Index8(np.array([0, 1, 0, 1, 0, 1], dtype=np.int8))
    index = ak.index.Index64(np.array([0, 0, 1, 1, 2, 2], dtype=np.int64))
    array = ak.contents.UnionArray(tags, index, [content1, content2])
    cuda_array = ak.to_backend(array, "cuda")

    assert to_list(cuda_array) == [[], ["two", "two"], [1.1], ["one"], [2.2, 2.2], []]

    padded_array = ak._do.pad_none(array, 2, 1)
    assert to_list(padded_array) == [
        [None, None],
        ["two", "two"],
        [1.1, None],
        ["one", None],
        [2.2, 2.2],
        [None, None],
    ]

    value = ak.contents.NumpyArray(np.array([777]))

    assert to_list(ak._do.fill_none(padded_array, value)) == [
        [777, 777],
        ["two", "two"],
        [1.1, 777],
        ["one", 777],
        [2.2, 2.2],
        [777, 777],
    ]


def test_0590_allow_regulararray_size_zero_ListOffsetArray_rpad_and_clip():
    array = ak.highlevel.Array([[1, 2, 3], [], [4, 5]])
    assert ak.operations.pad_none(array, 0, clip=True).to_list() == [
        [],
        [],
        [],
    ]

    array = ak.highlevel.Array([[1, 2, 3], [], [4, 5]])
    cuda_array = ak.to_backend(array, "cuda")

    assert ak.operations.pad_none(cuda_array, 0).to_list() == [
        [1, 2, 3],
        [],
        [4, 5],
    ]
