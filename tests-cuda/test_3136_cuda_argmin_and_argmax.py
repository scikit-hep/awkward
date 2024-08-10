from __future__ import annotations

import cupy as cp
import pytest

import awkward as ak

to_list = ak.operations.to_list


@pytest.fixture(scope="function", autouse=True)
def cleanup_cuda():
    yield
    cp._default_memory_pool.free_all_blocks()
    cp.cuda.Device().synchronize()


@pytest.mark.skip(
    reason="awkward_reduce_argmin and awkward_reduce_argmax are not implemented"
)
def test_0835_argmin_argmax_axis_None():
    array = ak.highlevel.Array(
        [
            [
                [2022, 2023, 2025],
                [],
                [2027, 2011],
                [2013],
            ],
            [],
            [[2017, 2019], [2023]],
        ],
    )
    cuda_array = ak.to_backend(array, "cuda")

    assert ak.operations.argmin(cuda_array) == 4
    assert ak.operations.argmax(cuda_array) == 3


@pytest.mark.skip(
    reason="awkward_reduce_argmin and awkward_reduce_argmax are not implemented"
)
def test_1106_argminmax_axis_None_missing_values():
    array = ak.highlevel.Array([1, 2, 3, None, 4])

    cuda_array = ak.to_backend(array, "cuda")

    assert ak.operations.argmax(cuda_array) == 4


@pytest.mark.skip(
    reason="awkward_reduce_argmin and awkward_reduce_argmax are not implemented"
)
def test_0070_argmin_and_argmax_jagged():
    v2_array = ak.operations.from_iter(
        [[2.2, 1.1, 3.3], [], [4.4, 5.5], [5.5], [-4.4, -5.5, -6.6]], highlevel=False
    )

    cuda_v2_array = ak.to_backend(v2_array, "cuda", highlevel=False)

    assert to_list(ak.argmin(cuda_v2_array, axis=1, highlevel=False)) == [
        1,
        None,
        0,
        0,
        2,
    ]
    assert (
        ak.argmin(cuda_v2_array.to_typetracer(), axis=1, highlevel=False).form
        == ak.argmin(cuda_v2_array, axis=1, highlevel=False).form
    )

    index2 = ak.index.Index64(cp.array([4, 3, 2, 1, 0], dtype=cp.int64))
    cuda_v2_array2 = ak.contents.IndexedArray(index2, cuda_v2_array)

    assert to_list(ak.argmin(cuda_v2_array2, axis=1, highlevel=False)) == [
        2,
        0,
        0,
        None,
        1,
    ]
    assert (
        ak.argmin(cuda_v2_array2.to_typetracer(), axis=1, highlevel=False).form
        == ak.argmin(cuda_v2_array2, axis=1, highlevel=False).form
    )

    index3 = ak.index.Index64(cp.array([4, 3, -1, 4, 0], dtype=cp.int64))
    cuda_v2_array2 = ak.contents.IndexedOptionArray(index3, cuda_v2_array)

    assert to_list(ak.argmin(cuda_v2_array2, axis=1, highlevel=False)) == [
        2,
        0,
        None,
        2,
        1,
    ]
    assert (
        ak.argmin(cuda_v2_array2.to_typetracer(), axis=1, highlevel=False).form
        == ak.argmin(cuda_v2_array2, axis=1, highlevel=False).form
    )
    assert to_list(ak.argmin(cuda_v2_array2, axis=-1, highlevel=False)) == [
        2,
        0,
        None,
        2,
        1,
    ]
    assert (
        ak.argmin(cuda_v2_array2.to_typetracer(), axis=-1, highlevel=False).form
        == ak.argmin(cuda_v2_array2, axis=-1, highlevel=False).form
    )


@pytest.mark.skip(
    reason="awkward_reduce_argmin and awkward_reduce_argmax are not implemented"
)
def test_0070_argmin_and_argmax_missing():
    array = ak.operations.from_iter(
        [[[2.2, 1.1, 3.3]], [[]], [None, None, None], [[-4.4, -5.5, -6.6]]],
        highlevel=False,
    )

    cuda_array = ak.to_backend(array, "cuda", highlevel=False)

    assert to_list(ak.argmin(cuda_array, axis=2, highlevel=False)) == [
        [1],
        [None],
        [None, None, None],
        [2],
    ]
    assert (
        ak.argmin(cuda_array.to_typetracer(), axis=2, highlevel=False).form
        == ak.argmin(cuda_array, axis=2, highlevel=False).form
    )


@pytest.mark.skip(
    reason="awkward_reduce_argmin and awkward_reduce_argmax are not implemented"
)
def test_0115_generic_reducer_operation_ByteMaskedArray():
    content = ak.operations.from_iter(
        [
            [[1.1, 0.0, 2.2], [], [3.3, 4.4]],
            [],
            [[5.5]],
            [[6.6, 9.9, 8.8, 7.7]],
            [[], [12.2, 11.1, 10.0]],
        ],
        highlevel=False,
    )
    mask = ak.index.Index8(cp.array([0, 0, 1, 1, 0], dtype=cp.int8))
    content = ak.to_backend(content, "cuda", highlevel=False)

    cuda_v2_array = ak.contents.ByteMaskedArray(mask, content, valid_when=False)

    assert to_list(cuda_v2_array) == [
        [[1.1, 0.0, 2.2], [], [3.3, 4.4]],
        [],
        None,
        None,
        [[], [12.2, 11.1, 10.0]],
    ]
    assert to_list(ak.argmin(cuda_v2_array, axis=-1, highlevel=False)) == [
        [1, None, 0],
        [],
        None,
        None,
        [None, 2],
    ]
    assert (
        ak.argmin(cuda_v2_array.to_typetracer(), axis=-1, highlevel=False).form
        == ak.argmin(cuda_v2_array, axis=-1, highlevel=False).form
    )


@pytest.mark.skip(
    reason="awkward_reduce_argmin and awkward_reduce_argmax are not implemented"
)
@pytest.mark.parametrize(
    "func",
    [
        ak.argmin,
        ak.argmax,
    ],
)
def test_2754_highlevel_behavior_missing_reducers(func):
    behavior_1 = {"foo": "bar"}
    behavior_2 = {"baz": "bargh!"}

    array = ak.Array([[1, 2, 3, 4], [5], [10]])

    cuda_array = ak.to_backend(array, "cuda")

    assert isinstance(func(cuda_array, axis=1, highlevel=True), ak.Array)
    assert isinstance(func(cuda_array, axis=1, highlevel=False), ak.contents.Content)
    assert (
        func(
            ak.Array(cuda_array, behavior=behavior_1),
            axis=1,
            highlevel=True,
            behavior=behavior_2,
        ).behavior
        == behavior_2
    )
    assert (
        func(
            ak.Array(cuda_array, behavior=behavior_1),
            axis=1,
            highlevel=True,
        ).behavior
        == behavior_1
    )
