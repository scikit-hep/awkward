from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

import awkward as ak

to_list = ak.operations.to_list

content = ak.contents.NumpyArray(
    np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
)
starts1 = ak.index.IndexU32(np.array([0, 3, 3, 5, 6], np.uint32))
stops1 = ak.index.IndexU32(np.array([3, 3, 5, 6, 9], np.uint32))
offsets1 = ak.index.IndexU32(np.array([0, 3, 3, 5, 6, 9], np.uint32))
starts2 = ak.index.IndexU32(np.array([0, 2, 3, 3], np.uint32))
stops2 = ak.index.IndexU32(np.array([2, 3, 3, 5], np.uint32))
offsets2 = ak.index.IndexU32(np.array([0, 2, 3, 3, 5], np.uint32))


@pytest.fixture(scope="function", autouse=True)
def cleanup_cuda():
    yield
    cp._default_memory_pool.free_all_blocks()
    cp.cuda.Device().synchronize()


def tests_0020_support_unsigned_indexes_listarray_ellipsis():
    array1 = ak.contents.ListArray(starts1, stops1, content)
    array2 = ak.contents.ListArray(starts2, stops2, array1)

    cuda_array1 = ak.to_backend(array1, "cuda", highlevel=False)
    cuda_array2 = ak.to_backend(array2, "cuda", highlevel=False)

    assert to_list(cuda_array1[Ellipsis, 1:]) == [
        [2.2, 3.3],
        [],
        [5.5],
        [],
        [8.8, 9.9],
    ]
    assert (
        cuda_array1.to_typetracer()[Ellipsis, 1:].form == cuda_array1[Ellipsis, 1:].form
    )
    assert to_list(cuda_array2[Ellipsis, 1:]) == [
        [[2.2, 3.3], []],
        [[5.5]],
        [],
        [[], [8.8, 9.9]],
    ]
    assert (
        cuda_array2.to_typetracer()[Ellipsis, 1:].form == cuda_array2[Ellipsis, 1:].form
    )


def tests_0020_support_unsigned_indexes_listoffsetarray_ellipsis():
    array1 = ak.contents.ListOffsetArray(offsets1, content)
    array2 = ak.contents.ListOffsetArray(offsets2, array1)

    cuda_array1 = ak.to_backend(array1, "cuda", highlevel=False)
    cuda_array2 = ak.to_backend(array2, "cuda", highlevel=False)

    assert to_list(cuda_array1[Ellipsis, 1:]) == [
        [2.2, 3.3],
        [],
        [5.5],
        [],
        [8.8, 9.9],
    ]
    assert (
        cuda_array1.to_typetracer()[Ellipsis, 1:].form == cuda_array1[Ellipsis, 1:].form
    )
    assert to_list(cuda_array2[Ellipsis, 1:]) == [
        [[2.2, 3.3], []],
        [[5.5]],
        [],
        [[], [8.8, 9.9]],
    ]
    assert (
        cuda_array2.to_typetracer()[Ellipsis, 1:].form == cuda_array2[Ellipsis, 1:].form
    )


def tests_0020_support_unsigned_indexes_listarray_array_slice():
    array1 = ak.contents.ListArray(starts1, stops1, content)
    array2 = ak.contents.ListArray(starts2, stops2, array1)

    cuda_array2 = ak.to_backend(array2, "cuda", highlevel=False)

    assert to_list(cuda_array2[[0, 0, 1, 1, 1, 0]]) == [
        [[1.1, 2.2, 3.3], []],
        [[1.1, 2.2, 3.3], []],
        [[4.4, 5.5]],
        [[4.4, 5.5]],
        [[4.4, 5.5]],
        [[1.1, 2.2, 3.3], []],
    ]
    assert (
        cuda_array2.to_typetracer()[[0, 0, 1, 1, 1, 0]].form
        == cuda_array2[[0, 0, 1, 1, 1, 0]].form
    )

    assert to_list(cuda_array2[[0, 0, 1, 1, 1, 0], :]) == [
        [[1.1, 2.2, 3.3], []],
        [[1.1, 2.2, 3.3], []],
        [[4.4, 5.5]],
        [[4.4, 5.5]],
        [[4.4, 5.5]],
        [[1.1, 2.2, 3.3], []],
    ]
    assert (
        cuda_array2.to_typetracer()[[0, 0, 1, 1, 1, 0], :].form
        == cuda_array2[[0, 0, 1, 1, 1, 0], :].form
    )
    assert to_list(cuda_array2[[0, 0, 1, 1, 1, 0], :, 1:]) == [
        [[2.2, 3.3], []],
        [[2.2, 3.3], []],
        [[5.5]],
        [[5.5]],
        [[5.5]],
        [[2.2, 3.3], []],
    ]
    assert (
        cuda_array2.to_typetracer()[[0, 0, 1, 1, 1, 0], :, 1:].form
        == cuda_array2[[0, 0, 1, 1, 1, 0], :, 1:].form
    )


def tests_0020_support_unsigned_indexes_listoffsetarray_array_slice():
    array1 = ak.contents.ListOffsetArray(offsets1, content)
    array2 = ak.contents.ListOffsetArray(offsets2, array1)

    cuda_array2 = ak.to_backend(array2, "cuda", highlevel=False)

    assert to_list(cuda_array2[[0, 0, 1, 1, 1, 0]]) == [
        [[1.1, 2.2, 3.3], []],
        [[1.1, 2.2, 3.3], []],
        [[4.4, 5.5]],
        [[4.4, 5.5]],
        [[4.4, 5.5]],
        [[1.1, 2.2, 3.3], []],
    ]
    assert (
        cuda_array2.to_typetracer()[[0, 0, 1, 1, 1, 0]].form
        == cuda_array2[[0, 0, 1, 1, 1, 0]].form
    )
    assert to_list(cuda_array2[[0, 0, 1, 1, 1, 0], :]) == [
        [[1.1, 2.2, 3.3], []],
        [[1.1, 2.2, 3.3], []],
        [[4.4, 5.5]],
        [[4.4, 5.5]],
        [[4.4, 5.5]],
        [[1.1, 2.2, 3.3], []],
    ]
    assert (
        cuda_array2.to_typetracer()[[0, 0, 1, 1, 1, 0], :].form
        == cuda_array2[[0, 0, 1, 1, 1, 0], :].form
    )
    assert to_list(cuda_array2[[0, 0, 1, 1, 1, 0], :, 1:]) == [
        [[2.2, 3.3], []],
        [[2.2, 3.3], []],
        [[5.5]],
        [[5.5]],
        [[5.5]],
        [[2.2, 3.3], []],
    ]
    assert (
        cuda_array2.to_typetracer()[[0, 0, 1, 1, 1, 0], :, 1:].form
        == cuda_array2[[0, 0, 1, 1, 1, 0], :, 1:].form
    )


def tests_0020_support_unsigned_indexes_listarray_array():
    array1 = ak.contents.ListArray(starts1, stops1, content)

    cuda_array1 = ak.to_backend(array1, "cuda", highlevel=False)

    assert to_list(cuda_array1[cp.array([2, 0, 0, 1, -1])]) == [
        [4.4, 5.5],
        [1.1, 2.2, 3.3],
        [1.1, 2.2, 3.3],
        [],
        [7.7, 8.8, 9.9],
    ]
    assert (
        cuda_array1.to_typetracer()[cp.array([2, 0, 0, 1, -1])].form
        == cuda_array1[cp.array([2, 0, 0, 1, -1])].form
    )
    assert to_list(cuda_array1[cp.array([2, 0, 0, -1]), cp.array([1, 1, 0, 0])]) == [
        5.5,
        2.2,
        1.1,
        7.7,
    ]
    assert (
        cuda_array1.to_typetracer()[
            cp.array([2, 0, 0, -1]), cp.array([1, 1, 0, 0])
        ].form
        == cuda_array1[cp.array([2, 0, 0, -1]), cp.array([1, 1, 0, 0])].form
    )


def tests_0020_support_unsigned_indexes_listoffsetarray_array():
    array1 = ak.contents.ListOffsetArray(offsets1, content)

    cuda_array1 = ak.to_backend(array1, "cuda", highlevel=False)

    assert to_list(cuda_array1[cp.array([2, 0, 0, 1, -1])]) == [
        [4.4, 5.5],
        [1.1, 2.2, 3.3],
        [1.1, 2.2, 3.3],
        [],
        [7.7, 8.8, 9.9],
    ]
    assert (
        cuda_array1.to_typetracer()[cp.array([2, 0, 0, 1, -1])].form
        == cuda_array1[cp.array([2, 0, 0, 1, -1])].form
    )
    assert to_list(cuda_array1[cp.array([2, 0, 0, -1]), cp.array([1, 1, 0, 0])]) == [
        5.5,
        2.2,
        1.1,
        7.7,
    ]
    assert (
        cuda_array1.to_typetracer()[
            cp.array([2, 0, 0, -1]), cp.array([1, 1, 0, 0])
        ].form
        == cuda_array1[cp.array([2, 0, 0, -1]), cp.array([1, 1, 0, 0])].form
    )


def tests_0020_support_unsigned_indexes_listarray_slice():
    array1 = ak.contents.ListArray(starts1, stops1, content)
    array2 = ak.contents.ListArray(starts2, stops2, array1)

    cuda_array1 = ak.to_backend(array1, "cuda", highlevel=False)
    cuda_array2 = ak.to_backend(array2, "cuda", highlevel=False)

    assert to_list(cuda_array1[1:-1]) == [[], [4.4, 5.5], [6.6]]
    assert cuda_array1.to_typetracer()[1:-1].form == cuda_array1[1:-1].form
    assert to_list(cuda_array1[1:-1,]) == [[], [4.4, 5.5], [6.6]]
    assert cuda_array1.to_typetracer()[1:-1,].form == cuda_array1[1:-1,].form
    assert to_list(cuda_array2[1:-1]) == [[[4.4, 5.5]], []]
    assert cuda_array2.to_typetracer()[1:-1].form == cuda_array2[1:-1].form
    assert to_list(cuda_array2[1:-1,]) == [[[4.4, 5.5]], []]
    assert cuda_array2.to_typetracer()[1:-1,].form == cuda_array2[1:-1,].form


def tests_0020_support_unsigned_indexes_listoffsetarray_slice():
    array1 = ak.contents.ListOffsetArray(offsets1, content)
    array2 = ak.contents.ListOffsetArray(offsets2, array1)

    cuda_array1 = ak.to_backend(array1, "cuda", highlevel=False)
    cuda_array2 = ak.to_backend(array2, "cuda", highlevel=False)

    assert to_list(cuda_array1[1:-1]) == [[], [4.4, 5.5], [6.6]]
    assert cuda_array1.to_typetracer()[1:-1].form == cuda_array1[1:-1].form
    assert to_list(cuda_array1[1:-1,]) == [[], [4.4, 5.5], [6.6]]
    assert cuda_array1.to_typetracer()[1:-1,].form == cuda_array1[1:-1,].form
    assert to_list(cuda_array2[1:-1]) == [[[4.4, 5.5]], []]
    assert cuda_array2.to_typetracer()[1:-1].form == cuda_array2[1:-1].form
    assert to_list(cuda_array2[1:-1,]) == [[[4.4, 5.5]], []]
    assert cuda_array2.to_typetracer()[1:-1,].form == cuda_array2[1:-1,].form


def tests_0020_support_unsigned_indexes_listarray_slice_slice():
    array1 = ak.contents.ListArray(starts1, stops1, content)

    cuda_array1 = ak.to_backend(array1, "cuda", highlevel=False)

    assert to_list(cuda_array1[2:]) == [[4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]
    assert cuda_array1.to_typetracer()[2:].form == cuda_array1[2:].form
    assert to_list(cuda_array1[2:, 1:]) == [[5.5], [], [8.8, 9.9]]
    assert cuda_array1.to_typetracer()[2:, 1:].form == cuda_array1[2:, 1:].form
    assert to_list(cuda_array1[2:, :-1]) == [[4.4], [], [7.7, 8.8]]
    assert cuda_array1.to_typetracer()[2:, :-1].form == cuda_array1[2:, :-1].form


def tests_0020_support_unsigned_indexes_listoffsetarray_slice_slice():
    array1 = ak.contents.ListOffsetArray(offsets1, content)

    cuda_array1 = ak.to_backend(array1, "cuda", highlevel=False)

    assert to_list(cuda_array1[2:]) == [[4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]
    assert cuda_array1.to_typetracer()[2:].form == cuda_array1[2:].form
    assert to_list(cuda_array1[2:, 1:]) == [[5.5], [], [8.8, 9.9]]
    assert cuda_array1.to_typetracer()[2:, 1:].form == cuda_array1[2:, 1:].form
    assert to_list(cuda_array1[2:, :-1]) == [[4.4], [], [7.7, 8.8]]
    assert cuda_array1.to_typetracer()[2:, :-1].form == cuda_array1[2:, :-1].form


class ReversibleArray(ak.Array):
    def reversed(self):
        return self[..., ::-1]


def test_2549_list_nominal_type_class():
    behavior = {"reversible": ReversibleArray}
    reversible_array = ak.with_parameter(
        [[1, 2, 3], [4, 5, 6, 7], [8, 9]], "__list__", "reversible", behavior=behavior
    )

    cuda_reversible_array = ak.to_backend(reversible_array, "cuda")

    assert isinstance(cuda_reversible_array, ReversibleArray)
    assert cuda_reversible_array.to_list() == [[1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert cuda_reversible_array.reversed().to_list() == [
        [3, 2, 1],
        [7, 6, 5, 4],
        [9, 8],
    ]


def test_2549_list_nominal_type_deep_class():
    behavior = {"reversible": ReversibleArray, ("*", "reversible"): ReversibleArray}
    reversible_array = ak.with_parameter(
        [[1, 2, 3], [4, 5, 6, 7], [8, 9]], "__list__", "reversible", behavior=behavior
    )
    outer_array = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64([0, 2, 3, 3]), reversible_array.layout
        ),
        behavior=behavior,
    )

    cuda_outer_array = ak.to_backend(outer_array, "cuda")

    assert isinstance(cuda_outer_array, ReversibleArray)
    assert cuda_outer_array.to_list() == [[[1, 2, 3], [4, 5, 6, 7]], [[8, 9]], []]
    assert cuda_outer_array.reversed().to_list() == [
        [[3, 2, 1], [7, 6, 5, 4]],
        [[9, 8]],
        [],
    ]


def test_2549_list_nominal_type_ufunc():
    behavior = {"reversible": ReversibleArray}
    reversible_array = ak.with_parameter(
        [[1, 2, 3], [4, 5, 6, 7], [8, 9]], "__list__", "reversible", behavior=behavior
    )
    cuda_reversible_array = ak.to_backend(reversible_array, "cuda")

    assert isinstance(cuda_reversible_array, ReversibleArray)

    def reversible_add(x, y):
        return ak.with_parameter(x.reversed(), "__list__", None) + ak.with_parameter(
            y.reversed(), "__list__", None
        )

    ak.behavior[np.add, "reversible", "reversible"] = reversible_add

    assert (cuda_reversible_array + cuda_reversible_array).to_list() == [
        [6, 4, 2],
        [14, 12, 10, 8],
        [18, 16],
    ]
    with pytest.raises(TypeError, match=r"overloads for custom types"):
        cuda_reversible_array + ak.with_parameter(
            cuda_reversible_array, "__list__", "non-reversible"
        )


def test_0150_flatten_ListOffsetArray():
    array = ak.highlevel.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).layout

    cuda_array = ak.to_backend(array, "cuda")

    assert ak.operations.to_list(ak.operations.flatten(cuda_array)) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]
    assert ak.operations.to_list(ak.operations.flatten(cuda_array[1:])) == [4.4, 5.5]

    array = ak.highlevel.Array(
        [[[0.0, 1.1, 2.2], [], [3.3, 4.4]], [], [[5.5]], [[], [6.6, 7.7, 8.8, 9.9]]]
    ).layout

    cuda_array = ak.to_backend(array, "cuda")

    assert ak.operations.to_list(ak.operations.flatten(cuda_array)) == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
        [5.5],
        [],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.operations.to_list(ak.operations.flatten(cuda_array[1:])) == [
        [5.5],
        [],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.operations.to_list(ak.operations.flatten(cuda_array[:, 1:])) == [
        [],
        [3.3, 4.4],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.operations.to_list(ak.operations.flatten(cuda_array, axis=2)) == [
        [0.0, 1.1, 2.2, 3.3, 4.4],
        [],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.operations.to_list(ak.operations.flatten(cuda_array[1:], axis=2)) == [
        [],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.operations.to_list(ak.operations.flatten(cuda_array[:, 1:], axis=2)) == [
        [3.3, 4.4],
        [],
        [],
        [6.6, 7.7, 8.8, 9.9],
    ]

    array = ak.highlevel.Array(
        cp.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7).tolist()
    ).layout

    cuda_array = ak.to_backend(array, "cuda")

    assert (
        ak.operations.to_list(ak.operations.flatten(cuda_array, axis=1))
        == cp.arange(2 * 3 * 5 * 7).reshape(2 * 3, 5, 7).tolist()
    )
    assert (
        ak.operations.to_list(ak.operations.flatten(cuda_array, axis=2))
        == cp.arange(2 * 3 * 5 * 7).reshape(2, 3 * 5, 7).tolist()
    )
    assert (
        ak.operations.to_list(ak.operations.flatten(cuda_array, axis=3))
        == cp.arange(2 * 3 * 5 * 7).reshape(2, 3, 5 * 7).tolist()
    )

    array = ak.highlevel.Array(
        ak.operations.from_iter(
            cp.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7).tolist(), highlevel=False
        )
    ).layout

    cuda_array = ak.to_backend(array, "cuda")

    assert (
        ak.operations.to_list(ak.operations.flatten(cuda_array, axis=1))
        == cp.arange(2 * 3 * 5 * 7).reshape(2 * 3, 5, 7).tolist()
    )
    assert (
        ak.operations.to_list(ak.operations.flatten(cuda_array, axis=2))
        == cp.arange(2 * 3 * 5 * 7).reshape(2, 3 * 5, 7).tolist()
    )
    assert (
        ak.operations.to_list(ak.operations.flatten(cuda_array, axis=3))
        == cp.arange(2 * 3 * 5 * 7).reshape(2, 3, 5 * 7).tolist()
    )

    array = ak.highlevel.Array(cp.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)).layout

    cuda_array = ak.to_backend(array, "cuda")

    assert (
        ak.operations.to_list(ak.operations.flatten(cuda_array, axis=1))
        == cp.arange(2 * 3 * 5 * 7).reshape(2 * 3, 5, 7).tolist()
    )
    assert (
        ak.operations.to_list(ak.operations.flatten(cuda_array, axis=2))
        == cp.arange(2 * 3 * 5 * 7).reshape(2, 3 * 5, 7).tolist()
    )
    assert (
        ak.operations.to_list(ak.operations.flatten(cuda_array, axis=3))
        == cp.arange(2 * 3 * 5 * 7).reshape(2, 3, 5 * 7).tolist()
    )


def test_0150_flatten_IndexedArray():
    array = ak.highlevel.Array(
        [[1.1, 2.2, None, 3.3], None, [], None, [4.4, 5.5], None]
    ).layout

    cuda_array = ak.to_backend(array, "cuda")

    assert ak.operations.to_list(ak.operations.flatten(cuda_array)) == [
        1.1,
        2.2,
        None,
        3.3,
        4.4,
        5.5,
    ]
    assert ak.operations.to_list(ak.operations.flatten(cuda_array[1:])) == [4.4, 5.5]

    array = ak.highlevel.Array(
        [
            [[0.0, 1.1, 2.2], None, None, [3.3, 4.4]],
            [],
            [[5.5]],
            [[], [6.6, 7.7, 8.8, 9.9]],
        ]
    ).layout

    cuda_array = ak.to_backend(array, "cuda")

    assert ak.operations.to_list(ak.operations.flatten(cuda_array, axis=2)) == [
        [0.0, 1.1, 2.2, 3.3, 4.4],
        [],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.operations.to_list(ak.operations.flatten(cuda_array[1:], axis=2)) == [
        [],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.operations.to_list(ak.operations.flatten(cuda_array[:, 1:], axis=2)) == [
        [3.3, 4.4],
        [],
        [],
        [6.6, 7.7, 8.8, 9.9],
    ]

    array = ak.highlevel.Array(
        [
            [[0.0, 1.1, 2.2], [3.3, 4.4]],
            [],
            [[5.5]],
            None,
            None,
            [[], [6.6, 7.7, 8.8, 9.9]],
        ]
    ).layout

    cuda_array = ak.to_backend(array, "cuda")

    assert ak.operations.to_list(ak.operations.flatten(cuda_array, axis=2)) == [
        [0.0, 1.1, 2.2, 3.3, 4.4],
        [],
        [5.5],
        None,
        None,
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.operations.to_list(ak.operations.flatten(cuda_array[1:], axis=2)) == [
        [],
        [5.5],
        None,
        None,
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.operations.to_list(ak.operations.flatten(cuda_array[:, 1:], axis=2)) == [
        [3.3, 4.4],
        [],
        [],
        None,
        None,
        [6.6, 7.7, 8.8, 9.9],
    ]

    array = ak.highlevel.Array(
        [
            [[0.0, 1.1, None, 2.2], None, [], None, [3.3, 4.4]],
            None,
            [],
            [[5.5]],
            None,
            [[], [6.6, None, 7.7, 8.8, 9.9], None],
        ]
    ).layout

    cuda_array = ak.to_backend(array, "cuda")

    assert ak.operations.to_list(ak.operations.flatten(cuda_array)) == [
        [0.0, 1.1, None, 2.2],
        None,
        [],
        None,
        [3.3, 4.4],
        [5.5],
        [],
        [6.6, None, 7.7, 8.8, 9.9],
        None,
    ]
    assert ak.operations.to_list(ak.operations.flatten(cuda_array, axis=2)) == [
        [0.0, 1.1, None, 2.2, 3.3, 4.4],
        None,
        [],
        [5.5],
        None,
        [6.6, None, 7.7, 8.8, 9.9],
    ]
    assert ak.operations.to_list(ak.operations.flatten(cuda_array[1:], axis=2)) == [
        None,
        [],
        [5.5],
        None,
        [6.6, None, 7.7, 8.8, 9.9],
    ]
    assert ak.operations.to_list(ak.operations.flatten(cuda_array[:, 1:], axis=2)) == [
        [3.3, 4.4],
        None,
        [],
        [],
        None,
        [6.6, None, 7.7, 8.8, 9.9],
    ]

    content = ak.operations.from_iter(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    index = ak.index.Index64(np.array([2, 1, 0, 3, 3, 4], dtype=np.int64))
    array = ak.contents.IndexedArray(index, content)

    cuda_array = ak.to_backend(array, "cuda")

    assert to_list(cuda_array) == [
        [3.3, 4.4],
        [],
        [0.0, 1.1, 2.2],
        [5.5],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.operations.to_list(ak.operations.flatten(cuda_array)) == [
        3.3,
        4.4,
        0.0,
        1.1,
        2.2,
        5.5,
        5.5,
        6.6,
        7.7,
        8.8,
        9.9,
    ]

    content = ak.operations.from_iter(
        [[[0.0, 1.1, 2.2], [], [3.3, 4.4]], [], [[5.5]], [[], [6.6, 7.7, 8.8, 9.9]]],
        highlevel=False,
    )
    index = ak.index.Index64(np.array([2, 2, 1, 0, 3], dtype=np.int64))
    array = ak.contents.IndexedArray(index, content)

    cuda_array = ak.to_backend(array, "cuda")

    assert to_list(cuda_array) == [
        [[5.5]],
        [[5.5]],
        [],
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
        [[], [6.6, 7.7, 8.8, 9.9]],
    ]
    assert ak.operations.to_list(ak.operations.flatten(cuda_array, axis=2)) == [
        [5.5],
        [5.5],
        [],
        [0.0, 1.1, 2.2, 3.3, 4.4],
        [6.6, 7.7, 8.8, 9.9],
    ]


def test_0150_flatten_RecordArray():
    array = ak.highlevel.Array(
        [
            {"x": [], "y": [[3, 3, 3]]},
            {"x": [[1]], "y": [[2, 2]]},
            {"x": [[2], [2]], "y": [[1]]},
            {"x": [[3], [3], [3]], "y": [[]]},
        ]
    ).layout

    cuda_array = ak.to_backend(array, "cuda")

    assert ak.operations.to_list(ak.operations.flatten(cuda_array, axis=2)) == [
        {"x": [], "y": [3, 3, 3]},
        {"x": [1], "y": [2, 2]},
        {"x": [2, 2], "y": [1]},
        {"x": [3, 3, 3], "y": []},
    ]
    assert ak.operations.to_list(ak.operations.flatten(cuda_array[1:], axis=2)) == [
        {"x": [1], "y": [2, 2]},
        {"x": [2, 2], "y": [1]},
        {"x": [3, 3, 3], "y": []},
    ]
    assert ak.operations.to_list(ak.operations.flatten(cuda_array[:, 1:], axis=2)) == [
        {"x": [], "y": []},
        {"x": [], "y": []},
        {"x": [2], "y": []},
        {"x": [3, 3], "y": []},
    ]
