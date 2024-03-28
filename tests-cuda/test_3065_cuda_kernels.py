from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

import awkward as ak
from awkward.operations.ak_concatenate import enforce_concatenated_form

to_list = ak.operations.to_list


def test_0582_propagate_context_in_broadcast_and_apply_firsts():
    array = ak.Array([[[0, 1, 2], []], [[3, 4]], [], [[5], [6, 7, 8, 9]]])
    cuda_array = ak.to_backend(array, "cuda")

    assert to_list(ak.operations.firsts(cuda_array, axis=0)) == [[0, 1, 2], []]
    assert to_list(ak.operations.firsts(cuda_array, axis=1)) == [
        [0, 1, 2],
        [3, 4],
        None,
        [5],
    ]
    assert to_list(ak.operations.firsts(cuda_array, axis=2)) == [
        [0, None],
        [3],
        [],
        [5, 6],
    ]
    assert to_list(ak.operations.firsts(cuda_array, axis=-1)) == [
        [0, None],
        [3],
        [],
        [5, 6],
    ]
    assert to_list(ak.operations.firsts(cuda_array, axis=-2)) == [
        [0, 1, 2],
        [3, 4],
        None,
        [5],
    ]
    assert to_list(ak.operations.firsts(cuda_array, axis=-3)) == [
        [0, 1, 2],
        [],
    ]

    with pytest.raises(ValueError):
        ak.operations.firsts(cuda_array, axis=-4)


def test_0198_tutorial_documentation_1_firsts():
    array = ak.Array([1.1, 2.2, None, 3.3, None, None, 4.4, 5.5])
    cuda_array = ak.to_backend(array, "cuda")

    assert to_list(
        ak.operations.firsts(
            ak.operations.singletons(cuda_array),
            axis=1,
        )
    ) == [1.1, 2.2, None, 3.3, None, None, 4.4, 5.5]

    array = ak.Array([[1.1, 2.2, None], [3.3, None], [None], [4.4, 5.5]])
    cuda_array = ak.to_backend(array, "cuda")

    assert to_list(
        ak.operations.firsts(
            ak.operations.singletons(cuda_array, axis=1),
            axis=2,
        )
    ) == [[1.1, 2.2, None], [3.3, None], [None], [4.4, 5.5]]

    array = ak.Array([[[1.1, 2.2, None]], [[3.3, None]], [[None]], [[4.4, 5.5]]])
    cuda_array = ak.to_backend(array, "cuda")

    assert to_list(
        ak.operations.firsts(
            ak.operations.singletons(
                cuda_array,
                axis=2,
            ),
            axis=3,
        )
    ) == [[[1.1, 2.2, None]], [[3.3, None]], [[None]], [[4.4, 5.5]]]


def test_0198_tutorial_documentation_1_singletons():
    array = ak.Array([1.1, 2.2, None, 3.3, None, None, 4.4, 5.5])
    cuda_array = ak.to_backend(array, "cuda")

    assert to_list(ak.operations.singletons(cuda_array)) == [
        [1.1],
        [2.2],
        [],
        [3.3],
        [],
        [],
        [4.4],
        [5.5],
    ]
    assert to_list(ak.operations.singletons(cuda_array)) == [
        [1.1],
        [2.2],
        [],
        [3.3],
        [],
        [],
        [4.4],
        [5.5],
    ]

    array = ak.Array([[1.1, 2.2, None], [3.3, None], [None], [4.4, 5.5]])
    cuda_array = ak.to_backend(array, "cuda")
    assert to_list(ak.operations.singletons(cuda_array, axis=1)) == [
        [[1.1], [2.2], []],
        [[3.3], []],
        [[]],
        [[4.4], [5.5]],
    ]

    array = ak.Array([[[1.1, 2.2, None]], [[3.3, None]], [[None]], [[4.4, 5.5]]])
    cuda_array = ak.to_backend(array, "cuda")
    assert to_list(
        ak.operations.singletons(
            cuda_array,
            axis=2,
        )
    ) == [[[[1.1], [2.2], []]], [[[3.3], []]], [[[]]], [[[4.4], [5.5]]]]


def test_0198_tutorial_documentation_1_allow_missing():
    array = ak.Array([1.1, 2.2, None, 3.3, None, None, 4.4, 5.5])
    cuda_array = ak.to_backend(array, "cuda")

    ak.operations.to_numpy(cuda_array)
    with pytest.raises(ValueError):
        ak.operations.to_numpy(cuda_array, allow_missing=False)


def test_0198_tutorial_documentation_1_flatten0():
    array = ak.Array([1.1, 2.2, None, 3.3, None, None, 4.4, 5.5])
    cuda_array = ak.to_backend(array, "cuda")

    assert to_list(ak.operations.flatten(cuda_array, axis=0)) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]

    content0 = ak.operations.from_iter(
        [1.1, 2.2, None, 3.3, None, None, 4.4, 5.5], highlevel=False
    )
    content1 = ak.operations.from_iter(
        ["one", None, "two", None, "three"], highlevel=False
    )
    array = ak.Array(
        ak.contents.UnionArray(
            ak.index.Index8(
                np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0], dtype=np.int8)
            ),
            ak.index.Index64(
                np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 4, 7], dtype=np.int64)
            ),
            [content0, content1],
        )
    )
    cuda_array = ak.to_backend(array, "cuda")

    assert to_list(cuda_array) == [
        1.1,
        "one",
        2.2,
        None,
        None,
        "two",
        3.3,
        None,
        None,
        None,
        4.4,
        "three",
        5.5,
    ]
    assert to_list(ak.operations.flatten(cuda_array, axis=0)) == [
        1.1,
        "one",
        2.2,
        "two",
        3.3,
        4.4,
        "three",
        5.5,
    ]


def test_0193_is_none_axis_parameter():
    array = ak.Array([1, 2, 3, None, 5])
    cuda_array = ak.to_backend(array, "cuda")

    assert ak.operations.is_none(cuda_array).to_list() == [
        False,
        False,
        False,
        True,
        False,
    ]

    array = ak.Array([[1, 2, 3], [], [None, 5]])
    cuda_array = ak.to_backend(array, "cuda")

    assert ak.operations.is_none(cuda_array).to_list() == [
        False,
        False,
        False,
    ]
    assert ak.operations.is_none(cuda_array, axis=1).to_list() == [
        [False, False, False],
        [],
        [True, False],
    ]

    array = ak.Array([[1, None, 2, 3], [], [None, 5]])
    cuda_array = ak.to_backend(array, "cuda")

    assert ak.operations.is_none(cuda_array, axis=1).to_list() == [
        [False, True, False, False],
        [],
        [True, False],
    ]
    assert ak.operations.is_none(cuda_array, axis=-1).to_list() == [
        [False, True, False, False],
        [],
        [True, False],
    ]
    assert ak.operations.is_none(cuda_array, axis=-2).to_list() == [
        False,
        False,
        False,
    ]
    with pytest.raises(ValueError):
        ak.operations.is_none(cuda_array, axis=-3)

    array = ak.Array([[1, None, 2, 3], None, [None, 5]])
    cuda_array = ak.to_backend(array, "cuda")

    assert ak.operations.is_none(cuda_array, axis=-2).to_list() == [False, True, False]

    one = ak.operations.from_iter([1, None, 3], highlevel=False)
    two = ak.operations.from_iter([[], [1], None, [3, 3, 3]], highlevel=False)
    tags = ak.index.Index8(np.array([0, 1, 1, 0, 0, 1, 1], dtype=np.int8))
    index = ak.index.Index64(np.array([0, 0, 1, 1, 2, 2, 3], dtype=np.int64))
    array = ak.Array(ak.contents.UnionArray(tags, index, [one, two]), check_valid=True)
    cuda_array = ak.to_backend(array, "cuda")

    assert ak.to_list(cuda_array) == [1, [], [1], None, 3, None, [3, 3, 3]]

    assert ak.to_list(ak.operations.is_none(cuda_array)) == [
        False,
        False,
        False,
        True,
        False,
        True,
        False,
    ]


def test_0150_flatten_axis_none_0866():
    a1 = ak.operations.zip(
        {"a": [[1], [], [2, 3]], "b": [[4], [], [5, 6]]}, with_name="a1"
    )
    a2 = ak.operations.zip(
        {"a": [[7, 8], [9], []], "b": [[10, 11], [12], []]}, with_name="a2"
    )
    cuda_a1 = ak.to_backend(a1, "cuda")
    cuda_a2 = ak.to_backend(a2, "cuda")

    condition = ak.Array([True, False, True])
    cuda_condition = ak.to_backend(condition, "cuda")
    union = ak.operations.where(cuda_condition, cuda_a1, cuda_a2)

    assert set(cp.asnumpy(ak.operations.flatten(union, axis=None))) == {
        1,
        2,
        3,
        4,
        5,
        6,
        9,
        12,
    }


def test_0150_flatten_0198():
    array = ak.highlevel.Array([1.1, 2.2, None, 3.3, None, None, 4.4, 5.5]).layout
    cuda_array = ak.to_backend(array, "cuda")

    assert ak.operations.to_list(ak.operations.flatten(cuda_array, axis=0)) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]

    content0 = ak.operations.from_iter(
        [1.1, 2.2, None, 3.3, None, None, 4.4, 5.5], highlevel=False
    )
    content1 = ak.operations.from_iter(
        ["one", None, "two", None, "three"], highlevel=False
    )
    array = ak.contents.UnionArray(
        ak.index.Index8(
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0], dtype=np.int8)
        ),
        ak.index.Index64(
            np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 4, 7], dtype=np.int64)
        ),
        [content0, content1],
    )
    cuda_array = ak.to_backend(array, "cuda")

    assert to_list(cuda_array) == [
        1.1,
        "one",
        2.2,
        None,
        None,
        "two",
        3.3,
        None,
        None,
        None,
        4.4,
        "three",
        5.5,
    ]
    assert ak.operations.to_list(ak.operations.flatten(cuda_array, axis=0)) == [
        1.1,
        "one",
        2.2,
        "two",
        3.3,
        4.4,
        "three",
        5.5,
    ]


def test_0150_flatten_0042():
    array = ak.highlevel.Array(
        [[1.1, 2.2, 3.3], [], [4.4, 5.5]], check_valid=True
    ).layout
    cuda_array = ak.to_backend(array, "cuda")

    assert ak.operations.to_list(ak.operations.flatten(cuda_array, axis=1)) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]


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


def test_2678_same_backend_where():
    result = ak.where(
        ak.to_backend([True, False, False], "cuda"),
        ak.to_backend([1, 2, 3], "cuda"),
        ak.to_backend([4, 5, 6], "cuda"),
    )

    assert ak.backend(result) == "cuda"


def test_2616_use_pyarrow_for_strings_to_categorical():
    assert (
        ak.str.to_categorical(["foo", "bar", "bar", "fee"]).layout.form
        == ak.str.to_categorical(
            ak.to_backend(["foo", "bar", "bar", "fee"], "cuda")
        ).layout.form
    )


def test_2426_is_equal_to_equal_union():
    union_1 = ak.from_iter([1, None, {"x": 2}], highlevel=False)
    union_2 = ak.from_iter([1, None, {"x": 2}], highlevel=False)

    cuda_union_1 = ak.to_backend(union_1, "cuda", highlevel=False)
    cuda_union_2 = ak.to_backend(union_2, "cuda", highlevel=False)

    assert cuda_union_1.is_equal_to(cuda_union_2)


def test_2426_is_equal_to_unequal_union():
    union_1 = ak.from_iter([1, None, {"x": 2}, 3], highlevel=False)
    union_2 = ak.from_iter([1, None, {"x": 2}, 2], highlevel=False)

    cuda_union_1 = ak.to_backend(union_1, "cuda", highlevel=False)
    cuda_union_2 = ak.to_backend(union_2, "cuda", highlevel=False)

    assert not cuda_union_1.is_equal_to(cuda_union_2)


# def test_RegularArray():
#     new = ak.contents.RegularArray(
#         ak.operations.from_numpy(np.arange(2 * 3 * 5).reshape(-1, 5)).layout,
#         3,
#     )

#     # new = ak.to_backend(new, "cuda")

#     assert to_list(new[1, 1:]) == [[20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]
#     assert new.to_typetracer()[1, 1:].form == new[1, 1:].form

#     with pytest.raises(IndexError):
#         new[1, "hello"]

#     with pytest.raises(IndexError):
#         new[1, ["hello", "there"]]

#     assert to_list(new[1, np.newaxis, -2]) == [[20, 21, 22, 23, 24]]
#     assert to_list(new[1, np.newaxis, np.newaxis, -2]) == [[[20, 21, 22, 23, 24]]]
#     assert new.to_typetracer()[1, np.newaxis, -2].form == new[1, np.newaxis, -2].form

#     assert new.minmax_depth == (3, 3)

#     assert to_list(new[1, ..., -2]) == [18, 23, 28]
#     assert new.to_typetracer()[1, ..., -2].form == new[1, ..., -2].form

#     expectation = [
#         [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
#         [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
#     ]
#     assert to_list(new[[1, 0],]) == expectation
#     assert new.to_typetracer()[[1, 0],].form == new[[1, 0],].form
#     assert to_list(new[[1, 0]]) == expectation

#     assert to_list(new[1, [2, 0]]) == [[25, 26, 27, 28, 29], [15, 16, 17, 18, 19]]

#     array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
#     # array = ak.to_backend(array, "cuda", highlevel=False)

#     assert (
#         repr(array[[True, False, True]])
#         == "<Array [[1.1, 2.2, 3.3], [4.4, 5.5]] type='2 * var * float64'>"
#     )
#     assert (
#         repr(array[[True, False, True], 1]) != "<Array [2.2, 5.5] type='2 * float64'>"
#     )


# def test_union_simplification():
#     array = ak.Array(
#         ak.contents.UnionArray(
#             ak.index.Index8(np.arange(64, dtype=np.int8) % 2),
#             ak.index.Index64(np.arange(64, dtype=np.int64) // 2),
#             [
#                 ak.contents.RecordArray(
#                     [ak.contents.NumpyArray(np.arange(64, dtype=np.int64))], ["x"]
#                 ),
#                 ak.contents.RecordArray(
#                     [
#                         ak.contents.NumpyArray(np.arange(64, dtype=np.int64)),
#                         ak.contents.NumpyArray(np.arange(64, dtype=np.int8)),
#                     ],
#                     ["x", "y"],
#                 ),
#             ],
#         )
#     )

#     cuda_array = ak.to_backend(array, "cuda")

#     form, length, container = ak.to_buffers(array)
#     cuda_form, cuda_length, cuda_container = ak.to_buffers(cuda_array)

#     assert form.to_dict() == cuda_form.to_dict()

#     projected_form = {
#         "class": "UnionArray",
#         "tags": "i8",
#         "index": "i64",
#         "contents": [
#             {
#                 "class": "RecordArray",
#                 "fields": ["x"],
#                 "contents": [
#                     {
#                         "class": "NumpyArray",
#                         "primitive": "int64",
#                         "inner_shape": [],
#                         "parameters": {},
#                         "form_key": "node2",
#                     }
#                 ],
#                 "parameters": {},
#                 "form_key": "node1",
#             },
#             {
#                 "class": "RecordArray",
#                 "fields": ["x"],
#                 "contents": [
#                     {
#                         "class": "NumpyArray",
#                         "primitive": "int64",
#                         "inner_shape": [],
#                         "parameters": {},
#                         "form_key": "node4",
#                     }
#                 ],
#                 "parameters": {},
#                 "form_key": "node3",
#             },
#         ],
#         "parameters": {},
#         "form_key": "node0",
#     }
#     cuda_container.pop("node5-data")
#     projected = ak.from_buffers(
#         projected_form, cuda_length, cuda_container, allow_noncanonical_form=True
#     )
#     assert projected.layout.form.to_dict(verbose=False) == {
#         "class": "IndexedArray",
#         "index": "i64",
#         "content": {"class": "RecordArray", "fields": ["x"], "contents": ["int64"]},
#     }
#     assert ak.almost_equal(cuda_array[["x"]], projected)

# def test_return_scalar_firsts():
#     array = ak.Array([1, 2, 3, 4])
#     cuda_array = ak.to_backend(array, "cuda")

#     first = ak.firsts(cuda_array, axis=0)
#     assert isinstance(first, np.int64) and first == 1
# incompatible backends

# def test_return_scalar_firsts_non_scalar():
#     array = ak.Array([[1, 2, 3], [4]])
#     cuda_array = ak.to_backend(array, "cuda")

#     first = ak.firsts(cuda_array, axis=0)
#     assert first.to_list() == [1, 2, 3]
# incompatible backends

# def test_fillna_unionarray():
#     content1 = ak.operations.from_iter([[], [1.1], [2.2, 2.2]], highlevel=False)
#     content2 = ak.operations.from_iter([["two", "two"], ["one"], []], highlevel=False)
#     tags = ak.index.Index8(np.array([0, 1, 0, 1, 0, 1], dtype=np.int8))
#     index = ak.index.Index64(np.array([0, 0, 1, 1, 2, 2], dtype=np.int64))
#     array = ak.contents.UnionArray(tags, index, [content1, content2])
#     cuda_array = ak.to_backend(array, "cuda", highlevel=False)

#     padded_array =  ak._do.pad_none(array, 2, 1)
#     padded_cupy_array = ak._do.pad_none(cuda_array, 2, 1)

#     assert padded_array == padded_cupy_array

#     value = ak.contents.NumpyArray(np.array([777]))
#     assert ak._do.fill_none(padded_array, value) == ak._do.fill_none(padded_cupy_array, value, highlevel=False)

# no field - pad_none, is_equal_to, parameter, parameters, backend, project
# snapshot
# unsupported type - datetime


#  converts int to float too if other elements are float
