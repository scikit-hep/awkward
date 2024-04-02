from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

import awkward as ak
from awkward._behavior import behavior_of

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


def test_2616_use_pyarrow_for_strings_to_categorical():
    assert (
        ak.str.to_categorical(["foo", "bar", "bar", "fee"]).layout.form
        == ak.str.to_categorical(
            ak.to_backend(["foo", "bar", "bar", "fee"], "cuda")
        ).layout.form
    )


def test_0334_fully_broadcastable_where():
    one = ak.highlevel.Array([[0, 1, 2], [], [3, 4], [5], [6, 7, 8, 9]])
    two = ak.highlevel.Array(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]
    )
    condition = ak.highlevel.Array(
        [[False, True, False], [], [True, False], [True], [False, False, True, True]]
    )

    cuda_one = ak.to_backend(one, "cuda")
    cuda_two = ak.to_backend(two, "cuda")
    cuda_condition = ak.to_backend(condition, "cuda")

    assert ak.operations.where(cuda_condition, cuda_one, cuda_two).to_list() == [
        [0, 1, 2.2],
        [],
        [3, 4.4],
        [5],
        [6.6, 7.7, 8, 9],
    ]


def test_0334_fully_broadcastable_where_issue_334():
    a = ak.highlevel.Array([1, 2, 3, 4])
    b = ak.highlevel.Array([-1])
    c = ak.highlevel.Array([True, False, True, True])

    cuda_a = ak.to_backend(a, "cuda")
    cuda_b = ak.to_backend(b, "cuda")
    cuda_c = ak.to_backend(c, "cuda")

    assert ak.operations.where(cuda_c, cuda_a, cuda_b).to_list() == [1, -1, 3, 4]
    assert ak.operations.where(
        *ak.operations.broadcast_arrays(cuda_c, cuda_a, cuda_b)
    ).to_list() == [
        1,
        -1,
        3,
        4,
    ]
    assert ak.operations.where(cuda_c, cuda_a, -1).to_list() == [1, -1, 3, 4]
    assert ak.operations.where(
        *ak.operations.broadcast_arrays(cuda_c, cuda_a, -1)
    ).to_list() == [
        1,
        -1,
        3,
        4,
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


def test_2678_same_backend_where():
    result = ak.where(
        ak.to_backend([True, False, False], "cuda"),
        ak.to_backend([1, 2, 3], "cuda"),
        ak.to_backend([4, 5, 6], "cuda"),
    )

    assert ak.backend(result) == "cuda"


def test_0493_zeros_ones_full_like():
    array = ak.Array(
        [
            [{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}],
            [],
            [
                {"x": 3.3, "y": [1, 2, None, 3]},
                False,
                False,
                True,
                {"x": 4.4, "y": [1, 2, None, 3, 4]},
            ],
        ]
    )
    cuda_array = ak.to_backend(array, "cuda")

    assert ak.operations.full_like(cuda_array, 12.3).to_list() == [
        [{"x": 12.3, "y": []}, {"x": 12.3, "y": [12]}, {"x": 12.3, "y": [12, 12]}],
        [],
        [
            {"x": 12.3, "y": [12, 12, None, 12]},
            True,
            True,
            True,
            {"x": 12.3, "y": [12, 12, None, 12, 12]},
        ],
    ]

    assert ak.operations.zeros_like(cuda_array).to_list() == [
        [{"x": 0.0, "y": []}, {"x": 0.0, "y": [0]}, {"x": 0.0, "y": [0, 0]}],
        [],
        [
            {"x": 0.0, "y": [0, 0, None, 0]},
            False,
            False,
            False,
            {"x": 0.0, "y": [0, 0, None, 0, 0]},
        ],
    ]

    assert ak.operations.ones_like(cuda_array).to_list() == [
        [{"x": 1.0, "y": []}, {"x": 1.0, "y": [1]}, {"x": 1.0, "y": [1, 1]}],
        [],
        [
            {"x": 1.0, "y": [1, 1, None, 1]},
            True,
            True,
            True,
            {"x": 1.0, "y": [1, 1, None, 1, 1]},
        ],
    ]


def test_0496_provide_local_index():
    array = ak.highlevel.Array(
        [[[0.0, 1.1, 2.2], []], [[3.3, 4.4]], [], [[5.5], [], [6.6, 7.7, 8.8, 9.9]]]
    )
    cuda_array = ak.to_backend(array, "cuda")

    assert to_list(ak.operations.local_index(cuda_array, axis=0)) == [
        0,
        1,
        2,
        3,
    ]
    assert to_list(ak.operations.local_index(cuda_array, axis=1)) == [
        [0, 1],
        [0],
        [],
        [0, 1, 2],
    ]
    assert to_list(ak.operations.local_index(cuda_array, axis=2)) == [
        [[0, 1, 2], []],
        [[0, 1]],
        [],
        [[0], [], [0, 1, 2, 3]],
    ]
    assert to_list(ak.operations.local_index(cuda_array, axis=-1)) == [
        [[0, 1, 2], []],
        [[0, 1]],
        [],
        [[0], [], [0, 1, 2, 3]],
    ]
    assert to_list(ak.operations.local_index(cuda_array, axis=-2)) == [
        [0, 1],
        [0],
        [],
        [0, 1, 2],
    ]
    assert to_list(ak.operations.local_index(cuda_array, axis=-3)) == [
        0,
        1,
        2,
        3,
    ]

    assert to_list(
        ak.operations.zip(
            [
                ak.operations.local_index(cuda_array, axis=0),
                ak.operations.local_index(cuda_array, axis=1),
                ak.operations.local_index(cuda_array, axis=2),
            ]
        )
    ) == [
        [[(0, 0, 0), (0, 0, 1), (0, 0, 2)], []],
        [[(1, 0, 0), (1, 0, 1)]],
        [],
        [[(3, 0, 0)], [], [(3, 2, 0), (3, 2, 1), (3, 2, 2), (3, 2, 3)]],
    ]


def test_0504_block_ufuncs_for_strings():
    def _apply_ufunc(ufunc, method, inputs, kwargs):
        nextinputs = []
        for x in inputs:
            if (
                isinstance(x, ak.highlevel.Array)
                and x.layout.is_indexed
                and not x.layout.is_option
            ):
                nextinputs.append(
                    ak.highlevel.Array(x.layout.project(), behavior=behavior_of(x))
                )
            else:
                nextinputs.append(x)

        return getattr(ufunc, method)(*nextinputs, **kwargs)

    behavior = {}
    behavior[np.ufunc, "categorical"] = _apply_ufunc

    array = ak.highlevel.Array(
        ak.contents.IndexedArray(
            ak.index.Index64(np.array([0, 1, 2, 1, 3, 1, 4])),
            ak.contents.NumpyArray(np.array([321, 1.1, 123, 999, 2])),
            parameters={"__array__": "categorical"},
        ),
        behavior=behavior,
    )
    cuda_array = ak.to_backend(array, "cuda")

    assert to_list(cuda_array * 10) == [3210, 11, 1230, 11, 9990, 11, 20]

    cuda_array = ak.highlevel.Array(["HAL"])

    with pytest.raises(TypeError):
        cuda_array + 1


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
