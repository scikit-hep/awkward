from __future__ import annotations

import pytest

import awkward as ak

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
