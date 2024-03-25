from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

import awkward as ak

to_list = ak.operations.to_list

try:
    ak.numba.register_and_check()
except ImportError:
    pytest.skip(reason="too old Numba version", allow_module_level=True)


def test_1753_indexedarray_merge_kernel():
    x = ak.contents.IndexedArray(
        ak.index.Index64(np.array([0, 0, 1], dtype=np.int64)),
        ak.contents.NumpyArray(np.array([9, 6, 5], dtype=np.int16)),
        parameters={"money": "doesn't buy happiness"},
    )
    y = ak.contents.IndexedArray(
        ak.index.Index64(np.array([0, 1, 2, 4, 3], dtype=np.int64)),
        ak.contents.NumpyArray(np.array([9, 6, 5, 8, 2], dtype=np.int16)),
        parameters={"age": "number"},
    )

    cuda_x = ak.to_backend(x, "cuda", highlevel=False)
    cuda_y = ak.to_backend(y, "cuda", highlevel=False)

    # Test that we invoke the merge pathway
    z = cuda_x._reverse_merge(cuda_y)
    assert z.to_list() == [9, 6, 5, 2, 8, 9, 9, 6]
    assert z.parameters == {"money": "doesn't buy happiness", "age": "number"}


def test_2651_parameter_union():
    layout = ak.contents.IndexedArray(
        ak.index.Index64([0, 1, 2]),
        ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.uint32)),
        parameters={"foo": {"bar": "baz"}},
    )

    cuda_layout = ak.to_backend(layout, "cuda", highlevel=False)

    result = cuda_layout.project()
    assert result.is_equal_to(
        ak.contents.NumpyArray(
            cp.array([1, 2, 3], dtype=cp.uint32), parameters={"foo": {"bar": "baz"}}
        )
    )


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

#     assert ak.firsts(array, axis=0) == ak.firsts(cuda_array, axis=0)
#     first = ak.firsts(array, axis=0)
#     assert isinstance(first, np.int64) and first == 1


# def test_return_scalar_firsts_non_scalar():
#     array = ak.Array([[1, 2, 3], [4]])
#     cuda_array = ak.to_backend(array, "cuda")

#     first = ak.firsts(array, axis=0)
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

# no field - project, pad_none, is_equal_to, is_union, _reverse_merge, parameter, parameters, backend, project
# snapshot
# unsupported type - datetime
