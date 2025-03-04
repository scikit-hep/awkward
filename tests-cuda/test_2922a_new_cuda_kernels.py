# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
from numpy.exceptions import AxisError

import awkward as ak
from awkward.forms import (
    BitMaskedForm,
    IndexedForm,
    ListOffsetForm,
    NumpyForm,
)

to_list = ak.operations.to_list


@pytest.fixture(scope="function", autouse=True)
def cleanup_cuda():
    yield
    cp._default_memory_pool.free_all_blocks()
    cp.cuda.Device().synchronize()


def test_0184_concatenate_operation_records():
    one = ak.highlevel.Array([[1, 2, 3], [None, 4], None, [None, 5]]).layout
    two = ak.highlevel.Array([6, 7, 8]).layout
    three = ak.highlevel.Array([[6.6], [7.7, 8.8]]).layout
    # four = ak.highlevel.Array([[6.6], [7.7, 8.8], None, [9.9]]).layout
    cuda_one = ak.to_backend(one, "cuda")
    cuda_two = ak.to_backend(two, "cuda")
    cuda_three = ak.to_backend(three, "cuda")

    # cuda_four = ak.to_backend(four, "cuda")

    assert to_list(ak.operations.concatenate([one, two], 0)) == to_list(
        ak.operations.concatenate([cuda_one, cuda_two], 0)
    )

    with pytest.raises(ValueError):
        to_list(ak.operations.concatenate([one, three], 1))
        to_list(ak.operations.concatenate([cuda_one, cuda_three], 1))

    # assert to_list(ak.operations.concatenate([cuda_one, cuda_four], 1)) == [
    #     [1, 2, 3, 6.6],
    #     [None, 4, 7.7, 8.8],
    #     [],
    #     [None, 5, 9.9],
    # ]


def test_1904_drop_none_BitMaskedArray_NumpyArray():
    array = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
            np.packbits(
                np.array(
                    [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1],
                    np.uint8,
                )
            )
        ),
        ak.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=True,
        length=13,
        lsb_order=False,
    )
    cuda_array = ak.to_backend(array, "cuda")

    assert to_list(array) == to_list(cuda_array)
    assert (
        to_list(ak.drop_none(array))
        == to_list(array[~ak.is_none(array)])
        == to_list(ak.drop_none(cuda_array))
        == to_list(cuda_array[~ak.is_none(cuda_array)])
    )


def test_1904_drop_none_BitMaskedArray_RecordArray_NumpyArray():
    array = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
            np.packbits(
                np.array(
                    [
                        True,
                        True,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        True,
                        False,
                        True,
                        False,
                        True,
                    ]
                )
            )
        ),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array(
                        [
                            0.0,
                            1.0,
                            2.0,
                            3.0,
                            4.0,
                            5.0,
                            6.0,
                            7.0,
                            1.1,
                            2.2,
                            3.3,
                            4.4,
                            5.5,
                            6.6,
                        ]
                    )
                )
            ],
            ["nest"],
        ),
        valid_when=True,
        length=13,
        lsb_order=False,
    )

    cuda_array = ak.to_backend(array, "cuda")

    assert to_list(array) == to_list(cuda_array)
    assert (
        to_list(ak.drop_none(array))
        == to_list(array[~ak.is_none(array)])
        == to_list(ak.drop_none(cuda_array))
        == to_list(cuda_array[~ak.is_none(cuda_array)])
    )


def test_1904_drop_none_ByteMaskedArray_NumpyArray():
    array = ak.contents.bytemaskedarray.ByteMaskedArray(
        ak.index.Index(np.array([1, 0, 1, 0, 1], np.int8)),
        ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        valid_when=True,
    )
    cuda_array = ak.to_backend(array, "cuda")

    assert to_list(array) == to_list(cuda_array)
    assert (
        to_list(ak.drop_none(array))
        == to_list(array[~ak.is_none(array)])
        == to_list(ak.drop_none(cuda_array))
        == to_list(cuda_array[~ak.is_none(cuda_array)])
    )


def test_1904_drop_none_ByteMaskedArray_RecordArray_NumpyArray():
    array = ak.contents.bytemaskedarray.ByteMaskedArray(
        ak.index.Index(np.array([1, 0, 1, 0, 1], np.int8)),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
        valid_when=True,
    )
    assert (
        to_list(ak.drop_none(array))
        == to_list(array[~ak.is_none(array, axis=0)])
        == [{"nest": 1.1}, {"nest": 3.3}, {"nest": 5.5}]
    )
    cuda_array = ak.to_backend(array, "cuda")

    assert to_list(array) == to_list(cuda_array)
    assert (
        to_list(ak.drop_none(array))
        == to_list(array[~ak.is_none(array, axis=0)])
        == to_list(ak.drop_none(cuda_array))
        == to_list(cuda_array[~ak.is_none(cuda_array, axis=0)])
    )


def test_1904_drop_none_IndexedOptionArray_NumpyArray_outoforder():
    index = ak.index.Index64(np.asarray([0, -1, 1, 5, 4, 2, 5]))
    content = ak.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
    )
    indexoptionarray = ak.contents.IndexedOptionArray(index, content)

    cuda_indexoptionarray = ak.to_backend(indexoptionarray, "cuda")

    assert to_list(indexoptionarray) == [0.0, None, 1.1, 5.5, 4.4, 2.2, 5.5]
    assert to_list(ak.drop_none(indexoptionarray)) == [0.0, 1.1, 5.5, 4.4, 2.2, 5.5]

    assert to_list(indexoptionarray) == to_list(cuda_indexoptionarray)
    assert to_list(ak.drop_none(indexoptionarray)) == to_list(
        ak.drop_none(cuda_indexoptionarray)
    )


def test_1904_drop_none_ListArray_IndexedOptionArray_RecordArray_NumpyArray():
    index = ak.index.Index64(np.asarray([0, -1, 1, -1, 4, -1, 5]))
    content = ak.contents.recordarray.RecordArray(
        [
            ak.contents.numpyarray.NumpyArray(
                np.array([6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7])
            )
        ],
        ["nest"],
    )
    indexoptionarray = ak.contents.IndexedOptionArray(index, content)
    array = ak.contents.listarray.ListArray(
        ak.index.Index(np.array([4, 100, 1], np.int64)),
        ak.index.Index(np.array([7, 100, 3, 200], np.int64)),
        indexoptionarray,
    )

    cuda_array = ak.to_backend(array, "cuda")

    assert to_list(array) == to_list(cuda_array)
    assert to_list(ak.drop_none(array, axis=0)) == to_list(
        ak.drop_none(cuda_array, axis=0)
    )


def test_1904_drop_none_all_axes():
    array = ak.Array(
        [
            None,
            [None, {"x": [1], "y": [[2]]}],
            [{"x": [3], "y": [None]}, {"x": [None], "y": [[None]]}],
        ]
    )

    cuda_array = ak.to_backend(array, "cuda")

    assert array.tolist() == cuda_array.tolist()

    assert to_list(ak.drop_none(array, axis=2)) == to_list(
        ak.drop_none(cuda_array, axis=2)
    )

    assert ak.is_none(array, axis=0).tolist() == ak.is_none(cuda_array, axis=0).tolist()
    assert ak.drop_none(array, axis=0).tolist() == ak.drop_none(array, axis=0).tolist()

    assert ak.is_none(array, axis=1).tolist() == ak.is_none(cuda_array, axis=1).tolist()
    assert ak.drop_none(array, axis=1).tolist() == ak.drop_none(array, axis=1).tolist()

    assert ak.is_none(array, axis=2).tolist() == ak.is_none(cuda_array, axis=2).tolist()
    assert ak.drop_none(array, axis=2).tolist() == ak.drop_none(array, axis=2).tolist()

    assert (
        ak.is_none(array, axis=-1).tolist() == ak.is_none(cuda_array, axis=-1).tolist()
    )
    assert (
        ak.drop_none(array, axis=-1).tolist() == ak.drop_none(array, axis=-1).tolist()
    )

    assert (
        ak.is_none(array, axis=-2).tolist() == ak.is_none(cuda_array, axis=-2).tolist()
    )
    with pytest.raises(AxisError):
        ak.drop_none(array, axis=-2).tolist()
        ak.drop_none(cuda_array, axis=-2).tolist()

    array2 = ak.Array(
        [
            None,
            [None, {"x": [1], "y": [[2]]}],
            [{"x": None, "y": [None]}, {"x": [None], "y": [[None]]}],
        ]
    )

    cuda_array2 = ak.to_backend(array2, "cuda")

    assert array2.tolist() == cuda_array2.tolist()

    assert (
        ak.is_none(array2, axis=-2).tolist()
        == ak.is_none(cuda_array2, axis=-2).tolist()
    )
    with pytest.raises(AxisError):
        ak.drop_none(array2, axis=-2).tolist()
        ak.drop_none(cuda_array2, axis=-2).tolist()


def test_1914_improved_axis_to_posaxis_is_none():
    array = ak.Array(
        [
            None,
            [None],
            [{"x": None, "y": None}],
            [{"x": [None], "y": [None]}],
            [{"x": [1], "y": [[None]]}],
            [{"x": [2], "y": [[1, 2, 3]]}],
        ]
    )

    cuda_array = ak.to_backend(array, "cuda")

    assert ak.is_none(array, axis=0).tolist() == ak.is_none(cuda_array, axis=0).tolist()

    assert ak.is_none(array, axis=1).tolist() == ak.is_none(cuda_array, axis=1).tolist()

    assert ak.is_none(array, axis=2).tolist() == ak.is_none(cuda_array, axis=2).tolist()

    with pytest.raises(AxisError):
        ak.is_none(array, axis=3)
        ak.is_none(cuda_array, axis=3)

    assert (
        ak.is_none(array, axis=-1).tolist() == ak.is_none(cuda_array, axis=-1).tolist()
    )

    assert (
        ak.is_none(array, axis=-2).tolist() == ak.is_none(cuda_array, axis=-2).tolist()
    )

    with pytest.raises(AxisError):
        ak.is_none(array, axis=-3)
        ak.is_none(cuda_array, axis=-3)


def test_1914_improved_axis_to_posaxis_singletons():
    array = ak.Array(
        [
            None,
            [None],
            [{"x": None, "y": None}],
            [{"x": [None], "y": [None]}],
            [{"x": [1], "y": [[None]]}],
            [{"x": [2], "y": [[1, 2, 3]]}],
        ]
    )

    cuda_array = ak.to_backend(array, "cuda")

    assert (
        ak.singletons(array, axis=0).tolist()
        == ak.singletons(cuda_array, axis=0).tolist()
    )

    assert (
        ak.singletons(array, axis=1).tolist()
        == ak.singletons(cuda_array, axis=1).tolist()
    )

    assert (
        ak.singletons(array, axis=2).tolist()
        == ak.singletons(cuda_array, axis=2).tolist()
    )

    with pytest.raises(AxisError):
        ak.singletons(array, axis=3)
        ak.singletons(cuda_array, axis=3)

    assert (
        ak.singletons(array, axis=-1).tolist()
        == ak.singletons(cuda_array, axis=-1).tolist()
    )

    assert (
        ak.singletons(array, axis=-2).tolist()
        == ak.singletons(cuda_array, axis=-2).tolist()
    )

    with pytest.raises(AxisError):
        ak.singletons(array, axis=-3)
        ak.singletons(cuda_array, axis=-3)


def test_2889_chunked_array_strings():
    pa = pytest.importorskip("pyarrow")

    array = pa.chunked_array([["foo", "bar"], ["blah", "bleh"]])
    ak_array = ak.from_arrow(array)

    cuda_array = ak.to_backend(ak_array, "cuda")

    assert ak_array.type == cuda_array.type
    assert cuda_array.type == ak.types.ArrayType(
        ak.types.ListType(
            ak.types.NumpyType("uint8", parameters={"__array__": "char"}),
            parameters={"__array__": "string"},
        ),
        4,
    )


def test_2889_chunked_array_strings_option():
    pa = pytest.importorskip("pyarrow")

    array = pa.chunked_array([["foo", "bar"], ["blah", "bleh", None]])
    ak_array = ak.from_arrow(array)

    cuda_array = ak.to_backend(ak_array, "cuda")

    assert ak_array.type == cuda_array.type
    assert cuda_array.type == ak.types.ArrayType(
        ak.types.OptionType(
            ak.types.ListType(
                ak.types.NumpyType("uint8", parameters={"__array__": "char"}),
                parameters={"__array__": "string"},
            )
        ),
        5,
    )


def test_2889_chunked_array_numbers_option():
    pa = pytest.importorskip("pyarrow")

    array = pa.chunked_array([[1, 2, 3], [4, 5, None]])
    ak_array = ak.from_arrow(array)

    cuda_array = ak.to_backend(ak_array, "cuda")

    assert ak_array.type == cuda_array.type
    assert cuda_array.type == ak.types.ArrayType(
        ak.types.OptionType(ak.types.NumpyType("int64")), 6
    )


def test_2660_expected_container_keys_from_form_UnionArray_NumpyArray():
    # 100 is inaccessible in index
    # 1.1 is inaccessible in contents[1]
    array = ak.contents.unionarray.UnionArray.simplified(
        ak.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], dtype=np.int8)),
        ak.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100])),
        [
            ak.contents.numpyarray.NumpyArray(np.array([1, 2, 3])),
            ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
    )

    cuda_array = ak.to_backend(array, "cuda")

    form, length, container = ak.to_buffers(cuda_array)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype


def test_2660_expected_container_keys_from_form_UnionArray_RecordArray_NumpyArray():
    # 100 is inaccessible in index
    # 1.1 is inaccessible in contents[1]
    array = ak.contents.unionarray.UnionArray.simplified(
        ak.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], dtype=np.int8)),
        ak.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100])),
        [
            ak.contents.recordarray.RecordArray(
                [ak.contents.numpyarray.NumpyArray(np.array([1, 2, 3]))], ["nest"]
            ),
            ak.contents.recordarray.RecordArray(
                [
                    ak.contents.numpyarray.NumpyArray(
                        np.array([1.1, 2.2, 3.3, 4.4, 5.5])
                    )
                ],
                ["nest"],
            ),
        ],
    )

    cuda_array = ak.to_backend(array, "cuda")

    form, length, container = ak.to_buffers(cuda_array)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype


def test_2623_optiontype_outside_record_strings_ByteMaskedArray():
    record = ak.zip(
        {
            "x": ak.mask(["foo", "bar", "world"], [True, True, False]),
            "y": ak.mask(["do", "re", "mi"], [False, True, True]),
        },
        optiontype_outside_record=True,
    )

    cuda_record = ak.to_backend(record, "cuda")

    assert cuda_record.to_list() == [None, {"x": "bar", "y": "re"}, None]
    assert cuda_record.type == ak.types.ArrayType(
        ak.types.OptionType(
            ak.types.RecordType(
                [
                    ak.types.ListType(
                        ak.types.NumpyType("uint8", parameters={"__array__": "char"}),
                        parameters={"__array__": "string"},
                    ),
                    ak.types.ListType(
                        ak.types.NumpyType("uint8", parameters={"__array__": "char"}),
                        parameters={"__array__": "string"},
                    ),
                ],
                ["x", "y"],
            )
        ),
        3,
        None,
    )


def test_2623_optiontype_outside_record_strings_IndexedOptionArray():
    record = ak.zip(
        {"x": ["foo", "bar", None], "y": [None, "re", "mi"]},
        optiontype_outside_record=True,
    )

    cuda_record = ak.to_backend(record, "cuda")

    assert cuda_record.to_list() == [None, {"x": "bar", "y": "re"}, None]
    assert cuda_record.type == ak.types.ArrayType(
        ak.types.OptionType(
            ak.types.RecordType(
                [
                    ak.types.ListType(
                        ak.types.NumpyType("uint8", parameters={"__array__": "char"}),
                        parameters={"__array__": "string"},
                    ),
                    ak.types.ListType(
                        ak.types.NumpyType("uint8", parameters={"__array__": "char"}),
                        parameters={"__array__": "string"},
                    ),
                ],
                ["x", "y"],
            )
        ),
        3,
        None,
    )


def test_2564_string_broadcast_regular_string_mixed_invalid():
    strings = ak.to_regular([["abc", "efg"]], axis=2)
    numbers = ak.to_regular([[[1, 2, 3], [3, 4, 3]]], axis=2)

    cuda_strings = ak.to_backend(strings, "cuda")
    cuda_numbers = ak.to_backend(numbers, "cuda")

    with pytest.raises(
        ValueError,
        match=r"cannot broadcast RegularArray of length 2 with NumpyArray of length 6",
    ):
        ak.broadcast_arrays(cuda_strings, cuda_numbers, right_broadcast=False)


def test_2564_string_broadcast_regular_string_mixed_valid():
    strings = ak.to_regular([["abc", "efg"]], axis=2)
    numbers = ak.to_regular([[[1], [3]]], axis=2)

    cuda_strings = ak.to_backend(strings, "cuda")
    cuda_numbers = ak.to_backend(numbers, "cuda")

    assert to_list(
        ak.broadcast_arrays(strings, numbers, right_broadcast=False)
    ) == to_list(ak.broadcast_arrays(cuda_strings, cuda_numbers, right_broadcast=False))

    x, y = ak.broadcast_arrays(strings, numbers, right_broadcast=False)
    cuda_x, cuda_y = ak.broadcast_arrays(
        cuda_strings, cuda_numbers, right_broadcast=False
    )
    assert x.tolist() == cuda_x.tolist() == [[["abc"], ["efg"]]]
    assert y.tolist() == cuda_y.tolist() == [[[1], [3]]]


def test_2564_string_broadcast_regular_string_mixed_below():
    strings = ak.to_regular([["abc", "efg"]], axis=2)
    numbers = ak.to_regular([[1, 6], [3, 7]], axis=1)

    cuda_strings = ak.to_backend(strings, "cuda")
    cuda_numbers = ak.to_backend(numbers, "cuda")

    assert to_list(
        ak.broadcast_arrays(strings, numbers, right_broadcast=False)
    ) == to_list(ak.broadcast_arrays(cuda_strings, cuda_numbers, right_broadcast=False))

    x, y = ak.broadcast_arrays(strings, numbers, right_broadcast=False)
    cuda_x, cuda_y = ak.broadcast_arrays(
        cuda_strings, cuda_numbers, right_broadcast=False
    )
    assert x.tolist() == cuda_x.tolist() == [["abc", "efg"], ["abc", "efg"]]
    assert y.tolist() == cuda_y.tolist() == [[1, 6], [3, 7]]


def test_2564_string_broadcast_regular_string_string_valid():
    strings = ak.to_regular([["abc", "efg"]], axis=2)
    numbers = ak.to_regular([[["ab"], ["bc", "de"]]], axis=3)

    cuda_strings = ak.to_backend(strings, "cuda")
    cuda_numbers = ak.to_backend(numbers, "cuda")

    assert to_list(
        ak.broadcast_arrays(strings, numbers, right_broadcast=False)
    ) == to_list(ak.broadcast_arrays(cuda_strings, cuda_numbers, right_broadcast=False))

    x, y = ak.broadcast_arrays(strings, numbers, right_broadcast=False)
    cuda_x, cuda_y = ak.broadcast_arrays(
        cuda_strings, cuda_numbers, right_broadcast=False
    )
    assert x.tolist() == cuda_x.tolist() == [[["abc"], ["efg", "efg"]]]
    assert y.tolist() == cuda_y.tolist() == [[["ab"], ["bc", "de"]]]


def test_2549_list_nominal_type_string_class():
    class ReversibleArray(ak.Array):
        def reversed(self):
            return self[..., ::-1]

    ak.behavior["reversible-string"] = ReversibleArray

    strings = ak.with_parameter(["hi", "book", "cats"], "__list__", "reversible-string")

    cuda_strings = ak.to_backend(strings, "cuda")

    assert isinstance(cuda_strings, ReversibleArray)
    assert cuda_strings.to_list() == ["hi", "book", "cats"]
    assert cuda_strings.reversed().to_list() == ["cats", "book", "hi"]


def test_2425_forms_from_type_from_iter():
    # We define `from_type` to match ArrayBuilder where possible. We can't
    # include options inside unions, though, because ArrayBuilder creates `UnmaskedArray`
    # nodes for the non-indexed option
    array = ak.to_packed(
        ak.from_iter([1, 2, "hi", [3, 4, {"x": 4}], {"y": [None, 2]}])[:0]
    )

    cuda_array = ak.to_backend(array, "cuda")

    assert ak.forms.from_type(array.type.content) == ak.forms.from_type(
        cuda_array.type.content
    )
    form_from_type = ak.forms.from_type(cuda_array.type.content)
    assert form_from_type == cuda_array.layout.form


def test_2425_forms_from_type_regular():
    array = ak.to_regular([[1, 2, 3]])[:0]

    cuda_array = ak.to_backend(array, "cuda")

    assert ak.forms.from_type(array.type.content) == ak.forms.from_type(
        cuda_array.type.content
    )
    form_from_type = ak.forms.from_type(cuda_array.type.content)
    assert form_from_type == cuda_array.layout.form


def test_2425_forms_from_type_categorical():
    pytest.importorskip("pyarrow")

    array = ak.str.to_categorical(["do", "re", "mi", "fa", "so"])

    cuda_array = ak.to_backend(array, "cuda")

    assert ak.forms.from_type(array.type.content) == ak.forms.from_type(
        cuda_array.type.content
    )
    form_from_type = ak.forms.from_type(cuda_array.type.content)
    assert form_from_type == cuda_array.layout.form


def test_2425_forms_from_type_categorical_option():
    pytest.importorskip("pyarrow")

    array = ak.str.to_categorical(["do", "re", "mi", "fa", "so", None])

    cuda_array = ak.to_backend(array, "cuda")

    assert ak.forms.from_type(array.type.content) == ak.forms.from_type(
        cuda_array.type.content
    )
    form_from_type = ak.forms.from_type(cuda_array.type.content)
    assert form_from_type == cuda_array.layout.form


def test_2417_bytemasked_singletons():
    array = ak.Array(
        ak.contents.ByteMaskedArray(
            mask=ak.index.Index8([1, 1, 0, 0]),
            content=ak.contents.NumpyArray([3, 4, 2, 5]),
            valid_when=True,
        )
    )

    cuda_array = ak.to_backend(array, "cuda")

    assert (
        ak.singletons(array, highlevel=False).form
        == ak.singletons(cuda_array, highlevel=False).form
    )

    assert (
        ak.singletons(cuda_array, highlevel=False).form
        == ak.singletons(
            cuda_array.layout.to_typetracer(forget_length=True), highlevel=False
        ).form
    )


def test_2411_cartesian_axis_validation_simple():
    left = ak.Array([1, 2, 3])
    right = ak.Array([["lambda", "sigma", "eta", "phi"], ["delta"]])

    cuda_left = ak.to_backend(left, "cuda")
    cuda_right = ak.to_backend(right, "cuda")

    pair = ak.cartesian([cuda_left, cuda_right], axis=0)
    assert pair.ndim == 1
    assert pair.tolist() == [
        (1, ["lambda", "sigma", "eta", "phi"]),
        (1, ["delta"]),
        (2, ["lambda", "sigma", "eta", "phi"]),
        (2, ["delta"]),
        (3, ["lambda", "sigma", "eta", "phi"]),
        (3, ["delta"]),
    ]


def test_2410_string_broadcast_deep_string_string():
    a = ak.Array([["x", "yz"], ["hello", "world", "foo", "bar"]])

    b = ak.Array(["x", "y"])
    cuda_a = ak.to_backend(a, "cuda")
    cuda_b = ak.to_backend(b, "cuda")

    left, right = ak.broadcast_arrays(cuda_a, cuda_b)
    assert right.to_list() == [["x", "x"], ["y", "y", "y", "y"]]


def test_2410_string_broadcast_deep_numbers_string():
    a = ak.Array([[1, 2], [3, 4, 5, 6]])

    b = ak.Array(["x", "y"])
    cuda_a = ak.to_backend(a, "cuda")
    cuda_b = ak.to_backend(b, "cuda")

    left, right = ak.broadcast_arrays(cuda_a, cuda_b)
    assert right.to_list() == [["x", "x"], ["y", "y", "y", "y"]]


def test_2410_string_broadcast_same_depth():
    a = ak.Array(["z"])

    b = ak.Array(["x", "y"])
    cuda_a = ak.to_backend(a, "cuda")
    cuda_b = ak.to_backend(b, "cuda")

    left, right = ak.broadcast_arrays(cuda_a, cuda_b)
    assert left.to_list() == ["z", "z"]
    assert right.to_list() == ["x", "y"]


def test_2385_with_field_empty_record_union_partial_record():
    no_fields = ak.Array([{}, []])

    cuda_no_fields = ak.to_backend(no_fields, "cuda")

    with pytest.raises(ValueError, match="cannot add a new field"):
        cuda_no_fields["new_field"] = ak.to_backend(ak.Array([1, 2, 3, 4, 5]), "cuda")


def test_2385_with_field_empty_record_union_record():
    no_fields = ak.Array([{"x": 1}, {"y": 2}, {}, {}, {}])

    cuda_no_fields = ak.to_backend(no_fields, "cuda")

    cuda_no_fields["new_field"] = ak.to_backend(ak.Array([1, 2, 3, 4, 5]), "cuda")
    assert cuda_no_fields.to_list() == [
        {"new_field": 1, "x": 1, "y": None},
        {"new_field": 2, "x": None, "y": 2},
        {"new_field": 3, "x": None, "y": None},
        {"new_field": 4, "x": None, "y": None},
        {"new_field": 5, "x": None, "y": None},
    ]


def test_2240_merge_union_parameters():
    one = ak.with_parameter([1, 2, [], [3, 4]], "one", "one")
    two = ak.with_parameter([100, 200, 300], "two", "two")
    three = ak.with_parameter([{"x": 1}, {"x": 2}, 5, 6, 7], "two", "two")

    cuda_one = ak.to_backend(one, "cuda")
    cuda_two = ak.to_backend(two, "cuda")
    cuda_three = ak.to_backend(three, "cuda")

    # No parameter unions should occur here
    result = ak.concatenate((cuda_two, cuda_one, cuda_three))
    assert ak.parameters(result) == {}


def test_2240_simplify_merge_as_union_many():
    result = ak.concatenate(
        [
            ak.to_backend(ak.Array(x), "cuda")
            for x in [[{"a": 3}], [{"c": 3}], [{"d": 3}], [{"e": 3}]]
        ]
    )
    assert result.tolist() == [{"a": 3}, {"c": 3}, {"d": 3}, {"e": 3}]


def test_2185_merge_option_of_records():
    a = ak.Array([None, {"a": 1, "b": 2}])
    cuda_a = ak.to_backend(a, "cuda")

    assert str(cuda_a.type) == "2 * ?{a: int64, b: int64}"

    b = ak.merge_option_of_records(cuda_a)

    assert b.tolist() == [{"a": None, "b": None}, {"a": 1, "b": 2}]

    assert str(b.type) == "2 * {a: ?int64, b: ?int64}"


def test_2185_merge_option_of_records_2():
    a = ak.Array([None, {"a": 1, "b": 2}, {"a": None, "b": None}])
    cuda_a = ak.to_backend(a, "cuda")

    assert str(a.type) == "3 * ?{a: ?int64, b: ?int64}"

    b = ak.merge_option_of_records(cuda_a)

    assert b.tolist() == [
        {"a": None, "b": None},
        {"a": 1, "b": 2},
        {"a": None, "b": None},
    ]

    assert str(b.type) == "3 * {a: ?int64, b: ?int64}"


def test_2185_merge_option_of_records_3():
    a = ak.Array([[[[None, {"a": 1, "b": 2}]]]])
    cuda_a = ak.to_backend(a, "cuda")

    assert str(a.type) == "1 * var * var * var * ?{a: int64, b: int64}"

    b = ak.merge_option_of_records(cuda_a, axis=-1)

    assert b.tolist() == [[[[{"a": None, "b": None}, {"a": 1, "b": 2}]]]]

    assert str(b.type) == "1 * var * var * var * {a: ?int64, b: ?int64}"


def test_2108_fill_none_indexed():
    layout = ak.contents.IndexedArray(
        ak.index.Index64(np.arange(10)),
        ak.contents.RecordArray(
            [
                ak.contents.IndexedOptionArray(
                    ak.index.Index64(
                        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, -1], dtype=np.int64)
                    ),
                    ak.contents.NumpyArray(np.arange(10, dtype=np.uint8)),
                )
            ],
            ["x"],
        ),
    )

    cuda_layout = ak.to_backend(layout, "cuda")

    assert ak.almost_equal(
        ak.fill_none(cuda_layout, 9, axis=0),
        ak.zip(
            {
                "x": cp.arange(
                    10,
                    # Default arraybuilder type
                    dtype=cp.int64,
                )
            }
        ),
    )


def test_2096_ak_scalar_type_array():
    array = ak.Array(["this", {"x": ["is", 1, 2, None]}])

    cuda_array = ak.to_backend(array, "cuda")

    assert ak.type(cuda_array) == cuda_array.type
    assert isinstance(cuda_array.type, ak.types.ArrayType)


def test_2096_ak_scalar_type_record():
    record = ak.Record({"y": ["this", {"x": ["is", 1, 2, None]}]})
    cuda_record = ak.to_backend(record, "cuda")

    assert ak.type(cuda_record) == cuda_record.type
    assert isinstance(cuda_record.type, ak.types.ScalarType)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_2082_broadcast_zero_size(axis):
    this_slice = (slice(None),) * axis + (slice(0, 0),) + (...,)
    that_slice = (slice(None),) * axis + (slice(0, 1),) + (...,)

    x = np.arange(3 * 5 * 7).reshape(3, 5, 7)
    cuda_x = ak.to_backend(x, "cuda")

    result = np.broadcast_arrays(cuda_x[this_slice], cuda_x[that_slice])

    y = ak.from_numpy(x)
    cuda_y = ak.to_backend(y, "cuda")

    result_ak = ak.broadcast_arrays(cuda_y[this_slice], cuda_y[that_slice])

    assert ak.almost_equal(result, result_ak)


def test_2078_array_function_wrap():
    left = ak.Array([1, 2, 3])
    right = ak.Array([[1, 2], [4, 5, 6], [None]])

    cuda_left = ak.to_backend(left, "cuda")
    cuda_right = ak.to_backend(right, "cuda")

    result = np.broadcast_arrays(cuda_left, cuda_right)
    assert isinstance(result, list)
    assert isinstance(result[0], ak.Array)
    assert isinstance(result[1], ak.Array)
    assert result[0].to_list() == [[1, 1], [2, 2, 2], [None]]
    assert result[1].to_list() == [[1, 2], [4, 5, 6], [None]]


def test_2071_unflatten_non_packed_indexed_counts():
    counts = ak.contents.IndexedArray(
        ak.index.Index64(np.arange(3)),
        ak.contents.NumpyArray(np.array([3, 0, 2], dtype=np.int64)),
    )

    cuda_counts = ak.to_backend(counts, "cuda")

    cuda_arr = ak.to_backend([[1.1, 2.2, 3.3], [], [4.4, 5.5]], "cuda")

    assert ak.almost_equal(
        ak.unflatten(cp.array([1.1, 2.2, 3.3, 4.4, 5.5]), cuda_counts),
        cuda_arr,
    )


def test_2071_unflatten_non_packed_counts_indexed_layout():
    layout = ak.contents.IndexedArray(
        ak.index.Index64(np.arange(5)),
        ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64)),
    )

    cuda_layout = ak.to_backend(layout, "cuda")

    cuda_arr = ak.to_backend([[1.1, 2.2, 3.3], [], [4.4, 5.5]], "cuda")

    assert ak.almost_equal(
        ak.unflatten(cuda_layout, cp.array([3, 0, 2])),
        cuda_arr,
    )


def test_2067_to_buffers_byteorder_default():
    array = ak.Array([[[1, 2, 3], [4, 5], None, "hi"]])

    cuda_array = ak.to_backend(array, "cuda")

    _, _, container_little = ak.to_buffers(cuda_array, byteorder="<")
    _, _, container_default = ak.to_buffers(cuda_array)

    for name, buffer in container_little.items():
        assert buffer.tobytes() == container_default[name].tobytes()


def test_2064_fill_none_record_axis_none():
    record = ak.zip({"x": [1, None], "y": [2, 3]})

    cuda_record = ak.to_backend(record, "cuda")

    assert ak.fill_none(cuda_record, 0, axis=None).to_list() == [
        {"x": 1, "y": 2},
        {"x": 0, "y": 3},
    ]


def test_2064_fill_none_record_axis_last():
    record = ak.zip({"x": [1, None], "y": [2, 3]})

    cuda_record = ak.to_backend(record, "cuda")

    assert ak.fill_none(cuda_record, 0, axis=-1).to_list() == [
        {"x": 1, "y": 2},
        {"x": 0, "y": 3},
    ]


def test_2064_fill_none_record_option_outside_record():
    record = ak.zip({"x": [1, 4], "y": [2, 3]})
    record = record.mask[[True, False]]

    cuda_record = ak.to_backend(record, "cuda")

    assert ak.fill_none(cuda_record, 0, axis=-1).to_list() == [{"x": 1, "y": 2}, 0]


def test_2058_merge_numpy_array():
    x = ak.from_numpy(
        np.arange(4 * 3, dtype=np.int64).reshape(4, 3), regulararray=False
    )
    y = ak.from_numpy(
        np.arange(4 * 2, dtype=np.int64).reshape(4, 2), regulararray=False
    )

    cuda_x = ak.to_backend(x, "cuda")
    cuda_y = ak.to_backend(y, "cuda")

    assert ak.concatenate((cuda_x, cuda_y)).type == ak.types.ArrayType(
        ak.types.ListType(ak.types.NumpyType("int64")), 8
    )


def test_2058_merge_numpy_array_regular():
    x = ak.from_numpy(np.arange(4 * 3, dtype=np.int64).reshape(4, 3), regulararray=True)
    y = ak.from_numpy(np.arange(4 * 2, dtype=np.int64).reshape(4, 2), regulararray=True)

    cuda_x = ak.to_backend(x, "cuda")
    cuda_y = ak.to_backend(y, "cuda")

    assert ak.concatenate((cuda_x, cuda_y)).type == ak.types.ArrayType(
        ak.types.ListType(ak.types.NumpyType("int64")), 8
    )


def test_2058_merge_numpy_array_regular_mergebool_false():
    x = ak.from_numpy(np.zeros((4, 3), dtype=np.bool_), regulararray=True)
    y = ak.from_numpy(np.ones((4, 2), dtype=np.int64), regulararray=True)

    cuda_x = ak.to_backend(x, "cuda")
    cuda_y = ak.to_backend(y, "cuda")

    assert ak.concatenate((cuda_x, cuda_y), mergebool=False).type == ak.types.ArrayType(
        ak.types.UnionType(
            [
                ak.types.RegularType(ak.types.NumpyType("bool"), 3),
                ak.types.RegularType(ak.types.NumpyType("int64"), 2),
            ]
        ),
        8,
    )


def test_2058_merge_numpy_array_regular_mergebool_true():
    x = ak.from_numpy(np.zeros((4, 3), dtype=np.bool_), regulararray=True)
    y = ak.from_numpy(np.ones((4, 2), dtype=np.int64), regulararray=True)

    cuda_x = ak.to_backend(x, "cuda")
    cuda_y = ak.to_backend(y, "cuda")

    assert ak.concatenate((cuda_x, cuda_y), mergebool=True).type == ak.types.ArrayType(
        ak.types.ListType(ak.types.NumpyType("int64")), 8
    )


def test_2021_check_TypeTracerArray_in_ak_where():
    conditionals = ak.Array([True, True, True, False, False, False])
    unionarray = ak.Array([1, 2, 3, [4, 5], [], [6]])
    otherarray = ak.Array(range(100, 106))

    cuda_conditionals = ak.to_backend(conditionals, "cuda")
    cuda_unionarray = ak.to_backend(unionarray, "cuda")
    cuda_otherarray = ak.to_backend(otherarray, "cuda")

    result = ak.where(cuda_conditionals, cuda_unionarray, cuda_otherarray)
    assert result.tolist() == [1, 2, 3, 103, 104, 105]
    assert str(result.type) == "6 * union[int64, var * int64]"

    conditionals_tt = ak.Array(cuda_conditionals.layout.to_typetracer())
    unionarray_tt = ak.Array(cuda_unionarray.layout.to_typetracer())
    otherarray_tt = ak.Array(cuda_otherarray.layout.to_typetracer())
    result_tt = ak.where(conditionals_tt, unionarray_tt, otherarray_tt)
    assert str(result_tt.type) == "6 * union[int64, var * int64]"


def test_1991_missed_a_NumpyArray_raw_call_without_underscore():
    cuda_record_a = ak.to_backend(ak.Record({"x": "hello"}), "cuda")
    cuda_record_b = ak.to_backend(ak.Record({"x": b"hello"}), "cuda")

    assert cuda_record_a["x"] == "hello"
    assert cuda_record_b["x"] == b"hello"

    cuda_array_a = ak.to_backend(ak.Array([{"x": "hello"}]), "cuda")
    cuda_array_b = ak.to_backend(ak.Array([{"x": b"hello"}]), "cuda")

    assert cuda_array_a["x"][0] == "hello"
    assert cuda_array_b["x"][0] == b"hello"


def test_1943_regular_indexing():
    array = ak.from_numpy(np.arange(4 * 4).reshape(4, 4))

    cuda_array = ak.to_backend(array, "cuda")

    mask_regular = ak.Array((cuda_array > 4).layout.to_RegularArray())
    assert cuda_array[mask_regular].to_list() == [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


def test_1943_regular_indexing_non_list_index():
    array = ak.Array(
        [
            {"x": 10, "y": 1.0},
            {"x": 30, "y": 20.0},
            {"x": 40, "y": 20.0},
            {"x": "hi", "y": 20.0},
        ]
    )

    cuda_array = ak.to_backend(array, "cuda")

    assert cuda_array[["x"]].to_list() == [{"x": 10}, {"x": 30}, {"x": 40}, {"x": "hi"}]

    fields_ak = ak.to_backend(ak.Array(["x"]), "cuda")
    assert cuda_array[fields_ak].to_list() == [
        {"x": 10},
        {"x": 30},
        {"x": 40},
        {"x": "hi"},
    ]

    fields_np = ak.to_backend(np.array(["x"]), "cuda")
    assert cuda_array[fields_np].to_list() == [
        {"x": 10},
        {"x": 30},
        {"x": 40},
        {"x": "hi"},
    ]

    class SizedIterable:
        def __len__(self):
            return 1

        def __iter__(self):
            return iter(["y"])

    fields_custom = SizedIterable()
    assert cuda_array[fields_custom].to_list() == [
        {"y": 1.0},
        {"y": 20.0},
        {"y": 20.0},
        {"y": 20.0},
    ]

    fields_tuple = ("x",)
    assert cuda_array[fields_tuple].to_list() == [10, 30, 40, "hi"]


def test_1826_ravel_preserve_none():
    array = ak.Array([[1, 2, 3, None], [4, 5, 6, 7, 8], [], [9], None, [10]])

    cuda_array = ak.to_backend(array, "cuda")

    assert ak.ravel(cuda_array).to_list() == [1, 2, 3, None, 4, 5, 6, 7, 8, 9, 10]
    assert ak.flatten(cuda_array, axis=None).to_list() == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ]


def test_1823_fill_none_axis_none():
    array = ak.Array([None, [1, 2, 3, [None, {"x": [None, 2], "y": [1, 4]}]]])

    cuda_array = ak.to_backend(array, "cuda")

    assert ak.fill_none(cuda_array, -1.0, axis=None).to_list() == [
        -1.0,
        [1, 2, 3, [-1.0, {"x": [-1.0, 2], "y": [1, 4]}]],
    ]


def test_1671_categorical_type():
    pytest.importorskip("pyarrow")

    array1 = ak.Array(["one", "two", "one", "one"])
    array2 = ak.str.to_categorical(array1)

    cuda_array1 = ak.to_backend(array1, "cuda")
    cuda_array2 = ak.to_backend(array2, "cuda")

    assert cuda_array1.type != cuda_array2.type
    assert cuda_array2.type == ak.types.ArrayType(
        ak.types.ListType(
            ak.types.NumpyType("uint8", parameters={"__array__": "char"}),
            parameters={"__array__": "string", "__categorical__": True},
        ),
        4,
    )


def test_1671_categorical_type_to_categorical():
    pytest.importorskip("pyarrow")

    array1 = ak.Array(["one", "two", "one", "one"])
    array2 = ak.str.to_categorical(array1)

    cuda_array1 = ak.to_backend(array1, "cuda")
    cuda_array2 = ak.to_backend(array2, "cuda")

    assert cuda_array1.type != cuda_array2.type
    assert cuda_array2.type == ak.types.ArrayType(
        ak.types.ListType(
            ak.types.NumpyType("uint8", parameters={"__array__": "char"}),
            parameters={"__array__": "string", "__categorical__": True},
        ),
        4,
    )


def test_1607_no_reducers_on_records():
    array = ak.Array([[{"rho": -1.1, "phi": -0.1}, {"rho": 1.1, "phi": 0.1}]])

    cuda_array = ak.to_backend(array, "cuda")

    with pytest.raises(TypeError):
        assert ak.to_list(ak.operations.all(cuda_array, axis=1)) == [
            {"phi": True, "rho": True}
        ]

    with pytest.raises(TypeError):
        assert ak.to_list(ak.operations.any(cuda_array, axis=1)) == [
            {"phi": True, "rho": True}
        ]

    with pytest.raises(TypeError):
        assert ak.to_list(ak.operations.argmax(cuda_array, axis=1)) == [
            {"phi": True, "rho": True}
        ]

    with pytest.raises(TypeError):
        assert ak.to_list(ak.operations.argmin(cuda_array, axis=1)) == [
            {"phi": 0, "rho": 0}
        ]

    with pytest.raises(TypeError):
        assert ak.to_list(ak.operations.count_nonzero(cuda_array, axis=1)) == [
            {"phi": 2, "rho": 2}
        ]

    with pytest.raises(TypeError):
        assert ak.to_list(ak.operations.count(cuda_array, axis=1)) == [
            {"phi": 2, "rho": 2}
        ]

    with pytest.raises(TypeError):
        assert ak.to_list(ak.operations.max(cuda_array, axis=1)) == [
            {"phi": 0.1, "rho": 1.1}
        ]

    with pytest.raises(TypeError):
        assert ak.to_list(ak.operations.min(cuda_array, axis=1)) == [
            {"phi": -0.1, "rho": -1.1}
        ]

    with pytest.raises(TypeError):
        assert ak.to_list(ak.operations.prod(cuda_array, axis=1)) == [
            {"phi": -0.010000000000000002, "rho": -1.2100000000000002}
        ]

    with pytest.raises(TypeError):
        assert ak.to_list(ak.operations.sum(cuda_array, axis=1)) == [
            {"phi": 0.0, "rho": 0.0}
        ]


def test_1604_preserve_form_in_concatenate_ListOffsetArray():
    a = ak.Array([[0.0, 1.1, 2.2], [], [3.3, 4.4]])
    b = ak.Array([[5.5], [6.6, 7.7, 8.8, 9.9]])

    cuda_a = ak.to_backend(a, "cuda")
    cuda_b = ak.to_backend(b, "cuda")

    c = ak.concatenate([cuda_a, cuda_b])
    ctt = ak.concatenate([cuda_a.layout.to_typetracer(), cuda_b.layout.to_typetracer()])
    assert c.to_list() == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]
    assert c.layout.form == ListOffsetForm("i64", NumpyForm("float64"))
    assert c.layout.form == ctt.layout.form


def test_1604_preserve_form_in_concatenate_ListOffsetArray_ListOffsetArray():
    a = ak.Array([[[0.0, 1.1, 2.2], []], [[3.3, 4.4]]])
    b = ak.Array([[[5.5], [6.6, 7.7, 8.8, 9.9]]])

    cuda_a = ak.to_backend(a, "cuda")
    cuda_b = ak.to_backend(b, "cuda")

    c = ak.concatenate([cuda_a, cuda_b])
    ctt = ak.concatenate([cuda_a.layout.to_typetracer(), cuda_b.layout.to_typetracer()])
    assert c.to_list() == [
        [[0.0, 1.1, 2.2], []],
        [[3.3, 4.4]],
        [[5.5], [6.6, 7.7, 8.8, 9.9]],
    ]
    assert c.layout.form == ListOffsetForm(
        "i64", ListOffsetForm("i64", NumpyForm("float64"))
    )
    assert c.layout.form == ctt.layout.form


def test_1604_preserve_form_in_concatenate_IndexedArray():
    a = ak.contents.indexedarray.IndexedArray(
        ak.index.Index(np.array([5, 4, 3, 2, 1, 0], np.int64)),
        ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
    )
    b = ak.contents.indexedarray.IndexedArray(
        ak.index.Index(np.array([0, 1, 2], np.int64)),
        ak.contents.numpyarray.NumpyArray(np.array([7.7, 8.8, 9.9])),
    )

    cuda_a = ak.to_backend(a, "cuda")
    cuda_b = ak.to_backend(b, "cuda")

    c = ak.concatenate([cuda_a, cuda_b])
    ctt = ak.concatenate([cuda_a.layout.to_typetracer(), cuda_b.layout.to_typetracer()])

    assert c.to_list() == [6.6, 5.5, 4.4, 3.3, 2.2, 1.1, 7.7, 8.8, 9.9]
    assert isinstance(c.layout, ak.contents.IndexedArray)
    assert c.layout.form == IndexedForm("i64", NumpyForm("float64"))
    assert c.layout.form == ctt.layout.form


def test_1604_preserve_form_in_concatenate_BitMaskedArray():
    a = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
            np.packbits(
                np.array(
                    [
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                    ],
                    np.uint8,
                )
            )
        ),
        ak.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=True,
        length=13,
        lsb_order=False,
    )

    cuda_a = ak.to_backend(a, "cuda")

    c = ak.concatenate([cuda_a, cuda_a])

    ctt = ak.concatenate([cuda_a.layout.to_typetracer(), cuda_a.layout.to_typetracer()])

    assert c.to_list() == [
        0.0,
        1.0,
        2.0,
        3.0,
        None,
        None,
        None,
        None,
        1.1,
        None,
        3.3,
        None,
        5.5,
        0.0,
        1.0,
        2.0,
        3.0,
        None,
        None,
        None,
        None,
        1.1,
        None,
        3.3,
        None,
        5.5,
    ]
    assert isinstance(c.layout, ak.contents.BitMaskedArray)
    assert c.layout.valid_when
    assert not c.layout.lsb_order
    assert c.layout.form == BitMaskedForm("u8", NumpyForm("float64"), True, False)
    assert c.layout.form == ctt.layout.form


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


def test_0111_jagged_and_masked_getitem_numpyarray():
    array = ak.highlevel.Array(
        [[1.1, 2.2, 3.3], [], [4.4, 5.5]], check_valid=True
    ).layout
    array2 = ak.highlevel.Array([[[], [], []], [], [[], []]], check_valid=True).layout

    cuda_array = ak.to_backend(array, "cuda")
    cuda_array2 = ak.to_backend(array2, "cuda")

    with pytest.raises(IndexError):
        cuda_array[cuda_array2]
