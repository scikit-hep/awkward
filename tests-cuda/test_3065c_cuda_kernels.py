from __future__ import annotations

from itertools import product

import cupy as cp
import numpy as np
import pytest

import awkward as ak

to_list = ak.operations.to_list


@pytest.fixture(scope="function", autouse=True)
def cleanup_cuda():
    yield
    cp._default_memory_pool.free_all_blocks()
    cp.cuda.Device().synchronize()


def test_0546_fill_none_replacement_value_type():
    array = ak.operations.values_astype(
        ak.highlevel.Array([1.1, 2.2, None, 3.3]), np.float32
    )
    cuda_array = ak.to_backend(array, "cuda")

    assert str(ak.operations.fill_none(cuda_array, cp.float32(0)).type) == "4 * float32"


def test_0912_packed_indexed_option_array():
    index = ak.index.Index64(np.r_[0, -1, 2, -1, 4])
    content = ak.contents.NumpyArray(np.arange(8))
    layout = ak.contents.IndexedOptionArray(index, content)
    cuda_layout = ak.to_backend(layout, "cuda")

    cuda_packed = ak.operations.to_packed(cuda_layout, highlevel=False)

    assert to_list(cuda_layout) == to_list(cuda_packed)
    assert isinstance(cuda_packed, ak.contents.IndexedOptionArray)
    assert cp.asarray(cuda_packed.index).tolist() == [0, -1, 1, -1, 2]
    assert len(cuda_packed.content) == 3


def test_0912_packed_indexed_array():
    index = ak.index.Index64(np.array([0, 1, 2, 3, 6, 7, 8]))
    content = ak.contents.NumpyArray(np.arange(10))
    layout = ak.contents.IndexedArray(index, content)
    cuda_layout = ak.to_backend(layout, "cuda")

    cuda_packed = ak.operations.to_packed(cuda_layout, highlevel=False)
    assert to_list(cuda_packed) == to_list(cuda_layout)

    assert isinstance(cuda_packed, ak.contents.NumpyArray)
    assert len(cuda_packed) == len(index)


def test_0912_packed_bit_masked_aray():
    mask = ak.index.IndexU8(np.array([0b10101010]))
    content = ak.contents.NumpyArray(np.arange(16))
    layout = ak.contents.BitMaskedArray(mask, content, False, 8, False)
    cuda_layout = ak.to_backend(layout, "cuda")

    cuda_packed = ak.operations.to_packed(cuda_layout, highlevel=False)
    assert to_list(cuda_packed) == to_list(cuda_layout)
    assert len(cuda_packed.content) == 8


def test_0912_packed_union_array():
    a = ak.contents.NumpyArray(np.arange(4))
    b = ak.contents.NumpyArray(np.arange(4) + 4)
    c = ak.contents.RegularArray(ak.contents.NumpyArray(np.arange(12)), 3)
    layout = ak.contents.UnionArray.simplified(
        ak.index.Index8([1, 1, 2, 2, 0, 0]),
        ak.index.Index64([0, 1, 0, 1, 0, 1]),
        [a, b, c],
    )
    cuda_layout = ak.to_backend(layout, "cuda")

    cuda_packed = ak.operations.to_packed(layout, highlevel=False)
    assert to_list(cuda_packed) == to_list(cuda_layout)
    # Check that it merges like contents
    assert len(cuda_packed.contents) == 2
    index_0 = cp.asarray(cuda_packed.index)[cp.asarray(cuda_packed.tags) == 0]
    assert index_0.tolist() == [0, 1, 2, 3]


empty = ak.highlevel.Array(
    ak.contents.RegularArray(
        ak.highlevel.Array([[1, 2, 3], [], [4, 5]]).layout, 0, zeros_length=0
    )
)
cuda_empty = ak.to_backend(empty, "cuda")


def test_0590_allow_regulararray_size_zero_carry():
    assert cuda_empty[[]].to_list() == []


def test_0590_allow_regulararray_size_zero_mergeable():
    assert ak.operations.concatenate([cuda_empty, cuda_empty]).to_list() == []


def test_0590_allow_regulararray_size_zero_localindex():
    assert ak.operations.local_index(cuda_empty, axis=0).to_list() == []
    assert ak.operations.local_index(cuda_empty, axis=1).to_list() == []
    assert ak.operations.local_index(cuda_empty, axis=2).to_list() == []


def test_0627_behavior_from_dict_of_arrays():
    simple = {"what": "ever"}
    one = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], behavior=simple)
    two = ak.Array([["one", "two"], ["three"], ["four", "five"]], behavior=simple)
    cuda_one = ak.to_backend(one, "cuda")
    cuda_two = ak.to_backend(two, "cuda")

    cuda_three = ak.operations.cartesian({"one": cuda_one, "two": cuda_two})
    assert cuda_three.behavior == {"what": "ever"}


def test_0713_getitem_field_should_simplify_optiontype():
    arr1 = ak.highlevel.Array({"a": [1, 2], "b": [1, None]})
    cuda_arr1 = ak.to_backend(arr1, "cuda")
    cuda_arr2 = ak.operations.mask(cuda_arr1, cp.array([True, True]))

    assert isinstance(cuda_arr2.layout, ak.contents.ByteMaskedArray)
    assert isinstance(cuda_arr2.layout.content, ak.contents.RecordArray)
    assert isinstance(cuda_arr2.layout.content["b"], ak.contents.IndexedOptionArray)

    assert isinstance(cuda_arr2.b.layout, ak.contents.IndexedOptionArray)
    assert isinstance(cuda_arr2.b.layout.content, ak.contents.NumpyArray)

    assert ak.operations.is_none(cuda_arr2.b).to_list() == [False, True]


def test_0766_prevent_combinations_of_characters_cartesian():
    one = ak.Array([1, 2, 3, 4])
    two = ak.Array(["aa", "bb", "cc", "dd"])
    cuda_one = ak.to_backend(one, "cuda")
    cuda_two = ak.to_backend(two, "cuda")

    with pytest.raises(ValueError):
        ak.operations.cartesian([cuda_one, cuda_two], axis=1)

    two = ak.Array([["aa", "bb"], ["cc"], [], ["dd"]])
    cuda_two = ak.to_backend(two, "cuda")

    assert to_list(ak.operations.cartesian([cuda_one, cuda_two], axis=1)) == [
        [(1, "aa"), (1, "bb")],
        [(2, "cc")],
        [],
        [(4, "dd")],
    ]
    with pytest.raises(ValueError):
        ak.operations.cartesian([cuda_one, cuda_two], axis=2)


def test_0773_typeparser_arraytype_12():
    array = ak.Array([[1, 2], [3, 4], [5, 6]])
    cuda_array = ak.to_backend(array, "cuda")

    text = str(ak.to_regular(cuda_array).type)
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert str(parsedtype) == text


def test_0773_typeparser_arraytype_13():
    array = ak.Array([[1, 2], [3, 4], [5, 6]])
    cuda_array = ak.to_backend(array, "cuda")

    text = str(ak.with_parameter(ak.to_regular(cuda_array), "wonky", "string").type)
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert str(parsedtype) == text


def test_0773_typeparser_arraytype_14():
    array = ak.Array([1, 2, 3, [1], [1, 2], [1, 2, 3]])
    cuda_array = ak.to_backend(array, "cuda")

    text = str(ak.with_parameter(cuda_array, "wonky", "string").type)
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert str(parsedtype) == text


def test_0773_typeparser_arraytype_15():
    array = ak.Array([1, 2, 3, None, [1], [1, 2], [1, 2, 3]])
    cuda_array = ak.to_backend(array, "cuda")

    text = str(ak.with_parameter(cuda_array, "wonky", "string").type)
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert str(parsedtype) == text


def test_0773_typeparser_arraytype_16():
    array = ak.Array([1, 2, 3, None, [1], [1, 2], [1, 2, 3]])
    cuda_array = ak.to_backend(array, "cuda")

    text = str(cuda_array.type)
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert str(parsedtype) == text


def test_0773_typeparser_arraytype_17():
    array = ak.Array([1, 2, 3, None, [], [], []])
    cuda_array = ak.to_backend(array, "cuda")

    text = str(cuda_array.type)
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert str(parsedtype) == text


def test_0773_typeparser_arraytype_categorical_1():
    pytest.importorskip("pyarrow")

    array = ak.Array(["one", "one", "two", "three", "one", "three"])
    cuda_array = ak.to_backend(array, "cuda")

    text = str(ak.str.to_categorical(cuda_array).type)
    parsedtype = ak.types.from_datashape(text, highlevel=True)
    assert isinstance(parsedtype, ak.types.ArrayType)
    assert str(parsedtype) == text


def test_0866_getitem_field_and_flatten_unions_getitem_field():
    a1 = ak.operations.zip(
        {"a": [[1], [], [2, 3]], "b": [[4], [], [5, 6]]}, with_name="a1"
    )
    a2 = ak.operations.zip(
        {"a": [[7, 8], [9], []], "b": [[10, 11], [12], []]}, with_name="a2"
    )
    cuda_a1 = ak.to_backend(a1, "cuda")
    cuda_a2 = ak.to_backend(a2, "cuda")

    union = ak.operations.where(cp.array([True, False, True]), cuda_a1, cuda_a2)
    assert str(union.a.type) == "3 * var * int64"


def test_0898_unzip_heterogeneous_records():
    a = ak.Array([{"this": 100}])
    b = ak.Array([{"this": 90, "that": 100}])
    cuda_a = ak.to_backend(a, "cuda")
    cuda_b = ak.to_backend(b, "cuda")

    cuda_c = ak.operations.concatenate((cuda_a, cuda_b))

    with pytest.raises(ValueError):
        ak.operations.unzip(cuda_c)

    a = ak.Array([{"this": 100}])
    b = ak.Array([{"this": 90}])
    cuda_a = ak.to_backend(a, "cuda")
    cuda_b = ak.to_backend(b, "cuda")
    cuda_c = ak.operations.concatenate((cuda_a, cuda_b))

    (tmp,) = ak.operations.unzip(cuda_c)

    assert tmp.to_list() == [100, 90]


def test_0896_content_classes_refactoring_BitMaskedArray_NumpyArray():
    # 4.0, 5.0, 6.0, 7.0, 2.2, 4.4, and 6.6 are inaccessible
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
                    dtype=np.uint8,
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
    cuda_a = ak.to_backend(a, "cuda", highlevel=False)

    assert cuda_a.to_typetracer().form == cuda_a.form
    assert cuda_a.to_typetracer().form.type == cuda_a.form.type
    assert len(cuda_a) == 13
    with pytest.raises(IndexError):
        cuda_a[13]
    with pytest.raises(IndexError):
        cuda_a[-14]
    assert cuda_a[0] == 0.0
    assert cuda_a[1] == 1.0
    assert cuda_a[2] == 2.0
    assert cuda_a[3] == 3.0
    assert cuda_a[4] is None
    assert cuda_a[5] is None
    assert cuda_a[6] is None
    assert cuda_a[7] is None
    assert cuda_a[8] == 1.1
    assert cuda_a[9] is None
    assert cuda_a[10] == 3.3
    assert cuda_a[11] is None
    assert cuda_a[12] == 5.5
    assert cuda_a[-13] == 0.0
    assert cuda_a[-12] == 1.0
    assert cuda_a[-11] == 2.0
    assert cuda_a[-10] == 3.0
    assert cuda_a[-9] is None
    assert cuda_a[-8] is None
    assert cuda_a[-7] is None
    assert cuda_a[-6] is None
    assert cuda_a[-5] == 1.1
    assert cuda_a[-4] is None
    assert cuda_a[-3] == 3.3
    assert cuda_a[-2] is None
    assert cuda_a[-1] == 5.5
    assert isinstance(cuda_a[5:], ak.contents.bytemaskedarray.ByteMaskedArray)
    assert cuda_a.to_typetracer()[5:].form == cuda_a[5:].form
    assert len(cuda_a[5:]) == 8
    assert len(cuda_a[-8:]) == 8
    assert len(cuda_a[5:100]) == 8
    assert len(cuda_a[-8:100]) == 8
    assert cuda_a[5:][2] is None
    assert cuda_a[5:][3] == 1.1
    assert cuda_a[-8:][2] is None
    assert cuda_a[-8:][3] == 1.1
    with pytest.raises(IndexError):
        cuda_a["bad"]

    # 4.0, 5.0, 6.0, 7.0, 2.2, 4.4, and 6.6 are inaccessible
    b = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
            np.packbits(
                np.array(
                    [
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        1,
                        0,
                        1,
                        0,
                    ],
                    dtype=np.uint8,
                )
            )
        ),
        ak.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=False,
        length=13,
        lsb_order=False,
    )
    cuda_b = ak.to_backend(b, "cuda", highlevel=False)

    assert cuda_b.to_typetracer().form == cuda_b.form
    assert cuda_b.to_typetracer().form.type == cuda_b.form.type
    assert len(cuda_b) == 13
    with pytest.raises(IndexError):
        cuda_b[13]
    with pytest.raises(IndexError):
        cuda_b[-14]
    assert cuda_b[0] == 0.0
    assert cuda_b[1] == 1.0
    assert cuda_b[2] == 2.0
    assert cuda_b[3] == 3.0
    assert cuda_b[4] is None
    assert cuda_b[5] is None
    assert cuda_b[6] is None
    assert cuda_b[7] is None
    assert cuda_b[8] == 1.1
    assert cuda_b[9] is None
    assert cuda_b[10] == 3.3
    assert cuda_b[11] is None
    assert cuda_b[12] == 5.5
    assert cuda_b[-13] == 0.0
    assert cuda_b[-12] == 1.0
    assert cuda_b[-11] == 2.0
    assert cuda_b[-10] == 3.0
    assert cuda_b[-9] is None
    assert cuda_b[-8] is None
    assert cuda_b[-7] is None
    assert cuda_b[-6] is None
    assert cuda_b[-5] == 1.1
    assert cuda_b[-4] is None
    assert cuda_b[-3] == 3.3
    assert cuda_b[-2] is None
    assert cuda_b[-1] == 5.5
    assert isinstance(cuda_b[5:], ak.contents.bytemaskedarray.ByteMaskedArray)
    assert cuda_b.to_typetracer()[5:].form == cuda_b[5:].form
    assert len(cuda_b[5:]) == 8
    assert len(cuda_b[-8:]) == 8
    assert len(cuda_b[5:100]) == 8
    assert len(cuda_b[-8:100]) == 8
    assert cuda_b[5:][2] is None
    assert cuda_b[5:][3] == 1.1
    assert cuda_b[-8:][2] is None
    assert cuda_b[-8:][3] == 1.1
    with pytest.raises(IndexError):
        cuda_b["bad"]

    # 4.0, 5.0, 6.0, 7.0, 2.2, 4.4, and 6.6 are inaccessible
    c = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
            np.packbits(
                np.array(
                    [
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                    ],
                    dtype=np.uint8,
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
        lsb_order=True,
    )
    cuda_c = ak.to_backend(c, "cuda", highlevel=False)

    assert cuda_c.to_typetracer().form == cuda_c.form
    assert cuda_c.to_typetracer().form.type == cuda_c.form.type
    assert len(cuda_c) == 13
    with pytest.raises(IndexError):
        cuda_c[13]
    with pytest.raises(IndexError):
        cuda_c[-14]
    assert cuda_c[0] == 0.0
    assert cuda_c[1] == 1.0
    assert cuda_c[2] == 2.0
    assert cuda_c[3] == 3.0
    assert cuda_c[4] is None
    assert cuda_c[5] is None
    assert cuda_c[6] is None
    assert cuda_c[7] is None
    assert cuda_c[8] == 1.1
    assert cuda_c[9] is None
    assert cuda_c[10] == 3.3
    assert cuda_c[11] is None
    assert cuda_c[12] == 5.5
    assert cuda_c[-13] == 0.0
    assert cuda_c[-12] == 1.0
    assert cuda_c[-11] == 2.0
    assert cuda_c[-10] == 3.0
    assert cuda_c[-9] is None
    assert cuda_c[-8] is None
    assert cuda_c[-7] is None
    assert cuda_c[-6] is None
    assert cuda_c[-5] == 1.1
    assert cuda_c[-4] is None
    assert cuda_c[-3] == 3.3
    assert cuda_c[-2] is None
    assert cuda_c[-1] == 5.5
    assert isinstance(cuda_c[5:], ak.contents.bytemaskedarray.ByteMaskedArray)
    assert cuda_c.to_typetracer()[5:].form == cuda_c[5:].form
    assert len(cuda_c[5:]) == 8
    assert len(cuda_c[-8:]) == 8
    assert len(cuda_c[5:100]) == 8
    assert len(cuda_c[-8:100]) == 8
    assert cuda_c[5:][2] is None
    assert cuda_c[5:][3] == 1.1
    assert cuda_c[-8:][2] is None
    assert cuda_c[-8:][3] == 1.1
    with pytest.raises(IndexError):
        cuda_c["bad"]

    # 4.0, 5.0, 6.0, 7.0, 2.2, 4.4, and 6.6 are inaccessible
    d = ak.contents.bitmaskedarray.BitMaskedArray(
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
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                    ],
                    dtype=np.uint8,
                )
            )
        ),
        ak.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=False,
        length=13,
        lsb_order=True,
    )
    cuda_d = ak.to_backend(d, "cuda", highlevel=False)

    assert cuda_d.to_typetracer().form == cuda_d.form
    assert cuda_d.to_typetracer().form.type == cuda_d.form.type
    assert len(cuda_d) == 13
    with pytest.raises(IndexError):
        cuda_d[13]
    with pytest.raises(IndexError):
        cuda_d[-14]
    assert cuda_d[0] == 0.0
    assert cuda_d[1] == 1.0
    assert cuda_d[2] == 2.0
    assert cuda_d[3] == 3.0
    assert cuda_d[4] is None
    assert cuda_d[5] is None
    assert cuda_d[6] is None
    assert cuda_d[7] is None
    assert cuda_d[8] == 1.1
    assert cuda_d[9] is None
    assert cuda_d[10] == 3.3
    assert cuda_d[11] is None
    assert cuda_d[12] == 5.5
    assert cuda_d[-13] == 0.0
    assert cuda_d[-12] == 1.0
    assert cuda_d[-11] == 2.0
    assert cuda_d[-10] == 3.0
    assert cuda_d[-9] is None
    assert cuda_d[-8] is None
    assert cuda_d[-7] is None
    assert cuda_d[-6] is None
    assert cuda_d[-5] == 1.1
    assert cuda_d[-4] is None
    assert cuda_d[-3] == 3.3
    assert cuda_d[-2] is None
    assert cuda_d[-1] == 5.5
    assert isinstance(cuda_d[5:], ak.contents.bytemaskedarray.ByteMaskedArray)
    assert cuda_d.to_typetracer()[5:].form == cuda_d[5:].form
    assert len(cuda_d[5:]) == 8
    assert len(cuda_d[-8:]) == 8
    assert len(cuda_d[5:100]) == 8
    assert len(cuda_d[-8:100]) == 8
    assert cuda_d[5:][2] is None
    assert cuda_d[5:][3] == 1.1
    assert cuda_d[-8:][2] is None
    assert cuda_d[-8:][3] == 1.1
    with pytest.raises(IndexError):
        cuda_d["bad"]


def test_0896_content_classes_refactoring_BitMaskedArray_RecordArray_NumpyArray():
    # 4.0, 5.0, 6.0, 7.0, 2.2, 4.4, and 6.6 are inaccessible
    a = ak.contents.bitmaskedarray.BitMaskedArray(
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
    cuda_a = ak.to_backend(a, "cuda", highlevel=False)

    assert cuda_a.to_typetracer().form == cuda_a.form
    assert cuda_a.to_typetracer().form.type == cuda_a.form.type
    assert len(cuda_a["nest"]) == 13
    assert cuda_a.to_typetracer()["nest"].form == cuda_a["nest"].form
    with pytest.raises(IndexError):
        cuda_a["nest"][13]
    with pytest.raises(IndexError):
        cuda_a["nest"][-14]
    assert cuda_a["nest"][0] == 0.0
    assert cuda_a["nest"][1] == 1.0
    assert cuda_a["nest"][2] == 2.0
    assert cuda_a["nest"][3] == 3.0
    assert cuda_a["nest"][4] is None
    assert cuda_a["nest"][5] is None
    assert cuda_a["nest"][6] is None
    assert cuda_a["nest"][7] is None
    assert cuda_a["nest"][8] == 1.1
    assert cuda_a["nest"][9] is None
    assert cuda_a["nest"][10] == 3.3
    assert cuda_a["nest"][11] is None
    assert cuda_a["nest"][12] == 5.5
    assert cuda_a["nest"][-13] == 0.0
    assert cuda_a["nest"][-12] == 1.0
    assert cuda_a["nest"][-11] == 2.0
    assert cuda_a["nest"][-10] == 3.0
    assert cuda_a["nest"][-9] is None
    assert cuda_a["nest"][-8] is None
    assert cuda_a["nest"][-7] is None
    assert cuda_a["nest"][-6] is None
    assert cuda_a["nest"][-5] == 1.1
    assert cuda_a["nest"][-4] is None
    assert cuda_a["nest"][-3] == 3.3
    assert cuda_a["nest"][-2] is None
    assert cuda_a["nest"][-1] == 5.5
    assert isinstance(cuda_a["nest"][5:], ak.contents.bytemaskedarray.ByteMaskedArray)
    assert cuda_a.to_typetracer()["nest"][5:].form == cuda_a["nest"][5:].form
    assert len(cuda_a["nest"][5:]) == 8
    assert len(cuda_a["nest"][-8:]) == 8
    assert len(cuda_a["nest"][5:100]) == 8
    assert len(cuda_a["nest"][-8:100]) == 8
    assert cuda_a["nest"][5:][2] is None
    assert cuda_a["nest"][5:][3] == 1.1
    assert cuda_a["nest"][-8:][2] is None
    assert cuda_a["nest"][-8:][3] == 1.1
    with pytest.raises(IndexError):
        cuda_a["nest"]["bad"]

    # 4.0, 5.0, 6.0, 7.0, 2.2, 4.4, and 6.6 are inaccessible
    b = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
            np.packbits(
                np.array(
                    [
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        1,
                        0,
                        1,
                        0,
                    ],
                    dtype=np.uint8,
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
        valid_when=False,
        length=13,
        lsb_order=False,
    )
    cuda_b = ak.to_backend(b, "cuda", highlevel=False)

    assert cuda_b.to_typetracer().form == cuda_b.form
    assert cuda_b.to_typetracer().form.type == cuda_b.form.type
    assert len(cuda_b["nest"]) == 13
    assert cuda_b.to_typetracer()["nest"].form == cuda_b["nest"].form
    with pytest.raises(IndexError):
        cuda_b["nest"][13]
    with pytest.raises(IndexError):
        cuda_b["nest"][-14]
    assert cuda_b["nest"][0] == 0.0
    assert cuda_b["nest"][1] == 1.0
    assert cuda_b["nest"][2] == 2.0
    assert cuda_b["nest"][3] == 3.0
    assert cuda_b["nest"][4] is None
    assert cuda_b["nest"][5] is None
    assert cuda_b["nest"][6] is None
    assert cuda_b["nest"][7] is None
    assert cuda_b["nest"][8] == 1.1
    assert cuda_b["nest"][9] is None
    assert cuda_b["nest"][10] == 3.3
    assert cuda_b["nest"][11] is None
    assert cuda_b["nest"][12] == 5.5
    assert cuda_b["nest"][-13] == 0.0
    assert cuda_b["nest"][-12] == 1.0
    assert cuda_b["nest"][-11] == 2.0
    assert cuda_b["nest"][-10] == 3.0
    assert cuda_b["nest"][-9] is None
    assert cuda_b["nest"][-8] is None
    assert cuda_b["nest"][-7] is None
    assert cuda_b["nest"][-6] is None
    assert cuda_b["nest"][-5] == 1.1
    assert cuda_b["nest"][-4] is None
    assert cuda_b["nest"][-3] == 3.3
    assert cuda_b["nest"][-2] is None
    assert cuda_b["nest"][-1] == 5.5
    assert isinstance(cuda_b["nest"][5:], ak.contents.bytemaskedarray.ByteMaskedArray)
    assert cuda_b.to_typetracer()["nest"][5:].form == cuda_b["nest"][5:].form
    assert len(cuda_b["nest"][5:]) == 8
    assert len(cuda_b["nest"][-8:]) == 8
    assert len(cuda_b["nest"][5:100]) == 8
    assert len(cuda_b["nest"][-8:100]) == 8
    assert cuda_b["nest"][5:][2] is None
    assert cuda_b["nest"][5:][3] == 1.1
    assert cuda_b["nest"][-8:][2] is None
    assert cuda_b["nest"][-8:][3] == 1.1
    with pytest.raises(IndexError):
        cuda_b["nest"]["bad"]

    # 4.0, 5.0, 6.0, 7.0, 2.2, 4.4, and 6.6 are inaccessible
    c = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
            np.packbits(
                np.array(
                    [
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                    ],
                    dtype=np.uint8,
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
        lsb_order=True,
    )
    cuda_c = ak.to_backend(c, "cuda", highlevel=False)

    assert cuda_c.to_typetracer().form == cuda_c.form
    assert cuda_c.to_typetracer().form.type == cuda_c.form.type
    assert len(cuda_c["nest"]) == 13
    assert cuda_c.to_typetracer()["nest"].form == cuda_c["nest"].form
    with pytest.raises(IndexError):
        cuda_c["nest"][13]
    with pytest.raises(IndexError):
        cuda_c["nest"][-14]
    assert cuda_c["nest"][0] == 0.0
    assert cuda_c["nest"][1] == 1.0
    assert cuda_c["nest"][2] == 2.0
    assert cuda_c["nest"][3] == 3.0
    assert cuda_c["nest"][4] is None
    assert cuda_c["nest"][5] is None
    assert cuda_c["nest"][6] is None
    assert cuda_c["nest"][7] is None
    assert cuda_c["nest"][8] == 1.1
    assert cuda_c["nest"][9] is None
    assert cuda_c["nest"][10] == 3.3
    assert cuda_c["nest"][11] is None
    assert cuda_c["nest"][12] == 5.5
    assert cuda_c["nest"][-13] == 0.0
    assert cuda_c["nest"][-12] == 1.0
    assert cuda_c["nest"][-11] == 2.0
    assert cuda_c["nest"][-10] == 3.0
    assert cuda_c["nest"][-9] is None
    assert cuda_c["nest"][-8] is None
    assert cuda_c["nest"][-7] is None
    assert cuda_c["nest"][-6] is None
    assert cuda_c["nest"][-5] == 1.1
    assert cuda_c["nest"][-4] is None
    assert cuda_c["nest"][-3] == 3.3
    assert cuda_c["nest"][-2] is None
    assert cuda_c["nest"][-1] == 5.5
    assert isinstance(cuda_c["nest"][5:], ak.contents.bytemaskedarray.ByteMaskedArray)
    assert cuda_c.to_typetracer()["nest"][5:].form == cuda_c["nest"][5:].form
    assert len(cuda_c["nest"][5:]) == 8
    assert len(cuda_c["nest"][-8:]) == 8
    assert len(cuda_c["nest"][5:100]) == 8
    assert len(cuda_c["nest"][-8:100]) == 8
    assert cuda_c["nest"][5:][2] is None
    assert cuda_c["nest"][5:][3] == 1.1
    assert cuda_c["nest"][-8:][2] is None
    assert cuda_c["nest"][-8:][3] == 1.1
    with pytest.raises(IndexError):
        cuda_c["nest"]["bad"]

    # 4.0, 5.0, 6.0, 7.0, 2.2, 4.4, and 6.6 are inaccessible
    d = ak.contents.bitmaskedarray.BitMaskedArray(
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
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                    ],
                    dtype=np.uint8,
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
        valid_when=False,
        length=13,
        lsb_order=True,
    )
    cuda_d = ak.to_backend(d, "cuda", highlevel=False)

    assert cuda_d.to_typetracer().form == cuda_d.form
    assert cuda_d.to_typetracer().form.type == cuda_d.form.type
    assert len(cuda_d["nest"]) == 13
    assert cuda_d.to_typetracer()["nest"].form == cuda_d["nest"].form
    with pytest.raises(IndexError):
        cuda_d["nest"][13]
    with pytest.raises(IndexError):
        cuda_d["nest"][-14]
    assert cuda_d["nest"][0] == 0.0
    assert cuda_d["nest"][1] == 1.0
    assert cuda_d["nest"][2] == 2.0
    assert cuda_d["nest"][3] == 3.0
    assert cuda_d["nest"][4] is None
    assert cuda_d["nest"][5] is None
    assert cuda_d["nest"][6] is None
    assert cuda_d["nest"][7] is None
    assert cuda_d["nest"][8] == 1.1
    assert cuda_d["nest"][9] is None
    assert cuda_d["nest"][10] == 3.3
    assert cuda_d["nest"][11] is None
    assert cuda_d["nest"][12] == 5.5
    assert cuda_d["nest"][-13] == 0.0
    assert cuda_d["nest"][-12] == 1.0
    assert cuda_d["nest"][-11] == 2.0
    assert cuda_d["nest"][-10] == 3.0
    assert cuda_d["nest"][-9] is None
    assert cuda_d["nest"][-8] is None
    assert cuda_d["nest"][-7] is None
    assert cuda_d["nest"][-6] is None
    assert cuda_d["nest"][-5] == 1.1
    assert cuda_d["nest"][-4] is None
    assert cuda_d["nest"][-3] == 3.3
    assert cuda_d["nest"][-2] is None
    assert cuda_d["nest"][-1] == 5.5
    assert isinstance(cuda_d["nest"][5:], ak.contents.bytemaskedarray.ByteMaskedArray)
    assert cuda_d.to_typetracer()["nest"][5:].form == cuda_d["nest"][5:].form
    assert len(cuda_d["nest"][5:]) == 8
    assert len(cuda_d["nest"][-8:]) == 8
    assert len(cuda_d["nest"][5:100]) == 8
    assert len(cuda_d["nest"][-8:100]) == 8
    assert cuda_d["nest"][5:][2] is None
    assert cuda_d["nest"][5:][3] == 1.1
    assert cuda_d["nest"][-8:][2] is None
    assert cuda_d["nest"][-8:][3] == 1.1
    with pytest.raises(IndexError):
        cuda_d["nest"]["bad"]


def test_0896_content_classes_refactoring_UnionArray_NumpyArray():
    # 100 is inaccessible in index
    # 1.1 is inaccessible in contents[1]
    a = ak.contents.unionarray.UnionArray.simplified(
        ak.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], dtype=np.int8)),
        ak.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100])),
        [
            ak.contents.numpyarray.NumpyArray(np.array([1, 2, 3])),
            ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
    )
    cuda_a = ak.to_backend(a, "cuda", highlevel=False)

    assert cuda_a.to_typetracer().form == cuda_a.form
    assert cuda_a.to_typetracer().form.type == cuda_a.form.type
    assert len(cuda_a) == 7
    with pytest.raises(IndexError):
        cuda_a[7]
    with pytest.raises(IndexError):
        cuda_a[-8]
    assert cuda_a[0] == 5.5
    assert cuda_a[1] == 4.4
    assert cuda_a[2] == 1.0
    assert cuda_a[3] == 2.0
    assert cuda_a[4] == 3.3
    assert cuda_a[5] == 3.0
    assert cuda_a[6] == 5.5
    assert cuda_a[-7] == 5.5
    assert cuda_a[-6] == 4.4
    assert cuda_a[-5] == 1.0
    assert cuda_a[-4] == 2.0
    assert cuda_a[-3] == 3.3
    assert cuda_a[-2] == 3.0
    assert cuda_a[-1] == 5.5
    assert cuda_a.to_typetracer()[3:].form == cuda_a[3:].form
    assert len(cuda_a[3:]) == 4
    assert len(cuda_a[-4:]) == 4
    assert len(cuda_a[3:100]) == 4
    assert len(cuda_a[-4:100]) == 4
    assert cuda_a[3:][1] == 3.3
    assert cuda_a[3:][2] == 3.0
    assert cuda_a[-4:][1] == 3.3
    assert cuda_a[-4:][2] == 3.0
    with pytest.raises(IndexError):
        cuda_a["bad"]
    with pytest.raises(IndexError):
        cuda_a[["bad", "good", "ok"]]


def test_0896_content_classes_refactoring_UnionArray_RecordArray_NumpyArray():
    # 100 is inaccessible in index
    # 1.1 is inaccessible in contents[1]
    a = ak.contents.unionarray.UnionArray.simplified(
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
    cuda_a = ak.to_backend(a, "cuda", highlevel=False)

    assert cuda_a.to_typetracer().form == cuda_a.form
    assert cuda_a.to_typetracer().form.type == cuda_a.form.type
    assert len(cuda_a["nest"]) == 7
    assert cuda_a.to_typetracer()["nest"].form == cuda_a["nest"].form
    with pytest.raises(IndexError):
        cuda_a["nest"][7]
    with pytest.raises(IndexError):
        cuda_a["nest"][-8]
    assert cuda_a["nest"][0] == 5.5
    assert cuda_a["nest"][1] == 4.4
    assert cuda_a["nest"][2] == 1.0
    assert cuda_a["nest"][3] == 2.0
    assert cuda_a["nest"][4] == 3.3
    assert cuda_a["nest"][5] == 3.0
    assert cuda_a["nest"][6] == 5.5
    assert cuda_a["nest"][-7] == 5.5
    assert cuda_a["nest"][-6] == 4.4
    assert cuda_a["nest"][-5] == 1.0
    assert cuda_a["nest"][-4] == 2.0
    assert cuda_a["nest"][-3] == 3.3
    assert cuda_a["nest"][-2] == 3.0
    assert cuda_a["nest"][-1] == 5.5
    assert cuda_a.to_typetracer()["nest"][3:].form == cuda_a["nest"][3:].form
    assert len(cuda_a["nest"][3:]) == 4
    assert len(cuda_a["nest"][-4:]) == 4
    assert len(cuda_a["nest"][3:100]) == 4
    assert len(cuda_a["nest"][-4:100]) == 4
    assert cuda_a["nest"][3:][1] == 3.3
    assert cuda_a["nest"][3:][2] == 3.0
    assert cuda_a["nest"][-4:][1] == 3.3
    assert cuda_a["nest"][-4:][2] == 3.0
    with pytest.raises(IndexError):
        cuda_a["nest"]["bad"]


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


behavior_1 = {"foo": "bar"}
behavior_2 = {"baz": "bargh!"}
behavior = {**behavior_1, **behavior_2}


@pytest.mark.parametrize(
    ("func", "axis"),
    [
        (ak.std, -1),
        (ak.var, -1),
        (ak.softmax, -1),
        *product(
            (
                ak.any,
                ak.min,
                ak.sum,
                ak.ptp,
                ak.count_nonzero,
                lambda *args, **kwargs: ak.moment(*args, **kwargs, n=3),
                ak.all,
                ak.mean,
                ak.max,
                ak.prod,
                ak.count,
            ),
            ([-1]),
        ),
    ],
)
def test_2754_highlevel_behavior_missing_reducers(axis, func):
    array = ak.Array([[1, 2, 3, 4], [5], [10]])
    cuda_array = ak.to_backend(array, "cuda", behavior=behavior_1)

    assert (
        func(
            cuda_array,
            axis=axis,
            highlevel=True,
            behavior=behavior_2,
        ).behavior
        == behavior_2
    )
    assert (
        func(
            cuda_array,
            axis=axis,
            highlevel=True,
        ).behavior
        == behavior_1
    )
    del cuda_array


def test_BitMaskedArray():
    content = ak.contents.NumpyArray(np.arange(13))
    mask = ak.index.IndexU8(np.array([58, 59], dtype=np.uint8))
    array = ak.contents.BitMaskedArray(
        mask, content, valid_when=False, length=13, lsb_order=False
    )
    array = ak.to_backend(array, "cuda", highlevel=False)
    assert cp.asarray(array.mask_as_bool(valid_when=True)).tolist() == [
        True,
        True,
        False,
        False,
        False,
        True,
        False,
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    assert cp.asarray(array.to_ByteMaskedArray().mask).tolist() == [
        0,
        0,
        1,
        1,
        1,
        0,
        1,
        0,
        0,
        0,
        1,
        1,
        1,
    ]
    assert cp.asarray(array.to_IndexedOptionArray64().index).tolist() == [
        0,
        1,
        -1,
        -1,
        -1,
        5,
        -1,
        7,
        8,
        9,
        -1,
        -1,
        -1,
    ]
    assert to_list(array) == [
        0,
        1,
        None,
        None,
        None,
        5,
        None,
        7,
        8,
        9,
        None,
        None,
        None,
    ]
    del array
