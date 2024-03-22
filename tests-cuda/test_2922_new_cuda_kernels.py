# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import cupy as cp  # noqa: F401
import numpy as np
import pytest

import awkward as ak

to_list = ak.operations.to_list

try:
    ak.numba.register_and_check()
except ImportError:
    pytest.skip(reason="too old Numba version", allow_module_level=True)


def test_concatenate_operation():
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


def test_drop_none_BitMaskedArray_NumpyArray():
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


def test_drop_none_BitMaskedArray_RecordArray_NumpyArray():
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


def test_drop_none_ByteMaskedArray_NumpyArray():
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


def test_drop_none_ByteMaskedArray_RecordArray_NumpyArray():
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


def test_drop_none_IndexedOptionArray_NumpyArray_outoforder():
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


def test_drop_none_ListArray_IndexedOptionArray_RecordArray_NumpyArray():
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


def test_drop_none_ListOffsetArray_ByteMaskedArray_NumpyArray():
    array = ak.contents.listoffsetarray.ListOffsetArray(
        ak.index.Index(np.array([1, 4, 4, 6, 7], np.int64)),
        ak.contents.bytemaskedarray.ByteMaskedArray(
            ak.index.Index(np.array([1, 0, 1, 0, 1], np.int8)),
            ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
            valid_when=True,
        ),
    )

    cuda_array = ak.to_backend(array, "cuda")

    assert to_list(array) == to_list(cuda_array)
    assert to_list(ak.drop_none(array, axis=1)) == to_list(
        ak.drop_none(cuda_array, axis=1)
    )


def test_drop_none_all_axes():
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
    with pytest.raises(np.AxisError):
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
    with pytest.raises(np.AxisError):
        ak.drop_none(array2, axis=-2).tolist()
        ak.drop_none(cuda_array2, axis=-2).tolist()


def test_improved_axis_to_posaxis_is_none():
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

    with pytest.raises(np.AxisError):
        ak.is_none(array, axis=3)
        ak.is_none(cuda_array, axis=3)

    assert (
        ak.is_none(array, axis=-1).tolist() == ak.is_none(cuda_array, axis=-1).tolist()
    )

    assert (
        ak.is_none(array, axis=-2).tolist() == ak.is_none(cuda_array, axis=-2).tolist()
    )

    with pytest.raises(np.AxisError):
        ak.is_none(array, axis=-3)
        ak.is_none(cuda_array, axis=-3)


def test_improved_axis_to_posaxis_singletons():
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

    with pytest.raises(np.AxisError):
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

    with pytest.raises(np.AxisError):
        ak.singletons(array, axis=-3)
        ak.singletons(cuda_array, axis=-3)


# def test_fillna_unionarray():
#     content1 = ak.operations.from_iter([[], [1.1], [2.2, 2.2]], highlevel=False)
#     content2 = ak.operations.from_iter([["two", "two"], ["one"], []], highlevel=False)
#     tags = ak.index.Index8(np.array([0, 1, 0, 1, 0, 1], dtype=np.int8))
#     index = ak.index.Index64(np.array([0, 0, 1, 1, 2, 2], dtype=np.int64))
#     array = ak.contents.UnionArray(tags, index, [content1, content2])
#     cuda_array = ak.to_backend(array, "cuda")

#     padded_array = ak._do.pad_none(array, 2, 1)
#     padded_cupy_array = ak._do.pad_none(cuda_array, 2, 1)

#     assert padded_array == padded_cupy_array

#     value = ak.contents.NumpyArray(np.array([777]))
#     assert ak._do.fill_none(padded_array, value) == ak._do.fill_none(padded_cupy_array, value)
