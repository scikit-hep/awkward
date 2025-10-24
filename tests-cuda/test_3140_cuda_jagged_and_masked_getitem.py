from __future__ import annotations

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


def test_0111_jagged_and_masked_getitem_bitmaskedarray2b():
    array = ak.operations.from_iter(
        [[0.0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )

    cuda_array = ak.to_backend(array, "cuda", highlevel=False)

    mask = ak.index.IndexU8(cp.array([66], dtype=cp.uint8))
    maskedarray = ak.contents.BitMaskedArray(
        mask, cuda_array, valid_when=False, length=4, lsb_order=True
    )  # lsb_order is irrelevant in this example
    cuda_array = ak.highlevel.Array([[1, 1], None, [0], [0, -1]], backend="cuda").layout

    assert to_list(maskedarray) == [
        [0.0, 1.1, 2.2],
        None,
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert to_list(maskedarray[cuda_array]) == [
        [1.1, 1.1],
        None,
        [5.5],
        [6.6, 9.9],
    ]
    assert maskedarray.to_typetracer()[cuda_array].form == maskedarray[cuda_array].form

    del cuda_array


def test_0111_jagged_and_masked_getitem_bytemaskedarray2b():
    array = ak.operations.from_iter(
        [[0.0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )

    cuda_array = ak.to_backend(array, "cuda", highlevel=False)

    mask = ak.index.Index8(cp.array([0, 1, 0, 0], dtype=cp.int8))
    maskedarray = ak.contents.ByteMaskedArray(mask, cuda_array, valid_when=False)

    cuda_array = ak.highlevel.Array([[1, 1], None, [0], [0, -1]], backend="cuda").layout

    assert to_list(maskedarray) == [
        [0.0, 1.1, 2.2],
        None,
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert to_list(maskedarray[cuda_array]) == [
        [1.1, 1.1],
        None,
        [5.5],
        [6.6, 9.9],
    ]
    assert maskedarray.to_typetracer()[cuda_array].form == maskedarray[cuda_array].form
    del cuda_array


def test_0111_jagged_and_masked_getitem_emptyarray():
    content = ak.contents.EmptyArray()
    offsets = ak.index.Index64(np.array([0, 0, 0, 0, 0], dtype=np.int64))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)

    cuda_listoffsetarray = ak.to_backend(listoffsetarray, "cuda", highlevel=False)

    array1 = ak.highlevel.Array([[], [], [], []], check_valid=True).layout
    cuda_array1 = ak.to_backend(array1, "cuda")

    array2 = ak.highlevel.Array([[], [None], [], []], check_valid=True).layout
    cuda_array2 = ak.to_backend(array2, "cuda")

    array3 = ak.highlevel.Array([[], [], None, []], check_valid=True).layout
    cuda_array3 = ak.to_backend(array3, "cuda")

    array4 = ak.highlevel.Array([[], [None], None, []], check_valid=True).layout
    cuda_array4 = ak.to_backend(array4, "cuda")

    array5 = ak.highlevel.Array([[], [0], [], []], check_valid=True).layout
    cuda_array5 = ak.to_backend(array5, "cuda")

    assert to_list(cuda_listoffsetarray) == [[], [], [], []]

    assert to_list(cuda_listoffsetarray[cuda_array1]) == [[], [], [], []]
    assert (
        cuda_listoffsetarray.to_typetracer()[cuda_array1].form
        == cuda_listoffsetarray[cuda_array1].form
    )

    assert to_list(cuda_listoffsetarray[cuda_array2]) == [[], [None], [], []]
    assert (
        cuda_listoffsetarray.to_typetracer()[cuda_array2].form
        == cuda_listoffsetarray[cuda_array2].form
    )
    assert to_list(cuda_listoffsetarray[cuda_array3]) == [[], [], None, []]
    assert (
        cuda_listoffsetarray.to_typetracer()[cuda_array3].form
        == cuda_listoffsetarray[cuda_array3].form
    )
    assert to_list(cuda_listoffsetarray[cuda_array4]) == [[], [None], None, []]
    assert (
        cuda_listoffsetarray.to_typetracer()[cuda_array4].form
        == cuda_listoffsetarray[cuda_array4].form
    )

    with pytest.raises(IndexError):
        cuda_listoffsetarray[cuda_array5]

    del cuda_listoffsetarray


def test_0111_jagged_and_masked_getitem_indexedarray():
    array = ak.operations.from_iter(
        [[0.0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    cuda_array = ak.to_backend(array, "cuda", highlevel=False)

    index = ak.index.Index64(cp.array([3, 2, 1, 0], dtype=cp.int64))
    cuda_indexedarray = ak.contents.IndexedArray(index, cuda_array)

    assert to_list(cuda_indexedarray) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [0.0, 1.1, 2.2],
    ]

    array1 = ak.highlevel.Array([[0, -1], [0], [], [1, 1]], check_valid=True).layout
    cuda_array1 = ak.to_backend(array1, "cuda")

    assert to_list(cuda_indexedarray[cuda_array1]) == [
        [6.6, 9.9],
        [5.5],
        [],
        [1.1, 1.1],
    ]
    assert (
        cuda_indexedarray.to_typetracer()[cuda_array1].form
        == cuda_indexedarray[cuda_array1].form
    )

    array1 = ak.highlevel.Array(
        [[0, -1], [0], [None], [1, None, 1]], check_valid=True
    ).layout
    cuda_array1 = ak.to_backend(array1, "cuda")

    assert to_list(cuda_indexedarray[cuda_array1]) == [
        [6.6, 9.9],
        [5.5],
        [None],
        [1.1, None, 1.1],
    ]
    assert (
        cuda_indexedarray.to_typetracer()[cuda_array1].form
        == cuda_indexedarray[cuda_array1].form
    )

    array1 = ak.highlevel.Array([[0, -1], [0], None, [1, 1]], check_valid=True).layout
    cuda_array1 = ak.to_backend(array1, "cuda")

    assert to_list(cuda_indexedarray[cuda_array1]) == [
        [6.6, 9.9],
        [5.5],
        None,
        [1.1, 1.1],
    ]
    assert (
        cuda_indexedarray.to_typetracer()[cuda_array1].form
        == cuda_indexedarray[cuda_array1].form
    )

    array1 = ak.highlevel.Array([[0, -1], [0], None, [None]], check_valid=True).layout
    cuda_array1 = ak.to_backend(array1, "cuda")

    assert to_list(cuda_indexedarray[cuda_array1]) == [[6.6, 9.9], [5.5], None, [None]]
    assert (
        cuda_indexedarray.to_typetracer()[cuda_array1].form
        == cuda_indexedarray[cuda_array1].form
    )

    index = ak.index.Index64(cp.array([3, 2, 1, 0], dtype=cp.int64))
    cuda_indexedarray = ak.contents.IndexedOptionArray(index, cuda_array)

    assert to_list(cuda_indexedarray) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [0.0, 1.1, 2.2],
    ]

    array1 = ak.highlevel.Array([[0, -1], [0], [], [1, 1]], check_valid=True)
    cuda_array1 = ak.to_backend(array1, "cuda")

    assert to_list(cuda_indexedarray[cuda_array1]) == [
        [6.6, 9.9],
        [5.5],
        [],
        [1.1, 1.1],
    ]
    assert (
        cuda_indexedarray.to_typetracer()[cuda_array1].form
        == cuda_indexedarray[cuda_array1].form
    )

    array1 = ak.highlevel.Array(
        [[0, -1], [0], [None], [1, None, 1]], check_valid=True
    ).layout
    cuda_array1 = ak.to_backend(array1, "cuda")

    assert to_list(cuda_indexedarray[cuda_array1]) == [
        [6.6, 9.9],
        [5.5],
        [None],
        [1.1, None, 1.1],
    ]
    assert (
        cuda_indexedarray.to_typetracer()[cuda_array1].form
        == cuda_indexedarray[cuda_array1].form
    )

    array1 = ak.highlevel.Array([[0, -1], [0], None, []], check_valid=True).layout
    cuda_array1 = ak.to_backend(array1, "cuda")

    assert to_list(cuda_indexedarray[cuda_array1]) == [[6.6, 9.9], [5.5], None, []]
    assert (
        cuda_indexedarray.to_typetracer()[array1].form
        == cuda_indexedarray[cuda_array1].form
    )

    array1 = ak.highlevel.Array(
        [[0, -1], [0], None, [1, None, 1]], check_valid=True
    ).layout
    cuda_array1 = ak.to_backend(array1, "cuda")

    assert to_list(cuda_indexedarray[cuda_array1]) == [
        [6.6, 9.9],
        [5.5],
        None,
        [1.1, None, 1.1],
    ]
    assert (
        cuda_indexedarray.to_typetracer()[cuda_array1].form
        == cuda_indexedarray[cuda_array1].form
    )

    del cuda_indexedarray
    del cuda_array1


def test_0111_jagged_and_masked_getitem_indexedarray2():
    array = ak.operations.from_iter(
        [[0.0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    cuda_array = ak.to_backend(array, "cuda", highlevel=False)

    index = ak.index.Index64(cp.array([3, 2, -1, 0], dtype=cp.int64))
    cuda_indexedarray = ak.contents.IndexedOptionArray(index, cuda_array)
    cuda_array = ak.highlevel.Array([[0, -1], [0], None, [1, 1]], backend="cuda").layout

    assert to_list(cuda_indexedarray) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        None,
        [0.0, 1.1, 2.2],
    ]
    assert to_list(cuda_indexedarray[cuda_array]) == [
        [6.6, 9.9],
        [5.5],
        None,
        [1.1, 1.1],
    ]
    assert (
        cuda_indexedarray.to_typetracer()[cuda_array].form
        == cuda_indexedarray[cuda_array].form
    )
    del cuda_indexedarray
    del cuda_array


def test_0111_jagged_and_masked_getitem_indexedarray2b():
    array = ak.operations.from_iter(
        [[0.0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    cuda_array = ak.to_backend(array, "cuda", highlevel=False)

    index = ak.index.Index64(cp.array([0, -1, 2, 3], dtype=cp.int64))
    cuda_indexedarray = ak.contents.IndexedOptionArray(index, cuda_array)
    cuda_array = ak.highlevel.Array([[1, 1], None, [0], [0, -1]], backend="cuda").layout

    assert to_list(cuda_indexedarray) == [
        [0.0, 1.1, 2.2],
        None,
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert to_list(cuda_indexedarray[cuda_array]) == [
        [1.1, 1.1],
        None,
        [5.5],
        [6.6, 9.9],
    ]
    assert (
        cuda_indexedarray.to_typetracer()[cuda_array].form
        == cuda_indexedarray[cuda_array].form
    )
    del cuda_indexedarray
    del cuda_array


def test_0111_jagged_and_masked_getitem_indexedarray3():
    array = ak.highlevel.Array([0.0, 1.1, 2.2, None, 4.4, None, None, 7.7]).layout
    cuda_array = ak.to_backend(array, "cuda")

    assert to_list(cuda_array[ak.highlevel.Array([4, 3, 2], backend="cuda")]) == [
        4.4,
        None,
        2.2,
    ]
    assert to_list(
        cuda_array[ak.highlevel.Array([4, 3, 2, None, 1], backend="cuda")]
    ) == [
        4.4,
        None,
        2.2,
        None,
        1.1,
    ]

    array = ak.highlevel.Array([[0.0, 1.1, None, 2.2], [3.3, None, 4.4], [5.5]]).layout
    array2 = ak.highlevel.Array([[3, 2, 2, 1], [1, 2], []]).layout

    cuda_array = ak.to_backend(array, "cuda", highlevel=False)
    cuda_array2 = ak.to_backend(array2, "cuda")

    assert to_list(cuda_array[cuda_array2]) == [
        [2.2, None, None, 1.1],
        [None, 4.4],
        [],
    ]
    assert cuda_array.to_typetracer()[cuda_array2].form == cuda_array[cuda_array2].form

    cuda_array = ak.highlevel.Array(
        [[0.0, 1.1, 2.2], [3.3, 4.4], None, [5.5]], backend="cuda"
    ).layout
    cuda_array2 = ak.highlevel.Array([3, 2, 1], backend="cuda").layout
    cuda_array3 = ak.highlevel.Array([3, 2, 1, None, 0], backend="cuda").layout
    cuda_array4 = ak.highlevel.Array(
        [[2, 1, 1, 0], [1], None, [0]], backend="cuda"
    ).layout
    cuda_array5 = ak.highlevel.Array(
        [[2, 1, 1, 0], None, [1], [0]], backend="cuda"
    ).layout
    cuda_array6 = ak.highlevel.Array(
        [[2, 1, 1, 0], None, [1], [0], None], backend="cuda"
    ).layout

    assert to_list(cuda_array[cuda_array2]) == [[5.5], None, [3.3, 4.4]]
    assert cuda_array.to_typetracer()[cuda_array2].form == cuda_array[cuda_array2].form
    assert to_list(cuda_array[cuda_array3]) == [
        [5.5],
        None,
        [3.3, 4.4],
        None,
        [0.0, 1.1, 2.2],
    ]
    assert cuda_array.to_typetracer()[cuda_array3].form == cuda_array[cuda_array3].form

    assert (to_list(cuda_array[cuda_array4])) == [
        [2.2, 1.1, 1.1, 0.0],
        [4.4],
        None,
        [5.5],
    ]
    assert cuda_array.to_typetracer()[cuda_array4].form == cuda_array[cuda_array4].form

    assert to_list(cuda_array[cuda_array5]) == [
        [2.2, 1.1, 1.1, 0],
        None,
        None,
        [5.5],
    ]
    assert cuda_array.to_typetracer()[cuda_array5].form == cuda_array[cuda_array5].form
    with pytest.raises(IndexError):
        cuda_array[cuda_array6]

    del cuda_array
    del cuda_array2
    del cuda_array3
    del cuda_array4
    del cuda_array5
    del cuda_array6


def test_0111_jagged_and_masked_getitem_jagged():
    array = ak.highlevel.Array(
        [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]], check_valid=True
    ).layout
    array2 = ak.highlevel.Array(
        [[0, -1], [], [-1, 0], [-1], [1, 1, -2, 0]], check_valid=True
    ).layout

    cuda_array = ak.to_backend(array, "cuda", highlevel=False)
    cuda_array2 = ak.to_backend(array2, "cuda")

    assert to_list(cuda_array[cuda_array2]) == [
        [1.1, 3.3],
        [],
        [5.5, 4.4],
        [6.6],
        [8.8, 8.8, 8.8, 7.7],
    ]
    assert cuda_array.to_typetracer()[cuda_array2].form == cuda_array[cuda_array2].form

    del cuda_array
    del cuda_array2


def test_0111_jagged_and_masked_getitem_double_jagged():
    array = ak.highlevel.Array(
        [[[0, 1, 2, 3], [4, 5]], [[6, 7, 8], [9, 10, 11, 12, 13]]], check_valid=True
    ).layout
    array2 = ak.highlevel.Array(
        [[[2, 1, 0], [-1]], [[-1, -2, -3], [2, 1, 1, 3]]], check_valid=True
    ).layout

    cuda_array = ak.to_backend(array, "cuda", highlevel=False)
    cuda_array2 = ak.to_backend(array2, "cuda")

    assert to_list(cuda_array[cuda_array2]) == [
        [[2, 1, 0], [5]],
        [[8, 7, 6], [11, 10, 10, 12]],
    ]
    assert cuda_array.to_typetracer()[cuda_array2].form == cuda_array[cuda_array2].form

    content = ak.operations.from_iter(
        [[0, 1, 2, 3], [4, 5], [6, 7, 8], [9, 10, 11, 12, 13]], highlevel=False
    )
    regulararray = ak.contents.RegularArray(content, 2, zeros_length=0)
    cuda_regulararray = ak.to_backend(regulararray, "cuda", highlevel=False)

    array1 = ak.highlevel.Array([[2, 1, 0], [-1]], check_valid=True).layout
    cuda_array1 = ak.to_backend(array1, "cuda")

    assert to_list(cuda_regulararray[:, cuda_array1]) == [
        [[2, 1, 0], [5]],
        [[8, 7, 6], [13]],
    ]
    assert (
        cuda_regulararray.to_typetracer()[:, cuda_array1].form
        == cuda_regulararray[:, cuda_array1].form
    )
    assert to_list(cuda_regulararray[1:, cuda_array1]) == [[[8, 7, 6], [13]]]
    assert (
        cuda_regulararray.to_typetracer()[1:, cuda_array1].form
        == cuda_regulararray[1:, cuda_array1].form
    )

    offsets = ak.index.Index64(np.array([0, 2, 4], dtype=np.int64))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)
    cuda_listoffsetarray = ak.to_backend(listoffsetarray, "cuda", highlevel=False)

    assert to_list(cuda_listoffsetarray[:, cuda_array1]) == [
        [[2, 1, 0], [5]],
        [[8, 7, 6], [13]],
    ]
    assert (
        cuda_listoffsetarray.to_typetracer()[:, cuda_array1].form
        == cuda_listoffsetarray[:, cuda_array1].form
    )
    assert to_list(cuda_listoffsetarray[1:, cuda_array1]) == [[[8, 7, 6], [13]]]
    assert (
        cuda_listoffsetarray.to_typetracer()[1:, cuda_array1].form
        == cuda_listoffsetarray[1:, cuda_array1].form
    )


def test_0111_jagged_and_masked_getitem_masked_jagged():
    array = ak.highlevel.Array(
        [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]], check_valid=True
    ).layout
    array1 = ak.highlevel.Array(
        [[-1, -2], None, [], None, [-2, 0]], check_valid=True
    ).layout
    cuda_array = ak.to_backend(array, "cuda", highlevel=False)
    cuda_array1 = ak.to_backend(array1, "cuda")

    assert to_list(cuda_array[cuda_array1]) == [[3.3, 2.2], None, [], None, [8.8, 7.7]]
    assert cuda_array.to_typetracer()[cuda_array1].form == cuda_array[cuda_array1].form


def test_0111_jagged_and_masked_getitem_jagged_masked():
    array = ak.highlevel.Array(
        [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]], check_valid=True
    ).layout
    array1 = ak.highlevel.Array(
        [[-1, None], [], [None, 0], [None], [1]], check_valid=True
    ).layout
    cuda_array = ak.to_backend(array, "cuda", highlevel=False)
    cuda_array1 = ak.to_backend(array1, "cuda")

    assert to_list(cuda_array[cuda_array1]) == [
        [3.3, None],
        [],
        [None, 4.4],
        [None],
        [8.8],
    ]
    assert cuda_array.to_typetracer()[cuda_array1].form == cuda_array[cuda_array1].form


def test_0111_jagged_and_masked_getitem_array_boolean_to_int():
    a = ak.operations.from_iter(
        [[True, True, True], [], [True, True], [True], [True, True, True, True]],
        highlevel=False,
    )
    cuda_a = ak.to_backend(a, "cuda", highlevel=False)
    b = ak._slicing._normalise_item_bool_to_int(cuda_a, backend=cuda_a.backend)
    assert to_list(b) == [[0, 1, 2], [], [0, 1], [0], [0, 1, 2, 3]]

    a = ak.operations.from_iter(
        [
            [True, True, False],
            [],
            [True, False],
            [False],
            [True, True, True, False],
        ],
        highlevel=False,
    )
    cuda_a = ak.to_backend(a, "cuda", highlevel=False)
    b = ak._slicing._normalise_item_bool_to_int(cuda_a, backend=cuda_a.backend)
    assert to_list(b) == [[0, 1], [], [0], [], [0, 1, 2]]

    a = ak.operations.from_iter(
        [
            [False, True, True],
            [],
            [False, True],
            [False],
            [False, True, True, True],
        ],
        highlevel=False,
    )
    cuda_a = ak.to_backend(a, "cuda", highlevel=False)
    b = ak._slicing._normalise_item_bool_to_int(cuda_a, backend=cuda_a.backend)
    assert to_list(b) == [[1, 2], [], [1], [], [1, 2, 3]]

    a = ak.operations.from_iter(
        [[True, True, None], [], [True, None], [None], [True, True, True, None]],
        highlevel=False,
    )
    cuda_a = ak.to_backend(a, "cuda", highlevel=False)
    b = ak._slicing._normalise_item_bool_to_int(cuda_a, backend=cuda_a.backend)
    assert to_list(b) == [[0, 1, None], [], [0, None], [None], [0, 1, 2, None]]
    assert (
        b.content.index.data[b.content.index.data >= 0].tolist()
        == np.arange(6).tolist()  # kernels expect nonnegative entries to be arange
    )

    a = ak.operations.from_iter(
        [[None, True, True], [], [None, True], [None], [None, True, True, True]],
        highlevel=False,
    )
    cuda_a = ak.to_backend(a, "cuda", highlevel=False)
    b = ak._slicing._normalise_item_bool_to_int(cuda_a, backend=cuda_a.backend)
    assert to_list(b) == [[None, 1, 2], [], [None, 1], [None], [None, 1, 2, 3]]
    assert (
        b.content.index.data[b.content.index.data >= 0].tolist()
        == np.arange(6).tolist()  # kernels expect nonnegative entries to be arange
    )

    a = ak.operations.from_iter(
        [[False, True, None], [], [False, None], [None], [False, True, True, None]],
        highlevel=False,
    )
    cuda_a = ak.to_backend(a, "cuda", highlevel=False)
    b = ak._slicing._normalise_item_bool_to_int(cuda_a, backend=cuda_a.backend)
    assert to_list(b) == [[1, None], [], [None], [None], [1, 2, None]]
    assert (
        b.content.index.data[b.content.index.data >= 0].tolist()
        == np.arange(3).tolist()  # kernels expect nonnegative entries to be arange
    )

    a = ak.operations.from_iter(
        [[None, True, False], [], [None, False], [None], [None, True, True, False]],
        highlevel=False,
    )
    cuda_a = ak.to_backend(a, "cuda", highlevel=False)
    b = ak._slicing._normalise_item_bool_to_int(cuda_a, backend=cuda_a.backend)
    assert to_list(b) == [[None, 1], [], [None], [None], [None, 1, 2]]
    assert (
        b.content.index.data[b.content.index.data >= 0].tolist()
        == np.arange(3).tolist()  # kernels expect nonnegative entries to be arange
    )


def test_0111_jagged_and_masked_getitem_array_slice():
    array = ak.highlevel.Array(
        [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], check_valid=True
    ).layout
    cuda_array = ak.to_backend(array, "cuda", highlevel=False)

    assert to_list(cuda_array[[5, 2, 2, 3, 9, 0, 1]]) == [
        5.5,
        2.2,
        2.2,
        3.3,
        9.9,
        0.0,
        1.1,
    ]
    assert (
        cuda_array.to_typetracer()[[5, 2, 2, 3, 9, 0, 1]].form
        == cuda_array[[5, 2, 2, 3, 9, 0, 1]].form
    )
    assert to_list(cuda_array[cp.array([5, 2, 2, 3, 9, 0, 1])]) == [
        5.5,
        2.2,
        2.2,
        3.3,
        9.9,
        0.0,
        1.1,
    ]
    assert (
        cuda_array.to_typetracer()[cp.array([5, 2, 2, 3, 9, 0, 1])].form
        == cuda_array[cp.array([5, 2, 2, 3, 9, 0, 1])].form
    )

    array2 = ak.contents.NumpyArray(np.array([5, 2, 2, 3, 9, 0, 1], dtype=np.int32))
    cuda_array2 = ak.to_backend(array2, "cuda")

    assert to_list(cuda_array[cuda_array2]) == [5.5, 2.2, 2.2, 3.3, 9.9, 0.0, 1.1]
    assert cuda_array.to_typetracer()[cuda_array2].form == cuda_array[cuda_array2].form
    assert to_list(
        cuda_array[
            ak.highlevel.Array(
                cp.array([5, 2, 2, 3, 9, 0, 1], dtype=cp.int32), check_valid=True
            )
        ]
    ) == [5.5, 2.2, 2.2, 3.3, 9.9, 0.0, 1.1]
    assert (
        cuda_array.to_typetracer()[
            ak.highlevel.Array(
                cp.array([5, 2, 2, 3, 9, 0, 1], dtype=cp.int32), check_valid=True
            )
        ].form
        == cuda_array[
            ak.highlevel.Array(
                cp.array([5, 2, 2, 3, 9, 0, 1], dtype=cp.int32), check_valid=True
            )
        ].form
    )
    assert to_list(
        cuda_array[
            ak.highlevel.Array(cp.array([5, 2, 2, 3, 9, 0, 1]), check_valid=True)
        ]
    ) == [
        5.5,
        2.2,
        2.2,
        3.3,
        9.9,
        0.0,
        1.1,
    ]
    assert (
        cuda_array.to_typetracer()[
            cp.array(ak.highlevel.Array([5, 2, 2, 3, 9, 0, 1]))
        ].form
        == cuda_array[cp.array(ak.highlevel.Array([5, 2, 2, 3, 9, 0, 1]))].form
    )

    array3 = ak.contents.NumpyArray(
        np.array([False, False, False, False, False, True, False, True, False, True])
    )
    cuda_array3 = ak.to_backend(array3, "cuda")

    assert to_list(cuda_array[cuda_array3]) == [5.5, 7.7, 9.9]
    assert cuda_array.to_typetracer()[cuda_array3].form == cuda_array[cuda_array3].form

    content = ak.contents.NumpyArray(np.array([1, 0, 9, 3, 2, 2, 5], dtype=np.int64))
    index = ak.index.Index64(np.array([6, 5, 4, 3, 2, 1, 0], dtype=np.int64))
    indexedarray = ak.contents.IndexedArray(index, content)
    cuda_indexedarray = ak.to_backend(indexedarray, "cuda")

    assert to_list(cuda_array[cuda_indexedarray]) == [5.5, 2.2, 2.2, 3.3, 9.9, 0.0, 1.1]
    assert (
        cuda_array.to_typetracer()[cuda_indexedarray].form
        == cuda_array[cuda_indexedarray].form
    )
    assert to_list(
        cuda_array[cp.array(ak.highlevel.Array(indexedarray, check_valid=True))]
    ) == [
        5.5,
        2.2,
        2.2,
        3.3,
        9.9,
        0.0,
        1.1,
    ]
    assert (
        cuda_array.to_typetracer()[ak.highlevel.Array(cuda_indexedarray)].form
        == cuda_array[ak.highlevel.Array(cuda_indexedarray)].form
    )

    emptyarray = ak.contents.EmptyArray()
    cuda_emptyarray = ak.to_backend(emptyarray, "cuda")

    assert to_list(cuda_array[cuda_emptyarray]) == []
    assert (
        cuda_array.to_typetracer()[cuda_emptyarray].form
        == cuda_array[cuda_emptyarray].form
    )

    array = ak.highlevel.Array(
        np.array([[0.0, 1.1, 2.2, 3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]),
        check_valid=True,
    ).layout
    cuda_array = ak.to_backend(array, "cuda", highlevel=False)

    numpyarray1 = ak.contents.NumpyArray(np.array([[0, 1], [1, 0]]))
    numpyarray2 = ak.contents.NumpyArray(np.array([[2, 4], [3, 3]]))

    cuda_numpyarray1 = ak.to_backend(numpyarray1, "cuda")
    cuda_numpyarray2 = ak.to_backend(numpyarray2, "cuda")

    assert to_list(
        cuda_array[
            cuda_numpyarray1,
            cuda_numpyarray2,
        ]
    ) == [[2.2, 9.9], [8.8, 3.3]]
    assert (
        cuda_array.to_typetracer()[
            cuda_numpyarray1,
            cuda_numpyarray2,
        ].form
        == cuda_array[
            cuda_numpyarray1,
            cuda_numpyarray2,
        ].form
    )
    assert to_list(cuda_array[cuda_numpyarray1]) == [
        [[0.0, 1.1, 2.2, 3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]],
        [[5.5, 6.6, 7.7, 8.8, 9.9], [0.0, 1.1, 2.2, 3.3, 4.4]],
    ]
    assert (
        cuda_array.to_typetracer()[cuda_numpyarray1].form
        == cuda_array[cuda_numpyarray1].form
    )


def test_0111_jagged_and_masked_getitem_array_slice_2():
    array = ak.highlevel.Array(
        [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], check_valid=True
    ).layout
    cuda_array = ak.to_backend(array, "cuda", highlevel=False)

    content0 = ak.contents.NumpyArray(np.array([5, 2, 2]))
    content1 = ak.contents.NumpyArray(np.array([3, 9, 0, 1]))
    tags = ak.index.Index8(np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.int8))
    index2 = ak.index.Index64(np.array([0, 1, 2, 0, 1, 2, 3], dtype=np.int64))
    unionarray = ak.contents.UnionArray.simplified(tags, index2, [content0, content1])
    cuda_unionarray = ak.to_backend(unionarray, "cuda")

    assert to_list(cuda_array[cuda_unionarray]) == [5.5, 2.2, 2.2, 3.3, 9.9, 0.0, 1.1]
    assert (
        cuda_array.to_typetracer()[cuda_unionarray].form
        == cuda_array[cuda_unionarray].form
    )


def test_0111_jagged_and_masked_getitem_array_slice_with_union():
    array = ak.highlevel.Array(
        [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], check_valid=True
    ).layout
    cuda_array = ak.to_backend(array, "cuda", highlevel=False)

    content0 = ak.contents.NumpyArray(np.array([5, 2, 2]))
    content1 = ak.contents.NumpyArray(np.array([3, 9, 0, 1]))
    tags = ak.index.Index8(np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.int8))
    index2 = ak.index.Index64(np.array([0, 1, 2, 0, 1, 2, 3], dtype=np.int64))
    unionarray = ak.contents.UnionArray.simplified(tags, index2, [content0, content1])
    cuda_unionarray = ak.to_backend(unionarray, "cuda")

    assert to_list(
        cuda_array[ak.highlevel.Array(cuda_unionarray, check_valid=True)]
    ) == [
        5.5,
        2.2,
        2.2,
        3.3,
        9.9,
        0.0,
        1.1,
    ]
    assert (
        cuda_array.to_typetracer()[ak.highlevel.Array(cuda_unionarray)].form
        == cuda_array[ak.highlevel.Array(cuda_unionarray)].form
    )


def test_0111_jagged_and_masked_getitem_jagged_mask():
    array = ak.highlevel.Array(
        [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]], check_valid=True
    ).layout
    cuda_array = ak.to_backend(array, "cuda", highlevel=False)

    assert to_list(
        cuda_array[[[True, True, True], [], [True, True], [True], [True, True, True]]]
    ) == [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]
    assert (
        cuda_array.to_typetracer()[
            [[True, True, True], [], [True, True], [True], [True, True, True]]
        ].form
        == cuda_array[
            [[True, True, True], [], [True, True], [True], [True, True, True]]
        ].form
    )
    assert to_list(
        cuda_array[[[False, True, True], [], [True, True], [True], [True, True, True]]]
    ) == [[2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]
    assert (
        cuda_array.to_typetracer()[
            [[False, True, True], [], [True, True], [True], [True, True, True]]
        ].form
        == cuda_array[
            [[False, True, True], [], [True, True], [True], [True, True, True]]
        ].form
    )
    assert to_list(
        cuda_array[[[True, False, True], [], [True, True], [True], [True, True, True]]]
    ) == [[1.1, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]
    assert (
        cuda_array.to_typetracer()[
            [[True, False, True], [], [True, True], [True], [True, True, True]]
        ].form
        == cuda_array[
            [[True, False, True], [], [True, True], [True], [True, True, True]]
        ].form
    )
    assert to_list(
        cuda_array[[[True, True, True], [], [False, True], [True], [True, True, True]]]
    ) == [[1.1, 2.2, 3.3], [], [5.5], [6.6], [7.7, 8.8, 9.9]]
    assert (
        cuda_array.to_typetracer()[
            [[True, True, True], [], [False, True], [True], [True, True, True]]
        ].form
        == cuda_array[
            [[True, True, True], [], [False, True], [True], [True, True, True]]
        ].form
    )
    assert to_list(
        cuda_array[[[True, True, True], [], [False, False], [True], [True, True, True]]]
    ) == [[1.1, 2.2, 3.3], [], [], [6.6], [7.7, 8.8, 9.9]]
    assert (
        cuda_array.to_typetracer()[
            [[True, True, True], [], [False, False], [True], [True, True, True]]
        ].form
        == cuda_array[
            [[True, True, True], [], [False, False], [True], [True, True, True]]
        ].form
    )


def test_0111_jagged_and_masked_getitem_jagged_missing_mask():
    array = ak.highlevel.Array(
        [[1.1, 2.2, 3.3], [], [4.4, 5.5]], check_valid=True
    ).layout
    cuda_array = ak.to_backend(array, "cuda", highlevel=False)

    assert to_list(cuda_array[[[True, True, True], [], [True, True]]]) == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
    ]
    assert (
        cuda_array.to_typetracer()[[[True, True, True], [], [True, True]]].form
        == cuda_array[[[True, True, True], [], [True, True]]].form
    )
    assert to_list(cuda_array[[[True, False, True], [], [True, True]]]) == [
        [1.1, 3.3],
        [],
        [4.4, 5.5],
    ]
    assert (
        cuda_array.to_typetracer()[[[True, False, True], [], [True, True]]].form
        == cuda_array[[[True, False, True], [], [True, True]]].form
    )
    assert to_list(cuda_array[[[True, True, False], [], [False, None]]]) == [
        [1.1, 2.2],
        [],
        [None],
    ]
    assert (
        cuda_array.to_typetracer()[[[True, True, False], [], [False, None]]].form
        == cuda_array[[[True, True, False], [], [False, None]]].form
    )
    assert to_list(cuda_array[[[True, True, False], [], [True, None]]]) == [
        [1.1, 2.2],
        [],
        [4.4, None],
    ]
    assert (
        cuda_array.to_typetracer()[[[True, True, False], [], [True, None]]].form
        == cuda_array[[[True, True, False], [], [True, None]]].form
    )

    assert to_list(cuda_array[[[True, None, True], [], [True, True]]]) == [
        [1.1, None, 3.3],
        [],
        [4.4, 5.5],
    ]
    assert (
        cuda_array.to_typetracer()[[[True, None, True], [], [True, True]]].form
        == cuda_array[[[True, None, True], [], [True, True]]].form
    )
    assert to_list(cuda_array[[[True, None, False], [], [True, True]]]) == [
        [1.1, None],
        [],
        [4.4, 5.5],
    ]
    assert (
        cuda_array.to_typetracer()[[[True, None, False], [], [True, True]]].form
        == cuda_array[[[True, None, False], [], [True, True]]].form
    )

    assert to_list(cuda_array[[[False, None, False], [], [True, True]]]) == [
        [None],
        [],
        [4.4, 5.5],
    ]
    assert (
        cuda_array.to_typetracer()[[[False, None, False], [], [True, True]]].form
        == cuda_array[[[False, None, False], [], [True, True]]].form
    )
    assert to_list(cuda_array[[[True, True, False], [], [False, True]]]) == [
        [1.1, 2.2],
        [],
        [5.5],
    ]
    assert (
        cuda_array.to_typetracer()[[[True, True, False], [], [False, True]]].form
        == cuda_array[[[True, True, False], [], [False, True]]].form
    )
    assert to_list(cuda_array[[[True, True, None], [], [False, True]]]) == [
        [1.1, 2.2, None],
        [],
        [5.5],
    ]
    assert (
        cuda_array.to_typetracer()[[[True, True, None], [], [False, True]]].form
        == cuda_array[[[True, True, None], [], [False, True]]].form
    )
    assert to_list(cuda_array[[[True, True, False], [None], [False, True]]]) == [
        [1.1, 2.2],
        [None],
        [5.5],
    ]
    assert (
        cuda_array.to_typetracer()[[[True, True, False], [None], [False, True]]].form
        == cuda_array[[[True, True, False], [None], [False, True]]].form
    )


def test_0111_jagged_and_masked_getitem_masked_of_jagged_of_whatever():
    content = ak.contents.NumpyArray(np.arange(2 * 3 * 5))
    regulararray1 = ak.contents.RegularArray(content, 5, zeros_length=0)
    regulararray2 = ak.contents.RegularArray(regulararray1, 3, zeros_length=0)
    cuda_regulararray2 = ak.to_backend(regulararray2, "cuda", highlevel=False)

    array1 = ak.highlevel.Array(
        [[[2], None, [-1, 2, 0]], [[-3], None, [-5, -3, 4]]], check_valid=True
    ).layout
    array2 = ak.highlevel.Array(
        [[[2], None, [-1, None, 0]], [[-3], None, [-5, None, 4]]],
        check_valid=True,
    ).layout

    cuda_array1 = ak.to_backend(array1, backend="cuda")
    cuda_array2 = ak.to_backend(array2, backend="cuda")

    assert to_list(cuda_regulararray2[cuda_array1]) == [
        [[2], None, [14, 12, 10]],
        [[17], None, [25, 27, 29]],
    ]
    assert (
        cuda_regulararray2.to_typetracer()[cuda_array1].form
        == cuda_regulararray2[cuda_array1].form
    )

    assert to_list(cuda_regulararray2[cuda_array2]) == [
        [[2], None, [14, None, 10]],
        [[17], None, [25, None, 29]],
    ]
    assert (
        cuda_regulararray2.to_typetracer()[cuda_array2].form
        == cuda_regulararray2[cuda_array2].form
    )


def test_0111_jagged_and_masked_getitem_missing():
    array = ak.highlevel.Array(
        [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], check_valid=True
    ).layout
    array2 = ak.highlevel.Array([3, 6, None, None, -2, 6], check_valid=True).layout

    cuda_array = ak.to_backend(array, "cuda", highlevel=False)
    cuda_array2 = ak.to_backend(array2, "cuda")

    assert to_list(cuda_array[cuda_array2]) == [
        3.3,
        6.6,
        None,
        None,
        8.8,
        6.6,
    ]
    assert cuda_array.to_typetracer()[cuda_array2].form == cuda_array[cuda_array2].form

    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0, 11.1, 999])
    )
    regulararray = ak.contents.RegularArray(content, 4, zeros_length=0)
    cuda_regulararray = ak.to_backend(regulararray, "cuda", highlevel=False)

    assert to_list(cuda_regulararray) == [
        [0.0, 1.1, 2.2, 3.3],
        [4.4, 5.5, 6.6, 7.7],
        [8.8, 9.9, 10.0, 11.1],
    ]
    array3 = ak.highlevel.Array([2, 1, 1, None, -1], check_valid=True).layout
    cuda_array3 = ak.to_backend(array3, "cuda")

    assert to_list(cuda_regulararray[cuda_array3]) == [
        [8.8, 9.9, 10.0, 11.1],
        [4.4, 5.5, 6.6, 7.7],
        [4.4, 5.5, 6.6, 7.7],
        None,
        [8.8, 9.9, 10.0, 11.1],
    ]
    assert (
        cuda_regulararray.to_typetracer()[cuda_array3].form
        == cuda_regulararray[cuda_array3].form
    )
    assert to_list(cuda_regulararray[:, cuda_array3]) == [
        [2.2, 1.1, 1.1, None, 3.3],
        [6.6, 5.5, 5.5, None, 7.7],
        [10.0, 9.9, 9.9, None, 11.1],
    ]
    assert (
        cuda_regulararray.to_typetracer()[:, cuda_array3].form
        == cuda_regulararray[:, cuda_array3].form
    )
    assert to_list(cuda_regulararray[1:, cuda_array3]) == [
        [6.6, 5.5, 5.5, None, 7.7],
        [10.0, 9.9, 9.9, None, 11.1],
    ]
    assert (
        cuda_regulararray.to_typetracer()[1:, cuda_array3].form
        == cuda_regulararray[1:, cuda_array3].form
    )

    maskedarray = np.ma.MaskedArray(
        [2, 1, 1, 999, -1], [False, False, False, True, False]
    )
    cuda_maskedarray = ak.to_backend(maskedarray, backend="cuda")

    assert to_list(cuda_regulararray[cuda_maskedarray]) == [
        [8.8, 9.9, 10.0, 11.1],
        [4.4, 5.5, 6.6, 7.7],
        [4.4, 5.5, 6.6, 7.7],
        None,
        [8.8, 9.9, 10.0, 11.1],
    ]
    assert (
        cuda_regulararray.to_typetracer()[cuda_maskedarray].form
        == cuda_regulararray[cuda_maskedarray].form
    )
    assert to_list(
        cuda_regulararray[
            :,
            cuda_maskedarray,
        ]
    ) == [
        [2.2, 1.1, 1.1, None, 3.3],
        [6.6, 5.5, 5.5, None, 7.7],
        [10.0, 9.9, 9.9, None, 11.1],
    ]
    assert (
        cuda_regulararray.to_typetracer()[
            :,
            cuda_maskedarray,
        ].form
        == cuda_regulararray[
            :,
            cuda_maskedarray,
        ].form
    )

    assert to_list(
        cuda_regulararray[
            1:,
            cuda_maskedarray,
        ]
    ) == [[6.6, 5.5, 5.5, None, 7.7], [10.0, 9.9, 9.9, None, 11.1]]
    assert (
        cuda_regulararray.to_typetracer()[
            1:,
            cuda_maskedarray,
        ].form
        == cuda_regulararray[
            1:,
            cuda_maskedarray,
        ].form
    )

    content = ak.contents.NumpyArray(
        np.array([[0.0, 1.1, 2.2, 3.3], [4.4, 5.5, 6.6, 7.7], [8.8, 9.9, 10.0, 11.1]])
    )
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)

    assert to_list(cuda_content[cuda_array3]) == [
        [8.8, 9.9, 10.0, 11.1],
        [4.4, 5.5, 6.6, 7.7],
        [4.4, 5.5, 6.6, 7.7],
        None,
        [8.8, 9.9, 10.0, 11.1],
    ]
    assert (
        cuda_content.to_typetracer()[cuda_array3].form == cuda_content[cuda_array3].form
    )
    assert to_list(cuda_content[:, cuda_array3]) == [
        [2.2, 1.1, 1.1, None, 3.3],
        [6.6, 5.5, 5.5, None, 7.7],
        [10.0, 9.9, 9.9, None, 11.1],
    ]
    assert (
        cuda_content.to_typetracer()[:, cuda_array3].form
        == cuda_content[:, cuda_array3].form
    )
    assert to_list(cuda_content[1:, cuda_array3]) == [
        [6.6, 5.5, 5.5, None, 7.7],
        [10.0, 9.9, 9.9, None, 11.1],
    ]
    assert (
        cuda_content.to_typetracer()[1:, cuda_array3].form
        == cuda_content[1:, cuda_array3].form
    )

    assert to_list(cuda_content[cuda_maskedarray]) == [
        [8.8, 9.9, 10.0, 11.1],
        [4.4, 5.5, 6.6, 7.7],
        [4.4, 5.5, 6.6, 7.7],
        None,
        [8.8, 9.9, 10.0, 11.1],
    ]
    assert (
        cuda_content.to_typetracer()[cuda_maskedarray].form
        == cuda_content[cuda_maskedarray].form
    )
    assert to_list(
        cuda_content[
            :,
            cuda_maskedarray,
        ]
    ) == [
        [2.2, 1.1, 1.1, None, 3.3],
        [6.6, 5.5, 5.5, None, 7.7],
        [10.0, 9.9, 9.9, None, 11.1],
    ]
    assert (
        cuda_content.to_typetracer()[
            :,
            cuda_maskedarray,
        ].form
        == cuda_content[
            :,
            cuda_maskedarray,
        ].form
    )
    assert to_list(
        cuda_content[
            1:,
            cuda_maskedarray,
        ]
    ) == [[6.6, 5.5, 5.5, None, 7.7], [10.0, 9.9, 9.9, None, 11.1]]
    assert (
        cuda_content.to_typetracer()[
            1:,
            cuda_maskedarray,
        ].form
        == cuda_content[
            1:,
            cuda_maskedarray,
        ].form
    )

    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0, 11.1, 999])
    )
    offsets = ak.index.Index64(np.array([0, 4, 8, 12], dtype=np.int64))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)
    cuda_listoffsetarray = ak.to_backend(listoffsetarray, "cuda", highlevel=False)

    assert to_list(cuda_listoffsetarray) == [
        [0.0, 1.1, 2.2, 3.3],
        [4.4, 5.5, 6.6, 7.7],
        [8.8, 9.9, 10.0, 11.1],
    ]
    assert to_list(cuda_listoffsetarray[:, cuda_array3]) == [
        [2.2, 1.1, 1.1, None, 3.3],
        [6.6, 5.5, 5.5, None, 7.7],
        [10.0, 9.9, 9.9, None, 11.1],
    ]
    assert (
        cuda_listoffsetarray.to_typetracer()[:, cuda_array3].form
        == cuda_listoffsetarray[:, cuda_array3].form
    )
    assert to_list(cuda_listoffsetarray[1:, cuda_array3]) == [
        [6.6, 5.5, 5.5, None, 7.7],
        [10.0, 9.9, 9.9, None, 11.1],
    ]
    assert (
        cuda_listoffsetarray.to_typetracer()[1:, cuda_array3].form
        == cuda_listoffsetarray[1:, cuda_array3].form
    )

    assert to_list(
        cuda_listoffsetarray[
            :,
            cuda_maskedarray,
        ]
    ) == [
        [2.2, 1.1, 1.1, None, 3.3],
        [6.6, 5.5, 5.5, None, 7.7],
        [10.0, 9.9, 9.9, None, 11.1],
    ]
    assert (
        cuda_listoffsetarray.to_typetracer()[
            :,
            cuda_maskedarray,
        ].form
        == cuda_listoffsetarray[
            :,
            cuda_maskedarray,
        ].form
    )
    assert to_list(
        cuda_listoffsetarray[
            1:,
            cuda_maskedarray,
        ]
    ) == [[6.6, 5.5, 5.5, None, 7.7], [10.0, 9.9, 9.9, None, 11.1]]
    assert (
        cuda_listoffsetarray.to_typetracer()[
            1:,
            cuda_maskedarray,
        ].form
        == cuda_listoffsetarray[
            1:,
            cuda_maskedarray,
        ].form
    )


def test_0111_jagged_and_masked_getitem_new_slices():
    content = ak.contents.NumpyArray(np.array([1, 0, 9, 3, 2, 2, 5], dtype=np.int64))
    index = ak.index.Index64(np.array([6, 5, -1, 3, 2, -1, 0], dtype=np.int64))
    indexedarray = ak.contents.IndexedOptionArray(index, content)
    cuda_indexedarray = ak.to_backend(indexedarray, backend="cuda")

    assert to_list(cuda_indexedarray) == [5, 2, None, 3, 9, None, 1]

    offsets = ak.index.Index64(np.array([0, 4, 4, 7], dtype=np.int64))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)
    cuda_listoffsetarray = ak.to_backend(listoffsetarray, backend="cuda")

    assert to_list(cuda_listoffsetarray) == [[1, 0, 9, 3], [], [2, 2, 5]]

    offsets = ak.index.Index64(np.array([1, 4, 4, 6], dtype=np.int64))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)
    cuda_listoffsetarray = ak.to_backend(listoffsetarray, backend="cuda")

    assert to_list(cuda_listoffsetarray) == [[0, 9, 3], [], [2, 2]]

    starts = ak.index.Index64(np.array([1, 99, 5], dtype=np.int64))
    stops = ak.index.Index64(np.array([4, 99, 7], dtype=np.int64))
    listarray = ak.contents.ListArray(starts, stops, content)
    cuda_listarray = ak.to_backend(listarray, backend="cuda")

    assert to_list(cuda_listarray) == [[0, 9, 3], [], [2, 5]]


def test_0111_jagged_and_masked_getitem_record():
    array = ak.highlevel.Array(
        [
            {"x": [0, 1, 2], "y": [0.0, 1.1, 2.2, 3.3]},
            {"x": [3, 4, 5, 6], "y": [4.4, 5.5]},
            {"x": [7, 8], "y": [6.6, 7.7, 8.8, 9.9]},
        ],
        check_valid=True,
    ).layout
    cuda_array = ak.to_backend(array, "cuda", highlevel=False)

    array2 = ak.highlevel.Array([[-1, 1], [0, 0, 1], [-1, -2]], check_valid=True).layout
    array3 = ak.highlevel.Array(
        [[-1, 1], [0, 0, None, 1], [-1, -2]], check_valid=True
    ).layout
    array4 = ak.highlevel.Array([[-1, 1], None, [-1, -2]], check_valid=True).layout
    cuda_array2 = ak.to_backend(array2, "cuda")
    cuda_array3 = ak.to_backend(array3, "cuda")
    cuda_array4 = ak.to_backend(array4, "cuda")

    assert to_list(cuda_array[cuda_array2]) == [
        {"x": [2, 1], "y": [3.3, 1.1]},
        {"x": [3, 3, 4], "y": [4.4, 4.4, 5.5]},
        {"x": [8, 7], "y": [9.9, 8.8]},
    ]
    assert cuda_array.to_typetracer()[cuda_array2].form == cuda_array[cuda_array2].form
    assert to_list(cuda_array[cuda_array3]) == [
        {"x": [2, 1], "y": [3.3, 1.1]},
        {"x": [3, 3, None, 4], "y": [4.4, 4.4, None, 5.5]},
        {"x": [8, 7], "y": [9.9, 8.8]},
    ]
    assert cuda_array.to_typetracer()[cuda_array3].form == cuda_array[cuda_array3].form
    assert to_list(cuda_array[cuda_array4]) == [
        {"x": [2, 1], "y": [3.3, 1.1]},
        None,
        {"x": [8, 7], "y": [9.9, 8.8]},
    ]
    assert cuda_array.to_typetracer()[cuda_array4].form == cuda_array[cuda_array4].form


def test_0111_jagged_and_masked_getitem_records_missing():
    array = ak.highlevel.Array(
        [
            {"x": 0, "y": 0.0},
            {"x": 1, "y": 1.1},
            {"x": 2, "y": 2.2},
            {"x": 3, "y": 3.3},
            {"x": 4, "y": 4.4},
            {"x": 5, "y": 5.5},
            {"x": 6, "y": 6.6},
            {"x": 7, "y": 7.7},
            {"x": 8, "y": 8.8},
            {"x": 9, "y": 9.9},
        ],
        check_valid=True,
    ).layout
    array2 = ak.highlevel.Array([3, 1, None, 1, 7], check_valid=True).layout

    cuda_array = ak.to_backend(array, "cuda", highlevel=False)
    cuda_array2 = ak.to_backend(array2, "cuda")

    assert to_list(cuda_array[cuda_array2]) == [
        {"x": 3, "y": 3.3},
        {"x": 1, "y": 1.1},
        None,
        {"x": 1, "y": 1.1},
        {"x": 7, "y": 7.7},
    ]
    assert cuda_array.to_typetracer()[cuda_array2].form == cuda_array[cuda_array2].form

    array = ak.highlevel.Array(
        [
            [
                {"x": 0, "y": 0.0},
                {"x": 1, "y": 1.1},
                {"x": 2, "y": 2.2},
                {"x": 3, "y": 3.3},
            ],
            [
                {"x": 4, "y": 4.4},
                {"x": 5, "y": 5.5},
                {"x": 6, "y": 6.6},
                {"x": 7, "y": 7.7},
                {"x": 8, "y": 8.8},
                {"x": 9, "y": 9.9},
            ],
        ],
        check_valid=True,
    ).layout
    array2 = ak.highlevel.Array([1, None, 2, -1], check_valid=True).layout

    cuda_array = ak.to_backend(array, "cuda", highlevel=False)
    cuda_array2 = ak.to_backend(array2, "cuda")

    assert to_list(cuda_array[:, cuda_array2]) == [
        [{"x": 1, "y": 1.1}, None, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}],
        [{"x": 5, "y": 5.5}, None, {"x": 6, "y": 6.6}, {"x": 9, "y": 9.9}],
    ]
    assert (
        cuda_array.to_typetracer()[:, cuda_array2].form
        == cuda_array[:, cuda_array2].form
    )

    array = ak.highlevel.Array(
        [
            {"x": [0, 1, 2, 3], "y": [0.0, 1.1, 2.2, 3.3]},
            {"x": [4, 5, 6, 7], "y": [4.4, 5.5, 6.6, 7.7]},
            {"x": [8, 9, 10, 11], "y": [8.8, 9.9, 10.0, 11.1]},
        ],
        check_valid=True,
    ).layout
    cuda_array = ak.to_backend(array, "cuda", highlevel=False)

    assert to_list(cuda_array[:, cuda_array2]) == [
        {"x": [1, None, 2, 3], "y": [1.1, None, 2.2, 3.3]},
        {"x": [5, None, 6, 7], "y": [5.5, None, 6.6, 7.7]},
        {"x": [9, None, 10, 11], "y": [9.9, None, 10.0, 11.1]},
    ]
    assert (
        cuda_array.to_typetracer()[:, cuda_array2].form
        == cuda_array[:, cuda_array2].form
    )
    assert to_list(cuda_array[1:, cuda_array2]) == [
        {"x": [5, None, 6, 7], "y": [5.5, None, 6.6, 7.7]},
        {"x": [9, None, 10, 11], "y": [9.9, None, 10.0, 11.1]},
    ]
    assert (
        cuda_array.to_typetracer()[1:, cuda_array2].form
        == cuda_array[1:, cuda_array2].form
    )


def test_0111_jagged_and_masked_getitem_regular_regular():
    content = ak.contents.NumpyArray(np.arange(2 * 3 * 5))
    regulararray1 = ak.contents.RegularArray(content, 5, zeros_length=0)
    regulararray2 = ak.contents.RegularArray(regulararray1, 3, zeros_length=0)

    cuda_regulararray2 = ak.to_backend(regulararray2, "cuda", highlevel=False)

    array1 = ak.highlevel.Array(
        [[[2], [1, -2], [-1, 2, 0]], [[-3], [-4, 3], [-5, -3, 4]]],
        check_valid=True,
    ).layout
    array2 = ak.highlevel.Array(
        [[[2], [1, -2], [-1, None, 0]], [[-3], [-4, 3], [-5, None, 4]]],
        check_valid=True,
    ).layout

    cuda_array1 = ak.to_backend(array1, "cuda")
    cuda_array2 = ak.to_backend(array2, "cuda")

    assert to_list(cuda_regulararray2[cuda_array1]) == [
        [[2], [6, 8], [14, 12, 10]],
        [[17], [21, 23], [25, 27, 29]],
    ]
    assert (
        cuda_regulararray2.to_typetracer()[cuda_array1].form
        == cuda_regulararray2[cuda_array1].form
    )

    assert to_list(cuda_regulararray2[cuda_array2]) == [
        [[2], [6, 8], [14, None, 10]],
        [[17], [21, 23], [25, None, 29]],
    ]
    assert (
        cuda_regulararray2.to_typetracer()[cuda_array2].form
        == cuda_regulararray2[cuda_array2].form
    )


def test_0111_jagged_and_masked_getitem_sequential():
    array = ak.highlevel.Array(
        np.arange(2 * 3 * 5).reshape(2, 3, 5).tolist(), check_valid=True
    ).layout
    array2 = ak.highlevel.Array([[2, 1, 0], [2, 1, 0]], check_valid=True).layout

    cuda_array = ak.to_backend(array, "cuda", highlevel=False)
    cuda_array2 = ak.to_backend(array2, "cuda")

    assert to_list(cuda_array[cuda_array2]) == [
        [[10, 11, 12, 13, 14], [5, 6, 7, 8, 9], [0, 1, 2, 3, 4]],
        [[25, 26, 27, 28, 29], [20, 21, 22, 23, 24], [15, 16, 17, 18, 19]],
    ]
    assert cuda_array.to_typetracer()[cuda_array2].form == cuda_array[cuda_array2].form
    assert to_list(cuda_array[cuda_array2, :2]) == [
        [[10, 11], [5, 6], [0, 1]],
        [[25, 26], [20, 21], [15, 16]],
    ]
    assert (
        cuda_array.to_typetracer()[cuda_array2, :2].form
        == cuda_array[cuda_array2, :2].form
    )


def test_0111_jagged_and_masked_getitem_union():
    one = ak.operations.from_iter(
        [["1.1", "2.2", "3.3"], [], ["4.4", "5.5"]], highlevel=False
    )
    two = ak.operations.from_iter(
        [[6.6], [7.7, 8.8], [], [9.9, 10.0, 11.1, 12.2]], highlevel=False
    )
    tags = ak.index.Index8(np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.int8))
    index = ak.index.Index64(np.array([0, 1, 2, 0, 1, 2, 3], dtype=np.int64))
    unionarray = ak.contents.UnionArray(tags, index, [one, two])

    cuda_unionarray = ak.to_backend(unionarray, "cuda")

    assert to_list(cuda_unionarray) == [
        ["1.1", "2.2", "3.3"],
        [],
        ["4.4", "5.5"],
        [6.6],
        [7.7, 8.8],
        [],
        [9.9, 10.0, 11.1, 12.2],
    ]


def test_0111_jagged_and_masked_getitem_union_2():
    one = ak.operations.from_iter([[1.1, 2.2, 3.3], [], [4.4, 5.5]], highlevel=False)
    two = ak.operations.from_iter(
        [[6.6], [7.7, 8.8], [], [9.9, 10.0, 11.1, 12.2]], highlevel=False
    )
    tags = ak.index.Index8(np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.int8))
    index = ak.index.Index64(np.array([0, 1, 2, 0, 1, 2, 3], dtype=np.int64))
    unionarray = ak.contents.UnionArray.simplified(tags, index, [one, two])
    array = ak.highlevel.Array(
        [[0, -1], [], [1, 1], [], [-1], [], [1, -2, -1]], check_valid=True
    ).layout

    cuda_unionarray = ak.to_backend(unionarray, "cuda", highlevel=False)
    cuda_array = ak.to_backend(array, "cuda")

    assert to_list(cuda_unionarray[cuda_array]) == [
        [1.1, 3.3],
        [],
        [5.5, 5.5],
        [],
        [8.8],
        [],
        [10.0, 11.1, 12.2],
    ]
    assert (
        cuda_unionarray.to_typetracer()[cuda_array].form
        == cuda_unionarray[cuda_array].form
    )
