# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import cupy as cp  # noqa: F401
import numpy as np
import pytest

import awkward as ak


@pytest.mark.xfail(reason="unimplemented CUDA Kernels (awkward_ByteMaskedArray_numnull")
def test_num_1():
    content = ak.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    ).layout
    bitmask = ak.index.IndexU8(np.array([40, 34], dtype=np.uint8))
    array = ak.Array(
        ak.contents.bitmaskedarray.BitMaskedArray(bitmask, content, False, 9, False)
    )
    cuda_array = ak.to_backend(array, "cuda")
    assert ak.num(cuda_array, 0) == ak.num(array, 0)
    with pytest.raises(NotImplementedError):
        ak.num(cuda_array, 1)


@pytest.mark.xfail(reason="unimplemented CUDA Kernels (awkward_ByteMaskedArray_numnull")
def test_num_2():
    content = ak.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    ).layout
    bytemask = ak.index.Index8(np.array([False, True, False], dtype=bool))
    array = ak.Array(
        ak.contents.bytemaskedarray.ByteMaskedArray(bytemask, content, True)
    )
    cuda_array = ak.to_backend(array, "cuda")
    assert ak.num(cuda_array, 0) == ak.num(array, 0)
    with pytest.raises(NotImplementedError):
        ak.num(cuda_array, 1)


def test_num_3():
    array = ak.Array(
        ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5]))
    )
    cuda_array = ak.to_backend(array, "cuda")
    assert ak.num(cuda_array, 0) == ak.num(array, 0)


def test_num_4():
    array = ak.Array(
        ak.contents.numpyarray.NumpyArray(
            np.array([[0.0, 1.1], [2.2, 3.3], [4.4, 5.5]])
        )
    )
    cuda_array = ak.to_backend(array, "cuda")
    assert ak.num(cuda_array, 0) == ak.num(array, 0)
    assert ak.num(cuda_array, 1).tolist() == ak.num(array, 1).tolist()


def test_num_5():
    array = ak.Array(ak.contents.EmptyArray())
    cuda_array = ak.to_backend(array, "cuda")
    assert ak.num(cuda_array, 0) == ak.num(array, 0)


def test_num_6():
    content = ak.contents.numpyarray.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 9]))
    array = ak.Array(ak.contents.listoffsetarray.ListOffsetArray(offsets, content))
    cuda_array = ak.to_backend(array, "cuda")
    assert ak.num(cuda_array, 0) == ak.num(array, 0)
    assert ak.num(cuda_array, 1).tolist() == ak.num(array, 1).tolist()


def test_num_7():
    content = ak.contents.numpyarray.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak.index.IndexU32(np.array([0, 3, 3, 5, 6, 9]))
    array = ak.Array(ak.contents.listoffsetarray.ListOffsetArray(offsets, content))
    cuda_array = ak.to_backend(array, "cuda")
    assert ak.num(cuda_array, 0) == ak.num(array, 0)
    assert ak.num(cuda_array, 1).tolist() == ak.num(array, 1).tolist()


def test_num_8():
    content = ak.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak.contents.listoffsetarray.ListOffsetArray(offsets, content)
    regulararray = ak.contents.regulararray.RegularArray(listoffsetarray, 2)
    starts = ak.index.Index64(np.array([0, 1]))
    stops = ak.index.Index64(np.array([2, 3]))
    listarray = ak.contents.listarray.ListArray(starts, stops, regulararray)

    cuda_listoffsetarray = ak.to_backend(listoffsetarray, "cuda")
    assert ak.num(cuda_listoffsetarray, 0) == ak.num(ak.Array(listoffsetarray), 0)
    assert (
        ak.num(cuda_listoffsetarray, 1).tolist()
        == ak.num(ak.Array(listoffsetarray), 1).tolist()
    )

    cuda_regulararray = ak.to_backend(regulararray, "cuda")
    assert ak.num(cuda_regulararray, 0) == ak.num(ak.Array(regulararray), 0)
    assert (
        ak.num(cuda_regulararray, 1).tolist()
        == ak.num(ak.Array(regulararray), 1).tolist()
    )

    cuda_listarray = ak.to_backend(listarray, "cuda")
    assert ak.num(cuda_listarray, 0) == ak.num(ak.Array(listarray), 0)
    assert ak.num(cuda_listarray, 1).tolist() == ak.num(ak.Array(listarray), 1).tolist()

    content1 = ak.contents.numpyarray.NumpyArray(np.array([1, 2, 3, 4, 5]))
    content2 = ak.contents.numpyarray.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak.index.Index32(np.array([0, 3, 3, 5, 6, 9]))

    recordarray = ak.Array(
        ak.contents.recordarray.RecordArray(
            [content1, listoffsetarray, content2, content1],
            fields=["one", "two", "2", "wonky"],
        )
    )

    cuda_recordarray = ak.to_backend(recordarray, "cuda")
    assert ak.num(cuda_recordarray, 0).tolist() == ak.num(recordarray, 0).tolist()

    content0 = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).layout
    content = ak.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    ).layout
    tags = ak.index.Index8(np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=np.int8))
    index = ak.index.Index32(np.array([0, 1, 0, 1, 2, 2, 4, 3], dtype=np.int32))
    unionarray = ak.Array(
        ak.contents.unionarray.UnionArray(tags, index, [content0, content1])
    )

    cuda_unionarray = ak.to_backend(unionarray, "cuda")
    assert ak.num(cuda_unionarray, 0) == ak.num(unionarray, 0)


def test_num_9():
    content = ak.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    index = ak.index.Index32(np.array([0, 2, 4, 6, 8, 9, 7, 5], dtype=np.int64))
    indexedarray = ak.Array(ak.contents.indexedarray.IndexedArray(index, content))

    cuda_indexedarray = ak.to_backend(indexedarray, "cuda")
    assert ak.num(cuda_indexedarray, 0) == ak.num(indexedarray, 0)

    ioa = ak.Array(
        ak.contents.indexedoptionarray.IndexedOptionArray(
            ak.index.Index32([-30, 19, 6, 7, -3, 21, 13, 22, 17, 9, -12, 16]),
            ak.contents.numpyarray.NumpyArray(
                np.array(
                    [
                        5.2,
                        1.7,
                        6.7,
                        -0.4,
                        4.0,
                        7.8,
                        3.8,
                        6.8,
                        4.2,
                        0.3,
                        4.6,
                        6.2,
                        6.9,
                        -0.7,
                        3.9,
                        1.6,
                        8.7,
                        -0.7,
                        3.2,
                        4.3,
                        4.0,
                        5.8,
                        4.2,
                        7.0,
                        5.6,
                        3.8,
                    ]
                )
            ),
        )
    )
    cuda_ioa = ak.to_backend(ioa, "cuda")
    ak.to_backend(cuda_ioa, "cpu")
    assert ak.num(cuda_ioa, 0) == ak.num(ioa, 0)
