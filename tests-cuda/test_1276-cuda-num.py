# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import cupy as cp  # noqa: F401
import awkward as ak  # noqa: F401


def test_num_1():
    content = ak._v2.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    ).layout
    bitmask = ak._v2.index.IndexU8(np.array([40, 34], dtype=np.uint8))
    array = ak._v2.Array(
        ak._v2.contents.bitmaskedarray.BitMaskedArray(bitmask, content, False, 9, False)
    )
    cuda_array = ak._v2.to_backend(array, "cuda")
    assert ak._v2.num(cuda_array, 0) == ak._v2.num(array, 0)
    with pytest.raises(NotImplementedError):
        ak._v2.num(cuda_array, 1)


def test_num_2():
    content = ak._v2.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    ).layout
    bytemask = ak._v2.index.Index8(np.array([False, True, False], dtype=bool))
    array = ak._v2.Array(
        ak._v2.contents.bytemaskedarray.ByteMaskedArray(bytemask, content, True)
    )
    cuda_array = ak._v2.to_backend(array, "cuda")
    assert ak._v2.num(cuda_array, 0) == ak._v2.num(array, 0)
    with pytest.raises(NotImplementedError):
        ak._v2.num(cuda_array, 1)


def test_num_3():
    array = ak._v2.Array(
        ak._v2.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5]))
    )
    cuda_array = ak._v2.to_backend(array, "cuda")
    assert ak._v2.num(cuda_array, 0) == ak._v2.num(array, 0)


def test_num_4():
    array = ak._v2.Array(
        ak._v2.contents.numpyarray.NumpyArray(
            np.array([[0.0, 1.1], [2.2, 3.3], [4.4, 5.5]])
        )
    )
    cuda_array = ak._v2.to_backend(array, "cuda")
    assert ak._v2.num(cuda_array, 0) == ak._v2.num(array, 0)
    assert ak._v2.num(cuda_array, 1).tolist() == ak._v2.num(array, 1).tolist()


def test_num_5():
    array = ak._v2.Array(ak._v2.contents.EmptyArray())
    cuda_array = ak._v2.to_backend(array, "cuda")
    assert ak._v2.num(cuda_array, 0) == ak._v2.num(array, 0)


def test_num_6():
    content = ak._v2.contents.numpyarray.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak._v2.index.Index64(np.array([0, 3, 3, 5, 6, 9]))
    array = ak._v2.Array(
        ak._v2.contents.listoffsetarray.ListOffsetArray(offsets, content)
    )
    cuda_array = ak._v2.to_backend(array, "cuda")
    assert ak._v2.num(cuda_array, 0) == ak._v2.num(array, 0)
    assert ak._v2.num(cuda_array, 1).tolist() == ak._v2.num(array, 1).tolist()


def test_num_7():
    content = ak._v2.contents.numpyarray.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak._v2.index.IndexU32(np.array([0, 3, 3, 5, 6, 9]))
    array = ak._v2.Array(
        ak._v2.contents.listoffsetarray.ListOffsetArray(offsets, content)
    )
    cuda_array = ak._v2.to_backend(array, "cuda")
    assert ak._v2.num(cuda_array, 0) == ak._v2.num(array, 0)
    assert ak._v2.num(cuda_array, 1).tolist() == ak._v2.num(array, 1).tolist()


def test_num_8():
    content = ak._v2.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10])
    )
    offsets = ak._v2.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak._v2.contents.listoffsetarray.ListOffsetArray(offsets, content)
    regulararray = ak._v2.contents.regulararray.RegularArray(listoffsetarray, 2)
    starts = ak._v2.index.Index64(np.array([0, 1]))
    stops = ak._v2.index.Index64(np.array([2, 3]))
    listarray = ak._v2.contents.listarray.ListArray(starts, stops, regulararray)

    cuda_listoffsetarray = ak._v2.to_backend(listoffsetarray, "cuda")
    assert ak._v2.num(cuda_listoffsetarray, 0) == ak._v2.num(
        ak._v2.Array(listoffsetarray), 0
    )
    assert (
        ak._v2.num(cuda_listoffsetarray, 1).tolist()
        == ak._v2.num(ak._v2.Array(listoffsetarray), 1).tolist()
    )

    cuda_regulararray = ak._v2.to_backend(regulararray, "cuda")
    assert ak._v2.num(cuda_regulararray, 0) == ak._v2.num(ak._v2.Array(regulararray), 0)
    assert (
        ak._v2.num(cuda_regulararray, 1).tolist()
        == ak._v2.num(ak._v2.Array(regulararray), 1).tolist()
    )

    cuda_listarray = ak._v2.to_backend(listarray, "cuda")
    assert ak._v2.num(cuda_listarray, 0) == ak._v2.num(ak._v2.Array(listarray), 0)
    assert (
        ak._v2.num(cuda_listarray, 1).tolist()
        == ak._v2.num(ak._v2.Array(listarray), 1).tolist()
    )

    content1 = ak._v2.contents.numpyarray.NumpyArray(np.array([1, 2, 3, 4, 5]))
    content2 = ak._v2.contents.numpyarray.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak._v2.index.Index32(np.array([0, 3, 3, 5, 6, 9]))

    recordarray = ak._v2.Array(
        ak._v2.contents.recordarray.RecordArray(
            [content1, listoffsetarray, content2, content1],
            fields=["one", "two", "2", "wonky"],
        )
    )

    cuda_recordarray = ak._v2.to_backend(recordarray, "cuda")
    assert (
        ak._v2.num(cuda_recordarray, 0).tolist() == ak._v2.num(recordarray, 0).tolist()
    )

    content0 = ak._v2.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).layout
    content = ak._v2.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    ).layout
    tags = ak._v2.index.Index8(np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=np.int8))
    index = ak._v2.index.Index32(np.array([0, 1, 0, 1, 2, 2, 4, 3], dtype=np.int32))
    unionarray = ak._v2.Array(
        ak._v2.contents.unionarray.UnionArray(tags, index, [content0, content1])
    )

    cuda_unionarray = ak._v2.to_backend(unionarray, "cuda")
    assert ak._v2.num(cuda_unionarray, 0) == ak._v2.num(unionarray, 0)


def test_num_9():
    content = ak._v2.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    index = ak._v2.index.Index32(np.array([0, 2, 4, 6, 8, 9, 7, 5], dtype=np.int64))
    indexedarray = ak._v2.Array(
        ak._v2.contents.indexedarray.IndexedArray(index, content)
    )

    cuda_indexedarray = ak._v2.to_backend(indexedarray, "cuda")
    assert ak._v2.num(cuda_indexedarray, 0) == ak._v2.num(indexedarray, 0)

    ioa = ak._v2.Array(
        ak._v2.contents.indexedoptionarray.IndexedOptionArray(
            ak._v2.index.Index32([-30, 19, 6, 7, -3, 21, 13, 22, 17, 9, -12, 16]),
            ak._v2.contents.numpyarray.NumpyArray(
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
    cuda_ioa = ak._v2.to_backend(ioa, "cuda")
    ak._v2.to_backend(cuda_ioa, "cpu")
    assert ak._v2.num(cuda_ioa, 0) == ak._v2.num(ioa, 0)
