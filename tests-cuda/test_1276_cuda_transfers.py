# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import cupy as cp  # noqa: F401
import numpy as np
import pytest

import awkward as ak


def test_tocuda():
    content = ak.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    ).layout
    bitmask = ak.index.IndexU8(np.array([40, 34], dtype=np.uint8))
    array = ak.contents.BitMaskedArray(bitmask, content, False, 9, False)
    cuda_array = ak.to_backend(array, "cuda")
    copyback_array = ak.to_backend(cuda_array, "cpu")
    assert ak.to_list(cuda_array) == ak.to_list(array)
    assert ak.to_list(copyback_array) == ak.to_list(array)

    bytemask = ak.index.Index8(np.array([False, True, False], dtype=bool))
    array = ak.contents.ByteMaskedArray(bytemask, content, True)
    cuda_array = ak.to_backend(array, "cuda")
    copyback_array = ak.to_backend(cuda_array, "cpu")
    assert ak.to_list(cuda_array) == ak.to_list(array)
    assert ak.to_list(copyback_array) == ak.to_list(array)

    array = ak.contents.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5]))
    cuda_array = ak.to_backend(array, "cuda")
    copyback_array = ak.to_backend(cuda_array, "cpu")
    assert ak.to_list(cuda_array) == ak.to_list(array)
    assert ak.to_list(copyback_array) == ak.to_list(array)

    array = ak.contents.NumpyArray(np.array([[0.0, 1.1], [2.2, 3.3], [4.4, 5.5]]))
    cuda_array = ak.to_backend(array, "cuda")
    copyback_array = ak.to_backend(cuda_array, "cpu")

    assert ak.to_list(cuda_array) == ak.to_list(array)
    assert ak.to_list(copyback_array) == ak.to_list(array)

    array = ak.contents.EmptyArray()
    cuda_array = ak.to_backend(array, "cuda")
    copyback_array = ak.to_backend(cuda_array, "cpu")
    assert ak.to_list(cuda_array) == ak.to_list(array)
    assert ak.to_list(copyback_array) == ak.to_list(array)


@pytest.mark.skip(
    reason="Can't test this right now because of unimplemented CUDA Kernels (awkward_ListOffsetArray_compact_offsets)"
)
def test_tocuda_unimplementedkernels():
    content = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 9]))
    array = ak.contents.ListOffsetArray(offsets, content)
    cuda_array = ak.to_backend(array, "cuda")
    copyback_array = ak.to_backend(cuda_array, "cpu")
    assert ak.to_list(cuda_array) == ak.to_list(array)
    assert ak.to_list(copyback_array) == ak.to_list(array)

    content = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak.index.IndexU32(np.array([0, 3, 3, 5, 6, 9]))
    array = ak.contents.ListOffsetArray(offsets, content)
    cuda_array = ak.to_backend(array, "cuda")
    copyback_array = ak.to_backend(cuda_array, "cpu")
    assert ak.to_list(cuda_array) == ak.to_list(array)
    assert ak.to_list(copyback_array) == ak.to_list(array)

    # Testing parameters
    content = ak.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    ).layout
    offsets = ak.index.Index32(np.array([0, 3, 3, 5, 6, 9]))
    array = ak.contents.ListOffsetArray(offsets, content)
    cuda_array = ak.to_backend(array, "cuda")
    copyback_array = ak.to_backend(cuda_array, "cpu")
    assert ak.to_list(cuda_array) == ak.to_list(array)
    assert ak.to_list(copyback_array) == ak.to_list(array)

    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)
    regulararray = ak.contents.RegularArray(listoffsetarray, 2)
    starts = ak.index.Index64(np.array([0, 1]))
    stops = ak.index.Index64(np.array([2, 3]))
    listarray = ak.contents.ListArray(starts, stops, regulararray)

    cuda_listoffsetarray = ak.to_backend(listoffsetarray, "cuda")
    copyback_listoffsetarray = ak.to_backend(cuda_listoffsetarray, "cpu")
    assert ak.to_list(cuda_listoffsetarray) == ak.to_list(listoffsetarray)
    assert ak.to_list(copyback_listoffsetarray) == ak.to_list(listoffsetarray)

    cuda_regulararray = ak.to_backend(regulararray, "cuda")
    copyback_regulararray = ak.to_backend(cuda_regulararray, "cpu")
    assert ak.to_list(cuda_regulararray) == ak.to_list(regulararray)
    assert ak.to_list(copyback_regulararray) == ak.to_list(regulararray)

    cuda_listarray = ak.to_backend(listarray, "cuda")
    copyback_listarray = ak.to_backend(cuda_listarray, "cpu")
    assert ak.to_list(cuda_listarray) == ak.to_list(listarray)
    assert ak.to_list(copyback_listarray) == ak.to_list(listarray)

    content1 = ak.contents.NumpyArray(np.array([1, 2, 3, 4, 5]))
    content2 = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak.index.Index32(np.array([0, 3, 3, 5, 6, 9]))

    recordarray = ak.contents.RecordArray(
        [content1, listoffsetarray, content2, content1],
        fields=["one", "two", "2", "wonky"],
    )

    cuda_recordarray = ak.to_backend(recordarray, "cuda")
    copyback_recordarray = ak.to_backend(cuda_recordarray, "cpu")
    assert ak.to_list(cuda_recordarray) == ak.to_list(recordarray)
    assert ak.to_list(copyback_recordarray) == ak.to_list(recordarray)

    content0 = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).layout
    content = ak.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    ).layout
    tags = ak.index.Index8(np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=np.int8))
    index = ak.index.Index32(np.array([0, 1, 0, 1, 2, 2, 4, 3], dtype=np.int32))
    unionarray = ak.contents.UnionArray8_32(tags, index, [content0, content1])

    cuda_unionarray = ak.to_backend(unionarray, "cuda")
    copyback_unionarray = ak.to_backend(cuda_unionarray, "cpu")
    assert ak.to_list(cuda_unionarray) == ak.to_list(unionarray)
    assert ak.to_list(copyback_unionarray) == ak.to_list(unionarray)

    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    index = ak.index.Index32(np.array([0, 2, 4, 6, 8, 9, 7, 5], dtype=np.int64))
    indexedarray = ak.index.IndexedArray32(index, content)

    cuda_indexedarray = ak.to_backend(indexedarray, "cuda")
    copyback_indexedarray = ak.to_backend(cuda_indexedarray, "cpu")
    assert ak.to_list(cuda_indexedarray) == ak.to_list(indexedarray)
    assert ak.to_list(copyback_indexedarray) == ak.to_list(indexedarray)

    bytemaskedarray = ak.contents.ByteMaskedArray(
        ak.index.Index8(np.array([True, True, False, False, False], dtype=np.int8)),
        listoffsetarray,
        True,
    )
    cuda_bytemaskedarray = ak.to_backend(bytemaskedarray, "cuda")
    copyback_bytemaskedarray = ak.to_backend(cuda_bytemaskedarray, "cpu")
    assert ak.to_list(cuda_bytemaskedarray) == ak.to_list(bytemaskedarray)
    assert ak.to_list(copyback_bytemaskedarray) == ak.to_list(bytemaskedarray)

    bytemaskedarray = ak.contents.ByteMaskedArray(
        ak.index.Index8(np.array([True, False], dtype=np.int8)), listarray, True
    )
    cuda_bytemaskedarray = ak.to_backend(bytemaskedarray, "cuda")
    copyback_bytemaskedarray = ak.to_backend(cuda_bytemaskedarray, "cpu")
    assert ak.to_list(cuda_bytemaskedarray) == ak.to_list(bytemaskedarray)
    assert ak.to_list(copyback_bytemaskedarray) == ak.to_list(bytemaskedarray)

    bytemaskedarray = ak.contents.ByteMaskedArray(
        ak.index.Index8(np.array([True, False], dtype=np.int8)), recordarray, True
    )
    cuda_bytemaskedarray = ak.to_backend(bytemaskedarray, "cuda")
    copyback_bytemaskedarray = ak.to_backend(cuda_bytemaskedarray, "cpu")
    assert ak.to_list(cuda_bytemaskedarray) == ak.to_list(bytemaskedarray)
    assert ak.to_list(copyback_bytemaskedarray) == ak.to_list(bytemaskedarray)

    bytemaskedarray = ak.contents.ByteMaskedArray(
        ak.index.Index8(np.array([True, False, False], dtype=np.int8)),
        indexedarray,
        True,
    )
    cuda_bytemaskedarray = ak.to_backend(bytemaskedarray, "cuda")
    copyback_bytemaskedarray = ak.to_backend(cuda_bytemaskedarray, "cpu")
    assert ak.to_list(cuda_bytemaskedarray) == ak.to_list(bytemaskedarray)
    assert ak.to_list(copyback_bytemaskedarray) == ak.to_list(bytemaskedarray)

    bytemaskedarray = ak.contents.ByteMaskedArray(
        ak.index.Index8(np.array([True, False, False], dtype=np.int8)),
        unionarray,
        True,
    )
    cuda_bytemaskedarray = ak.to_backend(bytemaskedarray, "cuda")
    copyback_bytemaskedarray = ak.to_backend(cuda_bytemaskedarray, "cpu")
    assert ak.to_list(cuda_bytemaskedarray) == ak.to_list(bytemaskedarray)
    assert ak.to_list(copyback_bytemaskedarray) == ak.to_list(bytemaskedarray)

    ioa = ak.contents.IndexedOptionArray(
        ak.index.Index32([-30, 19, 6, 7, -3, 21, 13, 22, 17, 9, -12, 16]),
        ak.contents.NumpyArray(
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
    cuda_ioa = ak.to_backend(ioa, "cuda")
    copyback_ioa = ak.to_backend(cuda_ioa, "cpu")
    assert ak.to_list(cuda_ioa) == ak.to_list(ioa)
    assert ak.to_list(copyback_ioa) == ak.to_list(ioa)
