# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

to_raggedtensor = ak.operations.to_raggedtensor
from_raggedtensor = ak.operations.from_raggedtensor

tf = pytest.importorskip("tensorflow")

content = ak.contents.NumpyArray(
    np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
)
starts1 = ak.index.Index64(np.array([0, 3, 3, 5, 6]))
stops1 = ak.index.Index64(np.array([3, 3, 5, 6, 9]))
starts2 = ak.index.Index64(np.array([0, 3]))
stops2 = ak.index.Index64(np.array([3, 5]))

array = np.arange(2 * 3 * 5).reshape(2, 3, 5)
content2 = ak.contents.NumpyArray(array.reshape(-1))
inneroffsets = ak.index.Index64(np.array([0, 5, 10, 15, 20, 25, 30]))
outeroffsets = ak.index.Index64(np.array([0, 3, 6]))


def test_convert_to_raggedtensor():
    # a test for ListArray -> RaggedTensor
    array1 = ak.contents.ListArray(starts1, stops1, content)
    assert to_raggedtensor(array1).to_list() == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
        [6.6],
        [7.7, 8.8, 9.9],
    ]

    # a test for awkward.highlevel.Array -> RaggedTensor
    array2 = ak.Array(array1)
    assert to_raggedtensor(array2).to_list() == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
        [6.6],
        [7.7, 8.8, 9.9],
    ]

    # a test for NumpyArray -> RaggedTensor
    array3 = content
    assert to_raggedtensor(array3).to_list() == [
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
    ]

    # a test for RegularArray -> RaggedTensor
    array4 = ak.contents.RegularArray(content, size=2)
    assert to_raggedtensor(array4).to_list() == [
        [1.1, 2.2],
        [3.3, 4.4],
        [5.5, 6.6],
        [7.7, 8.8],
    ]

    # try a single line awkward array
    array5 = ak.Array([3, 1, 4, 1, 9, 2, 6])
    assert to_raggedtensor(array5).to_list() == [[3, 1, 4, 1, 9, 2, 6]]

    # try a multiple ragged array
    array6 = ak.Array([[[1.1, 2.2], [3.3]], [], [[4.4, 5.5]]])
    assert to_raggedtensor(array6).to_list() == [[[1.1, 2.2], [3.3]], [], [[4.4, 5.5]]]

    # try a listoffset array inside a listoffset array
    array7 = ak.contents.ListOffsetArray(
        outeroffsets, ak.contents.ListOffsetArray(inneroffsets, content2)
    )
    assert to_raggedtensor(array7).to_list() == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]

    # try a list array inside a list array

    array8 = ak.contents.ListArray(
        starts2, stops2, ak.contents.ListArray(starts1, stops1, content)
    )
    assert to_raggedtensor(array8).to_list() == [
        [[1.1, 2.2, 3.3], [], [4.4, 5.5]],
        [[6.6], [7.7, 8.8, 9.9]],
    ]

    # try just a python list
    array9 = [3, 1, 4, 1, 9, 2, 6]
    assert to_raggedtensor(array9).to_list() == [[3, 1, 4, 1, 9, 2, 6]]


np_array1 = np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float32)

offsets1 = ak.index.Index64(np.array([0, 2, 3, 3, 5]))
content1 = ak.contents.NumpyArray(np_array1)


def test_convert_from_raggedtensor():
    tf_array1 = tf.RaggedTensor.from_row_splits(
        values=[1.1, 2.2, 3.3, 4.4, 5.5], row_splits=[0, 2, 3, 3, 5]
    )

    ak_array1 = ak.contents.ListOffsetArray(offsets1, content1)
    result1 = ak.to_layout(from_raggedtensor(tf_array1), allow_record=False)
    assert (
        result1.content.data == ak.to_backend(np_array1, result1.backend).layout.data
    ).all()
    assert (
        result1.offsets.data
        == ak.to_backend([0, 2, 3, 3, 5], result1.backend).layout.data
    ).all()
    assert from_raggedtensor(tf_array1).to_list() == ak_array1.to_list()

    tf_array2 = tf.RaggedTensor.from_nested_row_splits(
        flat_values=[3, 1, 4, 1, 5, 9, 2, 6],
        nested_row_splits=([0, 3, 3, 5], [0, 4, 4, 7, 8, 8]),
    )
    assert from_raggedtensor(tf_array2).to_list() == [
        [[3, 1, 4, 1], [], [5, 9, 2]],
        [],
        [[6], []],
    ]
