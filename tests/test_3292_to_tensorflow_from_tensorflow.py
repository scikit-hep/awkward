# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

to_tensorflow = ak.operations.to_tensorflow
from_tensorflow = ak.operations.from_tensorflow

tf = pytest.importorskip("tensorflow")

a = np.arange(2 * 2 * 2, dtype=np.float64).reshape(2, 2, 2)
b = np.arange(2 * 2 * 2).reshape(2, 2, 2)

array = np.arange(2 * 3 * 5).reshape(2, 3, 5)
content2 = ak.contents.NumpyArray(array.reshape(-1))
inneroffsets = ak.index.Index64(np.array([0, 5, 10, 15, 20, 25, 30]))
outeroffsets = ak.index.Index64(np.array([0, 3, 6]))


def test_to_tensorflow():
    # a basic test for a 4 dimensional array
    array1 = ak.Array([a, b])
    i = 0
    for sub_array in [
        [[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]],
        [[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]],
    ]:
        assert to_tensorflow(array1)[i].numpy().tolist() == sub_array
        i += 1

    # test that the data types are remaining the same (float64 in this case)
    assert array1.layout.to_backend_array().dtype.name in str(
        to_tensorflow(array1).dtype
    )

    # try a listoffset array inside a listoffset array
    array2 = ak.contents.ListOffsetArray(
        outeroffsets, ak.contents.ListOffsetArray(inneroffsets, content2)
    )
    assert to_tensorflow(array2)[0].numpy().tolist() == [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
    ]
    assert to_tensorflow(array2)[1].numpy().tolist() == [
        [15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29],
    ]

    # try just a python list
    array3 = [3, 1, 4, 1, 9, 2, 6]
    assert to_tensorflow(array3).numpy().tolist() == [3, 1, 4, 1, 9, 2, 6]


array1 = tf.constant([[1.0, -1.0], [1.0, -1.0]], dtype=tf.float32)
array2 = tf.constant(np.array([[1, 2, 3], [4, 5, 6]]))


def test_from_tensorflow():
    # Awkward.to_list() == Tensor.numpy().tolist()
    assert from_tensorflow(array1).to_list() == array1.numpy().tolist()

    assert from_tensorflow(array2).to_list() == [[1, 2, 3], [4, 5, 6]]

    # test that the data types are remaining the same (int64 in this case)
    assert from_tensorflow(array1).layout.dtype.name in str(array1.dtype)

    # test that the data types are remaining the same (float32 in this case)
    assert from_tensorflow(array2).layout.dtype.name in str(array2.dtype)
