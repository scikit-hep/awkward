# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy as np
import awkward as ak

def test_NumpyArray():
    array = ak.layout.NumpyArray(np.array(["1chchc", "1chchc", "2sss", "3", "4", "5"], dtype=object), parameters={"__array__": "categorical"})
    assert ak.is_valid(ak.Array(array)) == False
    # FIXME? assert array.is_unique() == False
    array2 = ak.layout.NumpyArray(np.array([5, 6, 1, 3, 4, 5]))
    assert array2.is_unique() == False

def test_ListOffsetArray():
    array = ak.from_iter(["one", "two", "three", "four", "five"], highlevel=False)
    assert ak.to_list(array.sort(0, True, True)) == ["five", "four", "one", "three", "two"]
    assert array.is_unique() == True

    array2 = ak.from_iter(["one", "two", "one", "four", "two"], highlevel=False)
    assert ak.to_list(array2.sort(0, True, True)) == ["four", "one", "one", "two", "two"]
    assert array2.is_unique() == False

    content = ak.layout.NumpyArray(
        np.array([3.3, 1.1, 2.2, 0.0, 4.4, 9.9, 6.6, 7.7, 8.8, 5.5])
    )
    offsets = ak.layout.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, content)
    assert ak.to_list(listoffsetarray) == [[3.3, 1.1, 2.2], [], [0.0, 4.4], [9.9], [6.6, 7.7, 8.8, 5.5], []]
    assert listoffsetarray.is_unique() == True

    content = ak.layout.NumpyArray(
        np.array([3.3, 1.1, 2.2, 0.0, 4.4, 9.9, 2.2, 3.3, 1.1, 5.5])
    )
    offsets = ak.layout.Index64(np.array([0, 3, 3, 5, 6, 9, 10]))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, content)
    assert ak.to_list(listoffsetarray) == [[3.3, 1.1, 2.2], [], [0.0, 4.4], [9.9], [2.2, 3.3, 1.1], [5.5]]
    assert listoffsetarray.is_unique() == False

    content2 = ak.layout.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 1.1, 5.5, 6.6, 7.7, 2.2, 9.9])
    )
    offsets2 = ak.layout.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray2 = ak.layout.ListOffsetArray64(offsets2, content2)
    assert ak.to_list(listoffsetarray2) == [[0.0, 1.1, 2.2], [], [3.3, 1.1], [5.5], [6.6, 7.7, 2.2, 9.9], []]
    assert listoffsetarray2.is_unique() == True

# def test_RecordArray():
#     array = ak.Array(
#         [
#             {"x": 0.0, "y": []},
#             {"x": 8.0, "y": [1]},
#             {"x": 2.2, "y": [2, 2]},
#             {"x": 3.3, "y": [3, 1, 3]},
#             {"x": 4.4, "y": [4, 1, 1, 4]},
#             {"x": 5.5, "y": [5, 4, 5]},
#             {"x": 1.1, "y": [6, 1]},
#             {"x": 7.7, "y": [7]},
#             {"x": 0.0, "y": []},
#         ]
#     )
#     assert ak.to_list(array.x.layout.argsort(0, True, True)) == [0, 8, 6, 2, 3, 4, 5, 7, 1]
#
#     assert ak.is_unique(array) == False

def test_same_categories():
    categories = ak.Array(["one", "two", "three"])
    index1 = ak.layout.Index64(np.array([0, 2, 2, 1, 2, 0, 1, 0], dtype=np.int64))
    index2 = ak.layout.Index64(np.array([1, 1, 2, 1, 0, 0, 0, 1], dtype=np.int64))
    categorical1 = ak.layout.IndexedArray64(index1, categories.layout, parameters={"__array__": "categorical"})
    categorical2 = ak.layout.IndexedArray64(index2, categories.layout, parameters={"__array__": "categorical"})
    array1 = ak.Array(categorical1)
    assert ak.to_list(categorical1.sort(0, True, True)) == ['one', 'one', 'one', 'three', 'three', 'three', 'two', 'two']
    #assert categorical1.is_unique() == False
    array2 = ak.Array(categorical2)
    assert array1.tolist() == ['one', 'three', 'three', 'two', 'three', 'one', 'two', 'one']
    assert array2.tolist() == ['two', 'two', 'three', 'two', 'one', 'one', 'one', 'two']

    assert (array1 == array2).tolist() == [False, False, True, True, False, True, False, False]
