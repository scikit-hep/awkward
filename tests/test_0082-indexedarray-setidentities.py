# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest
import numpy as np
import awkward1 as ak

def test_error():
    content = ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    index = ak.layout.Index64(np.array([0, 2, 4, 6, 8, 10, 12, 14], dtype=np.int64))
    indexedarray = ak.layout.IndexedArray64(index, content)

    with pytest.raises(ValueError) as err:
        indexedarray.setidentities()
    assert str(err.value).startswith("in IndexedArray64 attempting to get 10, max(index) > len(content)")

def test_passthrough_32():
    content = ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    index = ak.layout.Index32(np.array([0, 2, 4, 6, 8, 9, 7, 5], dtype=np.int32))
    indexedarray = ak.layout.IndexedArray32(index, content)

    assert ak.to_list(indexedarray) == [0.0, 2.2, 4.4, 6.6, 8.8, 9.9, 7.7, 5.5]
    indexedarray.setidentities()
    assert np.asarray(indexedarray.identities).tolist() == [[0], [1], [2], [3], [4], [5], [6], [7]]
    assert np.asarray(indexedarray.content.identities).tolist() == [[0], [-1], [1], [-1], [2], [7], [3], [6], [4], [5]]
    assert isinstance(indexedarray.content.identities, ak.layout.Identities32)

def test_passthrough_U32():
    content = ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    index = ak.layout.IndexU32(np.array([0, 2, 4, 6, 8, 9, 7, 5], dtype=np.uint32))
    indexedarray = ak.layout.IndexedArrayU32(index, content)

    assert ak.to_list(indexedarray) == [0.0, 2.2, 4.4, 6.6, 8.8, 9.9, 7.7, 5.5]
    indexedarray.setidentities()
    assert np.asarray(indexedarray.identities).tolist() == [[0], [1], [2], [3], [4], [5], [6], [7]]
    assert np.asarray(indexedarray.content.identities).tolist() == [[0], [-1], [1], [-1], [2], [7], [3], [6], [4], [5]]
    assert isinstance(indexedarray.content.identities, ak.layout.Identities64)

def test_passthrough_64():
    content = ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    index = ak.layout.Index64(np.array([0, 2, 4, 6, 8, 9, 7, 5], dtype=np.int64))
    indexedarray = ak.layout.IndexedArray64(index, content)

    assert ak.to_list(indexedarray) == [0.0, 2.2, 4.4, 6.6, 8.8, 9.9, 7.7, 5.5]
    indexedarray.setidentities()
    assert np.asarray(indexedarray.identities).tolist() == [[0], [1], [2], [3], [4], [5], [6], [7]]
    assert np.asarray(indexedarray.content.identities).tolist() == [[0], [-1], [1], [-1], [2], [7], [3], [6], [4], [5]]
    assert isinstance(indexedarray.content.identities, ak.layout.Identities64)

def test_dontpass_32():
    content = ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    index = ak.layout.Index32(np.array([0, 2, 4, 6, 8, 6, 4, 2, 0], dtype=np.int32))
    indexedarray = ak.layout.IndexedArray32(index, content)

    assert ak.to_list(indexedarray) == [0.0, 2.2, 4.4, 6.6, 8.8, 6.6, 4.4, 2.2, 0.0]
    indexedarray.setidentities()
    assert np.asarray(indexedarray.identities).tolist() == [[0], [1], [2], [3], [4], [5], [6], [7], [8]]
    assert indexedarray.content.identities is None

def test_dontpass_U32():
    content = ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    index = ak.layout.IndexU32(np.array([0, 2, 4, 6, 8, 6, 4, 2, 0], dtype=np.uint32))
    indexedarray = ak.layout.IndexedArrayU32(index, content)

    assert ak.to_list(indexedarray) == [0.0, 2.2, 4.4, 6.6, 8.8, 6.6, 4.4, 2.2, 0.0]
    indexedarray.setidentities()
    assert np.asarray(indexedarray.identities).tolist() == [[0], [1], [2], [3], [4], [5], [6], [7], [8]]
    assert indexedarray.content.identities is None

def test_dontpass_64():
    content = ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    index = ak.layout.Index64(np.array([0, 2, 4, 6, 8, 6, 4, 2, 0], dtype=np.int64))
    indexedarray = ak.layout.IndexedArray64(index, content)

    assert ak.to_list(indexedarray) == [0.0, 2.2, 4.4, 6.6, 8.8, 6.6, 4.4, 2.2, 0.0]
    indexedarray.setidentities()
    assert np.asarray(indexedarray.identities).tolist() == [[0], [1], [2], [3], [4], [5], [6], [7], [8]]
    assert indexedarray.content.identities is None
