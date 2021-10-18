# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test_basic():
    content = ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4]))

    ind = np.array([2, 2, 0, 3, 4], dtype=np.int32)
    index = ak.layout.Index32(ind)
    array = ak.layout.IndexedArray32(index, content)

    array = v1_to_v2(array)

    assert ak.to_list(array) == [2.2, 2.2, 0.0, 3.3, 4.4]
    ind[3] = 1
    assert ak.to_list(array) == [2.2, 2.2, 0.0, 1.1, 4.4]

    ind = np.array([2, 2, 0, 3, 4], dtype=np.uint32)
    index = ak.layout.IndexU32(ind)
    array = ak.layout.IndexedArrayU32(index, content)
    assert ak.to_list(array) == [2.2, 2.2, 0.0, 3.3, 4.4]
    ind[3] = 1
    assert ak.to_list(array) == [2.2, 2.2, 0.0, 1.1, 4.4]

    ind = np.array([2, 2, 0, 3, 4], dtype=np.int64)
    index = ak.layout.Index64(ind)
    array = ak.layout.IndexedArray64(index, content)
    assert ak.to_list(array) == [2.2, 2.2, 0.0, 3.3, 4.4]
    ind[3] = 1
    assert ak.to_list(array) == [2.2, 2.2, 0.0, 1.1, 4.4]

    ind = np.array([2, 2, 0, 3, 4], dtype=np.int32)
    index = ak.layout.Index32(ind)
    array = ak.layout.IndexedOptionArray32(index, content)
    assert ak.to_list(array) == [2.2, 2.2, 0.0, 3.3, 4.4]
    ind[3] = 1
    assert ak.to_list(array) == [2.2, 2.2, 0.0, 1.1, 4.4]

    ind = np.array([2, 2, 0, 3, 4], dtype=np.int64)
    index = ak.layout.Index64(ind)
    array = ak.layout.IndexedOptionArray64(index, content)
    assert ak.to_list(array) == [2.2, 2.2, 0.0, 3.3, 4.4]
    ind[3] = 1
    assert ak.to_list(array) == [2.2, 2.2, 0.0, 1.1, 4.4]


def test_null():
    content = ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4]))
    index = ak.layout.Index64(np.array([2, 2, 0, -1, 4], dtype=np.int64))
    array = ak.layout.IndexedOptionArray64(index, content)

    array = v1_to_v2(array)

    assert ak.to_list(array) == [2.2, 2.2, 0.0, None, 4.4]


def test_carry():
    content = ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4]))
    index = ak.layout.Index64(np.array([2, 2, 0, 3, 4], dtype=np.int64))
    indexedarray = ak.layout.IndexedArray64(index, content)
    offsets = ak.layout.Index64(np.array([0, 3, 3, 5], dtype=np.int64))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, indexedarray)

    listoffsetarray = v1_to_v2(listoffsetarray)

    assert ak.to_list(listoffsetarray) == [[2.2, 2.2, 0.0], [], [3.3, 4.4]]
    assert ak.to_list(listoffsetarray[::-1]) == [[3.3, 4.4], [], [2.2, 2.2, 0.0]]
    assert listoffsetarray.typetracer[::-1].form == listoffsetarray[::-1].form
    assert ak.to_list(listoffsetarray[[2, 0]]) == [[3.3, 4.4], [2.2, 2.2, 0.0]]
    assert listoffsetarray.typetracer[[2, 0]].form == listoffsetarray[[2, 0]].form
    assert ak.to_list(listoffsetarray[[2, 0], 1]) == [4.4, 2.2]  # invokes carry
    assert listoffsetarray.typetracer[[2, 0], 1].form == listoffsetarray[[2, 0], 1].form
    assert ak.to_list(listoffsetarray[2:, 1]) == [4.4]  # invokes carry
    assert listoffsetarray.typetracer[2:, 1].form == listoffsetarray[2:, 1].form

    index = ak.layout.Index64(np.array([2, 2, 0, 3, -1], dtype=np.int64))
    indexedarray = ak.layout.IndexedOptionArray64(index, content)
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, indexedarray)
    assert ak.to_list(listoffsetarray) == [[2.2, 2.2, 0.0], [], [3.3, None]]
    assert ak.to_list(listoffsetarray[::-1]) == [[3.3, None], [], [2.2, 2.2, 0.0]]
    assert ak.to_list(listoffsetarray[[2, 0]]) == [[3.3, None], [2.2, 2.2, 0.0]]
    assert ak.to_list(listoffsetarray[[2, 0], 1]) == [None, 2.2]  # invokes carry
    assert ak.to_list(listoffsetarray[2:, 1]) == [None]  # invokes carry


def test_others():
    content = ak.layout.NumpyArray(
        np.array(
            [[0.0, 0.0], [0.1, 1.0], [0.2, 2.0], [0.3, 3.0], [0.4, 4.0], [0.5, 5.0]]
        )
    )
    index = ak.layout.Index64(np.array([4, 0, 3, 1, 3], dtype=np.int64))
    indexedarray = ak.layout.IndexedArray64(index, content)

    indexedarray = v1_to_v2(indexedarray)

    assert indexedarray[3, 0] == 0.1
    assert indexedarray[3, 1] == 1.0
    assert ak.to_list(indexedarray[3, ::-1]) == [1.0, 0.1]
    assert indexedarray.typetracer[3, ::-1].form == indexedarray[3, ::-1].form
    assert ak.to_list(indexedarray[3, [1, 1, 0]]) == [1.0, 1.0, 0.1]
    assert indexedarray.typetracer[3, [1, 1, 0]].form == indexedarray[3, [1, 1, 0]].form
    assert ak.to_list(indexedarray[3:, 0]) == [0.1, 0.3]
    assert indexedarray.typetracer[3:, 0].form == indexedarray[3:, 0].form
    assert ak.to_list(indexedarray[3:, 1]) == [1.0, 3.0]
    assert indexedarray.typetracer[3:, 1].form == indexedarray[3:, 1].form
    assert ak.to_list(indexedarray[3:, ::-1]) == [[1.0, 0.1], [3.0, 0.3]]
    assert indexedarray.typetracer[3:, ::-1].form == indexedarray[3:, ::-1].form
    assert ak.to_list(indexedarray[3:, [1, 1, 0]]) == [[1.0, 1.0, 0.1], [3.0, 3.0, 0.3]]
    assert (
        indexedarray.typetracer[3:, [1, 1, 0]].form == indexedarray[3:, [1, 1, 0]].form
    )


def test_missing():
    content = ak.layout.NumpyArray(
        np.array(
            [[0.0, 0.0], [0.1, 1.0], [0.2, 2.0], [0.3, 3.0], [0.4, 4.0], [0.5, 5.0]]
        )
    )
    index = ak.layout.Index64(np.array([4, 0, 3, -1, 3], dtype=np.int64))
    indexedarray = ak.layout.IndexedOptionArray64(index, content)

    indexedarray = v1_to_v2(indexedarray)
    assert ak.to_list(indexedarray[3:, 0]) == [None, 0.3]
    assert indexedarray.typetracer[3:, 0].form == indexedarray[3:, 0].form
    assert ak.to_list(indexedarray[3:, 1]) == [None, 3.0]
    assert indexedarray.typetracer[3:, 1].form == indexedarray[3:, 1].form
    assert ak.to_list(indexedarray[3:, ::-1]) == [None, [3.0, 0.3]]
    assert indexedarray.typetracer[3:, ::-1].form == indexedarray[3:, ::-1].form
    assert ak.to_list(indexedarray[3:, [1, 1, 0]]) == [None, [3.0, 3.0, 0.3]]
    assert (
        indexedarray.typetracer[3:, [1, 1, 0]].form == indexedarray[3:, [1, 1, 0]].form
    )
