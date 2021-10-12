# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test_empty_array_slice():
    # inspired by PR021::test_getitem
    a = ak.from_json("[[], [[], []], [[], [], []]]")
    a = v1_to_v2(a.layout)
    assert ak.to_list(a[2, 1, np.array([], dtype=int)]) == []
    assert (
        a.typetracer[2, 1, np.array([], dtype=int)].form
        == a[2, 1, np.array([], dtype=int)].form
    )
    assert ak.to_list(a[2, np.array([1], dtype=int), np.array([], dtype=int)]) == [[]]
    assert (
        a.typetracer[2, np.array([1], dtype=int), np.array([], dtype=int)].form
        == a[2, np.array([1], dtype=int), np.array([], dtype=int)].form
    )

    # inspired by PR015::test_deep_numpy
    content = ak.layout.NumpyArray(
        np.array([[0.0, 1.1], [2.2, 3.3], [4.4, 5.5], [6.6, 7.7], [8.8, 9.9]])
    )
    listarray = ak.layout.ListArray64(
        ak.layout.Index64(np.array([0, 3, 3])),
        ak.layout.Index64(np.array([3, 3, 5])),
        content,
    )
    content = v1_to_v2(content)
    listarray = v1_to_v2(listarray)
    assert ak.to_list(listarray[[2, 0, 0, -1], [1, -1, 0, 0], [0, 1, 0, 1]]) == [
        8.8,
        5.5,
        0.0,
        7.7,
    ]
    assert (
        listarray.typetracer[[2, 0, 0, -1], [1, -1, 0, 0], [0, 1, 0, 1]].form
        == listarray[[2, 0, 0, -1], [1, -1, 0, 0], [0, 1, 0, 1]].form
    )
    assert ak.to_list(listarray[2, 1, np.array([], dtype=int)]) == []
    assert (
        listarray.typetracer[2, 1, np.array([], dtype=int)].form
        == listarray[2, 1, np.array([], dtype=int)].form
    )
    assert ak.to_list(listarray[2, 1, []]) == []
    assert listarray.typetracer[2, 1, []].form == listarray[2, 1, []].form
    assert ak.to_list(listarray[2, [1], []]) == []
    assert listarray.typetracer[2, [1], []].form == listarray[2, [1], []].form
    assert ak.to_list(listarray[2, [], []]) == []
    assert listarray.typetracer[2, [], []].form == listarray[2, [], []].form


def test_nonflat_slice():
    array = np.arange(2 * 3 * 5).reshape(2, 3, 5)

    content = ak.layout.NumpyArray(array.reshape(-1))
    inneroffsets = ak.layout.Index64(np.array([0, 5, 10, 15, 20, 25, 30]))
    outeroffsets = ak.layout.Index64(np.array([0, 3, 6]))
    listoffsetarray = ak.layout.ListOffsetArray64(
        outeroffsets, ak.layout.ListOffsetArray64(inneroffsets, content)
    )
    listoffsetarray = v1_to_v2(listoffsetarray)

    assert ak.to_list(
        array[[1, 0, 1, 1, 1, 0], [2, 0, 1, 1, 2, 0], [2, 4, 2, 4, 0, 1]]
    ) == [27, 4, 22, 24, 25, 1]
    assert ak.to_list(
        array[
            [[1, 0], [1, 1], [1, 0]], [[2, 0], [1, 1], [2, 0]], [[2, 4], [2, 4], [0, 1]]
        ]
    ) == [[27, 4], [22, 24], [25, 1]]

    one = listoffsetarray[[1, 0, 1, 1, 1, 0], [2, 0, 1, 1, 2, 0], [2, 4, 2, 4, 0, 1]]
    assert ak.to_list(one) == [27, 4, 22, 24, 25, 1]
    assert (
        listoffsetarray.typetracer[
            [1, 0, 1, 1, 1, 0], [2, 0, 1, 1, 2, 0], [2, 4, 2, 4, 0, 1]
        ].form
        == one.form
    )


@pytest.mark.skip(
    reason="Should be working now that we have toListOffsetArray64, but isn't."
)
def test_nonflat_slice_2():
    array = np.arange(2 * 3 * 5).reshape(2, 3, 5)
    content = ak.layout.NumpyArray(array.reshape(-1))
    inneroffsets = ak.layout.Index64(np.array([0, 5, 10, 15, 20, 25, 30]))
    outeroffsets = ak.layout.Index64(np.array([0, 3, 6]))
    listoffsetarray = ak.layout.ListOffsetArray64(
        outeroffsets, ak.layout.ListOffsetArray64(inneroffsets, content)
    )
    listoffsetarray = v1_to_v2(listoffsetarray)

    two = listoffsetarray[
        [[1, 0], [1, 1], [1, 0]], [[2, 0], [1, 1], [2, 0]], [[2, 4], [2, 4], [0, 1]]
    ]
    assert ak.to_list(two) == [[27, 4], [22, 24], [25, 1]]


def test_newaxis():
    array = np.arange(2 * 3 * 5).reshape(2, 3, 5)

    content = ak.layout.NumpyArray(array.reshape(-1))
    inneroffsets = ak.layout.Index64(np.array([0, 5, 10, 15, 20, 25, 30]))
    outeroffsets = ak.layout.Index64(np.array([0, 3, 6]))
    listoffsetarray = ak.layout.ListOffsetArray64(
        outeroffsets, ak.layout.ListOffsetArray64(inneroffsets, content)
    )
    listoffsetarray = v1_to_v2(listoffsetarray)

    assert ak.to_list(array[:, np.newaxis]) == [
        [[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]],
        [[[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]],
    ]

    assert ak.to_list(listoffsetarray[:, np.newaxis]) == [
        [[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]],
        [[[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]],
    ]
    assert (
        listoffsetarray.typetracer[:, np.newaxis].form
        == listoffsetarray[:, np.newaxis].form
    )
