# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test_1d():
    array = ak.from_iter(
        [3.3, 2.2, 5.5, 1.1, 4.4],
        highlevel=False,
    )
    array = v1_to_v2(array)
    assert ak.to_list(array.argmin(axis=0)) == 3
    assert ak.to_list(array.argmax(axis=0)) == 2
    assert ak.to_list(array.count(axis=0)) == 5
    assert ak.to_list(array.count_nonzero(axis=0)) == 5
    assert ak.to_list(array.sum(axis=0)) == 16.5
    assert ak.to_list(array.prod(axis=0)) == 193.26120000000003
    assert ak.to_list(array.any(axis=0)) is True
    assert ak.to_list(array.all(axis=0)) is True
    assert ak.to_list(array.min(axis=0)) == 1.1
    assert ak.to_list(array.max(axis=0)) == 5.5


def test_2d():
    array = ak.from_iter(
        [
            [3.3, 2.2, 5.5, 1.1, 4.4],
            [4.4, 2.2, 1.1, 3.3, 5.5],
            [2.2, 1.1, 4.4, 3.3, 5.5],
        ],
        highlevel=False,
    )
    assert ak.to_list(array.argmin(axis=0)) == [2, 2, 1, 0, 0]
    array = v1_to_v2(array)
    assert ak.to_list(array.argmin(axis=0)) == [2, 2, 1, 0, 0]
    assert array.typetracer.argmin(axis=0).form == array.argmin(axis=0).form
    assert ak.to_list(array.argmin(axis=1)) == [3, 2, 1]
    assert array.typetracer.argmin(axis=1).form == array.argmin(axis=1).form


def test_3d():
    array = ak.from_iter(
        [
            [
                [3.3, 2.2, 5.5, 1.1, 4.4],
                [4.4, 2.2, 1.1, 3.3, 5.5],
                [2.2, 1.1, 4.4, 3.3, 5.5],
            ],
            [
                [-3.3, 2.2, -5.5, 1.1, 4.4],
                [4.4, -2.2, 1.1, -3.3, 5.5],
                [2.2, 1.1, 4.4, 3.3, -5.5],
            ],
        ],
        highlevel=False,
    )
    array = v1_to_v2(array)

    assert ak.to_list(array.argmin(axis=0)) == [
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]
    assert ak.to_list(array.argmin(axis=1)) == [[2, 2, 1, 0, 0], [0, 1, 0, 1, 2]]
    assert array.typetracer.argmin(axis=1).form == array.argmin(axis=1).form
    assert ak.to_list(array.argmin(axis=2)) == [[3, 2, 1], [2, 3, 4]]
    assert array.typetracer.argmin(axis=2).form == array.argmin(axis=2).form
    assert ak.to_list(array.argmin(axis=-1)) == [[3, 2, 1], [2, 3, 4]]
    assert array.typetracer.argmin(axis=-1).form == array.argmin(axis=-1).form
    assert ak.to_list(array.argmin(axis=-2)) == [[2, 2, 1, 0, 0], [0, 1, 0, 1, 2]]
    assert array.typetracer.argmin(axis=-2).form == array.argmin(axis=-2).form
    assert ak.to_list(array.argmin(axis=-3)) == [
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]
    assert array.typetracer.argmin(axis=-3).form == array.argmin(axis=-3).form


def test_jagged():
    array = ak.from_iter(
        [[2.2, 1.1, 3.3], [], [4.4, 5.5], [5.5], [-4.4, -5.5, -6.6]], highlevel=False
    )
    v2_array = v1_to_v2(array)
    assert ak.to_list(v2_array.argmin(axis=1)) == [1, None, 0, 0, 2]
    assert v2_array.typetracer.argmin(axis=1).form == v2_array.argmin(axis=1).form

    index2 = ak.layout.Index64(np.array([4, 3, 2, 1, 0], dtype=np.int64))
    array2 = ak.layout.IndexedArray64(index2, array)
    assert ak.to_list(array2.argmin(axis=1)) == [2, 0, 0, None, 1]
    v2_array2 = v1_to_v2(array2)
    assert ak.to_list(v2_array2.argmin(axis=1)) == [2, 0, 0, None, 1]
    assert v2_array2.typetracer.argmin(axis=1).form == v2_array2.argmin(axis=1).form

    index3 = ak.layout.Index64(np.array([4, 3, -1, 4, 0], dtype=np.int64))
    array2 = ak.layout.IndexedOptionArray64(index3, array)
    v2_array2 = v1_to_v2(array2)
    assert ak.to_list(v2_array2.argmin(axis=1)) == [2, 0, None, 2, 1]
    assert v2_array2.typetracer.argmin(axis=1).form == v2_array2.argmin(axis=1).form
    assert ak.to_list(v2_array2.argmin(axis=-1)) == [2, 0, None, 2, 1]
    assert v2_array2.typetracer.argmin(axis=-1).form == v2_array2.argmin(axis=-1).form


def test_missing():
    array = ak.from_iter(
        [[[2.2, 1.1, 3.3]], [[]], [None, None, None], [[-4.4, -5.5, -6.6]]],
        highlevel=False,
    )
    assert ak.to_list(array.argmin(axis=2)) == [[1], [None], [None, None, None], [2]]

    array = v1_to_v2(array)
    assert ak.to_list(array.argmin(axis=2)) == [[1], [None], [None, None, None], [2]]
    assert array.typetracer.argmin(axis=2).form == array.argmin(axis=2).form
