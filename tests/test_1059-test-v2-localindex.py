# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2, v2_to_v1

def test_listoffsetarray_localindex():
    v1_array = ak.from_iter(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    v2_array = v1_to_v2(v1_array)
    assert ak.to_list(v1_array.localindex(0)) == ak.to_list(v2_to_v1(v2_array.localindex(0)))
    assert ak.to_list(v1_array.localindex(1)) == ak.to_list(v2_to_v1(v2_array.localindex(1)))

    v1_array = ak.from_iter(
        [[[0.0, 1.1, 2.2], [], [3.3, 4.4]], [], [[5.5]], [[6.6, 7.7, 8.8, 9.9]]], highlevel=False
    )
    v2_array = v1_to_v2(v1_array)
    assert ak.to_list(v1_array.localindex(0)) == ak.to_list(v2_to_v1(v2_array.localindex(0)))
    assert ak.to_list(v1_array.localindex(1)) == ak.to_list(v2_to_v1(v2_array.localindex(1)))
    assert ak.to_list(v1_array.localindex(2)) == ak.to_list(v2_to_v1(v2_array.localindex(2)))

def test_regulararray_localindex():
    v1_array = ak.from_numpy(
        np.arange(2 * 3 * 5).reshape(2, 3, 5), regulararray=True, highlevel=False
    )
    v2_array = v1_to_v2(v1_array)
    assert ak.to_list(v1_array.localindex(0)) == ak.to_list(v2_to_v1(v2_array.localindex(0)))
    assert ak.to_list(v1_array.localindex(1)) == ak.to_list(v2_to_v1(v2_array.localindex(1)))
    assert ak.to_list(v1_array.localindex(2)) == ak.to_list(v2_to_v1(v2_array.localindex(2)))

    v1_array = ak.from_numpy(
        np.arange(2 * 3 * 5 * 10).reshape(2, 3, 5, 10), regulararray=True, highlevel=False
    )
    v2_array = v1_to_v2(v1_array)
    assert ak.to_list(v1_array.localindex(0)) == ak.to_list(v2_to_v1(v2_array.localindex(0)))
    assert ak.to_list(v1_array.localindex(1)) == ak.to_list(v2_to_v1(v2_array.localindex(1)))
    assert ak.to_list(v1_array.localindex(2)) == ak.to_list(v2_to_v1(v2_array.localindex(2)))
    assert ak.to_list(v1_array.localindex(3)) == ak.to_list(v2_to_v1(v2_array.localindex(3)))

    v1_array = ak.Array(
        ak.layout.RegularArray(ak.Array([[1, 2, 3], [], [4, 5]]).layout, 0, zeros_length=0)).layout
    v2_array = v1_to_v2(v1_array)
    assert ak.to_list(v1_array.localindex(0)) == ak.to_list(v2_to_v1(v2_array.localindex(0)))
    assert ak.to_list(v1_array.localindex(1)) == ak.to_list(v2_to_v1(v2_array.localindex(1)))
    assert ak.to_list(v1_array.localindex(2)) == ak.to_list(v2_to_v1(v2_array.localindex(2)))

def test_bytemaskedarray_localindex():
    content = ak.from_iter(
        [
            [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
            [],
            [[5.5]],
            [[6.6, 7.7, 8.8, 9.9]],
            [[], [10.0, 11.1, 12.2]],
        ],
        highlevel=False,
    )
    mask = ak.layout.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    v1_array = ak.layout.ByteMaskedArray(mask, content, valid_when=False)
    v2_array = v1_to_v2(v1_array)
    
    assert ak.to_list(v1_array) == [
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
        [],
        None,
        None,
        [[], [10.0, 11.1, 12.2]],
    ]
    assert ak.to_list(v2_to_v1(v2_array.localindex(axis=0))) == [0, 1, 2, 3, 4]
    assert ak.to_list(v2_to_v1(v2_array.localindex(axis=-3))) == [0, 1, 2, 3, 4]
    assert ak.to_list(v2_to_v1(v2_array.localindex(axis=1))) == [[0, 1, 2], [], None, None, [0, 1]]
    assert ak.to_list(v2_to_v1(v2_array.localindex(axis=-2))) == [[0, 1, 2], [], None, None, [0, 1]]
    assert ak.to_list(v2_to_v1(v2_array.localindex(axis=2))) == [
        [[0, 1, 2], [], [0, 1]],
        [],
        None,
        None,
        [[], [0, 1, 2]],
    ]
    assert ak.to_list(v2_to_v1(v2_array.localindex(axis=-1))) == [
        [[0, 1, 2], [], [0, 1]],
        [],
        None,
        None,
        [[], [0, 1, 2]],
    ]