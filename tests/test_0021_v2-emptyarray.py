# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2

def test_getitem():
    a = ak.from_json("[]")
    #FIX ME v2 forms only accepts dict // v1 = <Array [[], [[], []], [[], [], []]] type='3 * var * var * unknown'>
    a = ak.from_json("[[], [[], []], [[], [], []]]")

    a = v1_to_v2(a.layout)

    assert ak.to_list(a[2]) == [[], [], []]

    assert ak.to_list(a[2, 1]) == []
    with pytest.raises(ValueError):
        a[2, 1, 0]
    assert ak.to_list(a[2, 1][()]) == []
    with pytest.raises(IndexError):
        a[2, 1][0]
    assert ak.to_list(a[2, 1][100:200]) == []
    assert ak.to_list(a[2, 1, 100:200]) == []
    assert ak.to_list(a[2, 1][np.array([], dtype=int)]) == []
    assert ak.to_list(a[2, 1, np.array([], dtype=int)]) == []
    with pytest.raises(ValueError):
        a[2, 1, np.array([0], dtype=int)]
    with pytest.raises(IndexError):
        a[2, 1][100:200, 0]
    with pytest.raises(IndexError):
        a[2, 1][100:200, 200:300]
    #FIXME
    # with pytest.raises(ValueError):
    #     a[2, 1][100:200, np.array([], dtype=int)]

    # assert ak.to_list(a[1:, 1:]) == [[[]], [[], []]]
    with pytest.raises(ValueError):
        a[1:, 1:, 0]
