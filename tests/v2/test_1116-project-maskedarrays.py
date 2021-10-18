# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2, v1_to_v2_index

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test_bytemaskedarray():
    array = ak.from_iter(
        [[0.0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    mask = ak.layout.Index8(np.array([0, 1, 0, 0], dtype=np.int8))
    maskedarray = ak.layout.ByteMaskedArray(mask, array, valid_when=False)

    maskedarray2 = v1_to_v2(maskedarray)
    mask2 = v1_to_v2_index(mask)

    assert ak.to_list(maskedarray.project()) == ak.to_list(maskedarray2.project())
    assert ak.to_list(maskedarray.project(mask)) == ak.to_list(
        maskedarray2.project(mask2)
    )


def test_bitmaskedarray():
    array = ak.from_iter(
        [[0.0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    mask = ak.layout.IndexU8(np.array([0, 1, 0, 0], dtype=np.uint8))
    maskedarray = ak.layout.BitMaskedArray(
        mask, array, valid_when=False, length=4, lsb_order=True
    )

    maskedarray2 = v1_to_v2(maskedarray)

    assert ak.to_list(maskedarray.project()) == ak.to_list(maskedarray2.project())


def test_unmasked():
    array = ak.from_iter(
        [[0.0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    unmaskedarray = ak.layout.UnmaskedArray(array)

    unmaskedarray2 = v1_to_v2(unmaskedarray)

    assert ak.to_list(unmaskedarray.project()) == ak.to_list(unmaskedarray2.project())


def test_indexed():
    array = ak.Array([1, 2, 3, None, 4, None, None, 5]).layout
    mask = ak.layout.Index8(np.array([0, 1, 0, 0, 1, 0, 1, 1], dtype=np.int8))

    array2 = v1_to_v2(array)
    mask2 = v1_to_v2_index(mask)

    assert ak.to_list(array.project()) == ak.to_list(array2.project())
    assert ak.to_list(array.project(mask)) == ak.to_list(array2.project(mask2))
