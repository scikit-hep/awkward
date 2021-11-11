# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test():
    np_data = np.random.random(size=(4, 100 * 1024 * 1024 // 8 // 4))
    array = ak.from_numpy(np_data, regulararray=False)
    array = v1_to_v2(array.layout)

    assert np_data.nbytes == array.nbytes()


def test_NumpyArray_nbytes():
    np_data = np.random.random(size=(4, 100 * 1024 * 1024 // 8 // 4))

    identifier = ak._v2.identifier.Identifier.zeros(
        123, {1: "one", 2: "two"}, 5, 10, np, np.int64
    )

    largest = {0: 0}
    identifier._nbytes_part(largest)
    assert sum(largest.values()) == 8 * 5 * 10

    array = ak._v2.contents.numpyarray.NumpyArray(np_data, identifier)
    assert array.nbytes() == np_data.nbytes + 8 * 5 * 10


def test_ByteMaskedArray_nbytes():
    content = ak.from_iter(
        [
            [[1.1, 0.0, 2.2], [], [3.3, 4.4]],
            [],
            [[5.5]],
            [[6.6, 9.9, 8.8, 7.7]],
            [[], [12.2, 11.1, 10.0]],
        ],
        highlevel=False,
    )
    mask = ak.layout.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    v1_array = ak.layout.ByteMaskedArray(mask, content, valid_when=False)
    assert v1_array.nbytes == 221
    v2_array = v1_to_v2(v1_array)

    assert v2_array.nbytes() == v1_array.nbytes


def test_BitMaskedArray_nbytes():
    np_array = np.array(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
    )
    np_index = np.array(
        [
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            1,
            0,
            1,
        ],
        dtype=np.uint8,
    )
    v2_array = ak._v2.contents.bitmaskedarray.BitMaskedArray(
        ak._v2.index.Index(np.packbits(np_index)),
        ak._v2.contents.numpyarray.NumpyArray(np_array),
        valid_when=True,
        length=13,
        lsb_order=False,
    )
    assert np_array.nbytes == 112
    assert np_index.nbytes == 13
    assert np.packbits(np_index).nbytes == 2
    assert v2_array.nbytes() == 114
