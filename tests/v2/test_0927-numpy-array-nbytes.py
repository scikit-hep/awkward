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


def test_NumpyArray():
    np_data = np.random.random(size=(4, 100 * 1024 * 1024 // 8 // 4))

    identifier = ak._v2.identifier.Identifier.zeros(
        123, {1: "one", 2: "two"}, 5, 10, np, np.int64
    )

    largest = {0: 0}
    identifier._nbytes_part(largest)
    assert sum(largest.values()) == 8 * 5 * 10

    array = ak._v2.contents.numpyarray.NumpyArray(np_data, identifier)
    assert array.nbytes() == np_data.nbytes + 8 * 5 * 10
