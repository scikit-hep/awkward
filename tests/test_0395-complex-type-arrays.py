# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

def test_count_complex():
    content2 = ak.layout.NumpyArray(
        np.array([(1.1+0.1j), 2.2, 3.3, 0.0, 2.2, 0.0, 0.0, 2.2, 0.0, 4.4])
    )
    offsets3 = ak.layout.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak.layout.ListOffsetArray64(offsets3, content2)
    assert ak.to_list(depth1) == [
        [(1.1+0.1j), (2.2+0j), (3.3+0j)],
        [0j, (2.2+0j), 0j],
        [0j, (2.2+0j), 0j, (4.4+0j)]
    ]

    assert ak.to_list(depth1.count(-1)) == [3, 3, 4]
    assert ak.to_list(depth1.count(1)) == [3, 3, 4]

    assert ak.to_list(depth1.count(-2)) == [3, 3, 3, 1]
    assert ak.to_list(depth1.count(0)) == [3, 3, 3, 1]


def test_UnmaskedArray():
    content_float64 = ak.layout.NumpyArray(
        np.array([0.25, 0.5, 3.5, 4.5, 5.5], dtype=np.float64)
    )
    assert content_float64.argmin() == 0
    assert content_float64.argmax() == 4

    array_float64 = ak.layout.UnmaskedArray(content_float64)
    assert ak.to_list(array_float64) == [0.25, 0.5, 3.5, 4.5, 5.5]
    assert str(ak.type(content_float64)) == "float64"
    assert str(ak.type(ak.Array(content_float64))) == "5 * float64"
    assert str(ak.type(array_float64)) == "?float64"
    assert str(ak.type(ak.Array(array_float64))) == "5 * ?float64"

    assert np.can_cast(np.float32, np.float64) == True
    assert np.can_cast(np.float64, np.float32, "unsafe") == True
    assert np.can_cast(np.float64, np.int8, "unsafe") == True
    assert np.can_cast(np.float64, np.complex64, "unsafe") == True
    assert np.can_cast(np.float64, np.complex128, "unsafe") == True
    assert np.can_cast(np.complex64, np.float64, "unsafe") == True
    assert np.can_cast(np.complex128, np.float64, "unsafe") == True

    content_complex64 = ak.values_astype(content_float64, "complex64", highlevel=False)
    array_complex64 = ak.layout.UnmaskedArray(content_complex64)
    assert ak.to_list(array_complex64) == [0.25, 0.5, 3.5, 4.5, 5.5]
    assert str(ak.type(content_complex64)) == "complex64"
    assert str(ak.type(ak.Array(content_complex64))) == "5 * complex64"
    assert str(ak.type(array_complex64)) == "?complex64"
    assert str(ak.type(ak.Array(array_complex64))) == "5 * ?complex64"

    assert content_complex64.prod() == (10.828125+0j)
    assert content_complex64.min() == (0.25+0j)
    assert content_complex64.max() == (5.5+0j)
    assert content_complex64.argmin() == 0
    assert content_complex64.argmax() == 4
