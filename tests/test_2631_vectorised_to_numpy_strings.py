# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import numpy as np

import awkward as ak


def test_string():
    source = ak.Array(["abc$¢€", "d¢#", "€e¢", "💰💰"])
    result = source.to_numpy(False)
    expected = np.array(["abc$¢€", "d¢#", "€e¢", "💰💰"])
    assert result.dtype == expected.dtype
    np.testing.assert_equal(result, expected)


def test_bytestring():
    source = ak.Array([b"foo", b"bar", b"catastrophic", b"\x03\x07"])
    result = source.to_numpy(False)
    expected = np.array([b"foo", b"bar", b"catastrophic", b"\x03\x07"])
    assert result.dtype == expected.dtype
    np.testing.assert_equal(result, expected)
