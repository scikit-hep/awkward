# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

import awkward as ak


def test_non_empty_string():
    array = ak.Array(["this", "that", "foo", "bar"])
    result = ak.to_numpy(array)
    assert result.dtype == np.dtype("U4")
    assert result.tolist() == array.tolist()


def test_non_empty_bytestring():
    array = ak.Array([b"this", b"that", b"foo", b"bar"])
    result = ak.to_numpy(array)
    assert result.dtype == np.dtype("S4")
    assert result.tolist() == array.tolist()


def test_empty_string():
    array = ak.Array(["this", "that", "foo", "bar"])
    result = ak.to_numpy(array[:0])
    assert result.dtype == np.dtype("U1")
    assert result.tolist() == []


def test_empty_bytestring():
    array = ak.Array([b"this", b"that", b"foo", b"bar"])
    result = ak.to_numpy(array[:0])
    assert result.dtype == np.dtype("S1")
    assert result.tolist() == []
