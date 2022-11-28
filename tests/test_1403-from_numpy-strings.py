# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test_unicode():
    data = np.array(["this", "that"])
    array = ak.from_numpy(data)
    assert array.to_list() == ["this", "that"]


def test_bytes():
    data = np.array([b"this", b"that"], dtype="S")
    array = ak.from_numpy(data)
    assert array.to_list() == [b"this", b"that"]
