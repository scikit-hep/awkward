# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import awkward as ak  # noqa: F401
import numpy as np


def test_unicode():
    data = np.array(["this", "that"])
    array = ak._v2.from_numpy(data)
    assert array.to_list() == ["this", "that"]


def test_bytes():
    data = np.array([b"this", b"that"], dtype="S")
    array = ak._v2.from_numpy(data)
    assert array.to_list() == [b"this", b"that"]
