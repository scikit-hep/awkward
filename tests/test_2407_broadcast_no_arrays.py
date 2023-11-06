# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import pytest  # noqa: F401

import awkward as ak


def test():
    result = ak.broadcast_arrays()
    assert result == []
