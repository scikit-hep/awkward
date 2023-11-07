# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest  # noqa: F401

import awkward as ak


def test():
    record = ak.Record({"x": 1, "y": 2, "z": 3})
    assert ak.fields(record) == ["x", "y", "z"]
    assert ak.unzip(record) == (1, 2, 3)
