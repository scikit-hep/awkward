# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak


def test():
    record = ak.with_name(ak.Record({"x": 10.0}), "X")
    assert ak.parameters(record) == {"__record__": "X"}
