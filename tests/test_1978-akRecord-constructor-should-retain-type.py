# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    j1 = ak.from_numpy(np.empty(0, np.int32))
    assert str(ak.Record({"d": j1}).type) == "{d: 0 * int32}"
