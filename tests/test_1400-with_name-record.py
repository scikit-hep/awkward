# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    record = ak._v2.with_name(ak._v2.Record({"x": 10.0}), "X")
    assert ak._v2.parameters(record) == {"__record__": "X"}
