# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

to_list = ak._v2.operations.to_list


def test():

    assert isinstance(
        ak._v2.operations.zip({"x": 1, "y": 0, "z": 0}),
        ak._v2.highlevel.Record,
    )
