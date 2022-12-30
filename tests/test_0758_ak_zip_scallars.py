# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak

to_list = ak.operations.to_list


def test():

    assert isinstance(
        ak.operations.zip({"x": 1, "y": 0, "z": 0}),
        ak.highlevel.Record,
    )
