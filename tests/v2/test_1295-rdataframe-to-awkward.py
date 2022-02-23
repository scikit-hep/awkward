# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401
from awkward._v2._connect.rdataframe._from_rdataframe import *


ROOT = pytest.importorskip("ROOT")

def test_from_rdf_to_awkward():
    rdf = ROOT.RDataFrame(100)
    array = rdf.AsAwkward()
    assert array.tolist() == []

    rdf_x = rdf.Define("x", "gRandom->Rndm()")
    array_x = rdf_x.AsAwkward()
    assert (
        str(array_x.form)
        == """{
    "class": "RecordArray",
    "contents": {
        "x": "float64"
    }
}"""
    )

    array_none = rdf_x.AsAwkward(exclude=["x"])
    assert array_none.to_list() == []

def test_vectors():
    list = []
    print(ROOT.is_iterable(list))
