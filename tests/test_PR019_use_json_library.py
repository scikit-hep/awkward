# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import os

import pytest
import numpy

import awkward1

def test_string():
    a = awkward1.fromjson("[[1.1, 2.2, 3], [], [4, 5.5]]")
    assert awkward1.tolist(a) == [[1.1, 2.2, 3.0], [], [4.0, 5.5]]

def test_file(tmp_path):
    with open(os.path.join(tmp_path, "tmp.json"), "w") as f:
        f.write("[[1.1, 2.2, 3], [], [4, 5.5]]")

    a = awkward1.fromjson(os.path.join(tmp_path, "tmp.json"))
    assert awkward1.tolist(a) == [[1.1, 2.2, 3.0], [], [4.0, 5.5]]
