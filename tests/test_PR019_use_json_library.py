# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import os

import pytest
import numpy

import awkward1

def test_string():
    a = awkward1.fromjson("[[1.1, 2.2, 3], [], [4, 5.5]]")
    assert awkward1.tolist(a) == [[1.1, 2.2, 3.0], [], [4.0, 5.5]]

    with pytest.raises(ValueError):
        awkward1.fromjson("[[1.1, 2.2, 3], [blah], [4, 5.5]]")

def test_file(tmp_path):
    with open(os.path.join(str(tmp_path), "tmp1.json"), "w") as f:
        f.write("[[1.1, 2.2, 3], [], [4, 5.5]]")

    a = awkward1.fromjson(os.path.join(str(tmp_path), "tmp1.json"))
    assert awkward1.tolist(a) == [[1.1, 2.2, 3.0], [], [4.0, 5.5]]

    with pytest.raises(ValueError):
        awkward1.fromjson("nonexistent.json")

    with open(os.path.join(str(tmp_path), "tmp2.json"), "w") as f:
        f.write("[[1.1, 2.2, 3], []], [4, 5.5]]")

    with pytest.raises(ValueError):
        awkward1.fromjson(os.path.join(str(tmp_path), "tmp2.json"))
