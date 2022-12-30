# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import re

import awkward as ak


def test_simple():
    assert re.match(r"3 \* u?int\d+", str(ak.type([1, 2, 3])))


def test_complex():
    assert re.match(
        r"2 \* \{x: union\[var \* \?u?int\d+, float\d+\]\}",
        str(ak.type([{"x": [None, 10]}, {"x": 40.0}])),
    )
