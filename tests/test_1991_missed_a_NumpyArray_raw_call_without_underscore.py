# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak


def test():
    assert ak.Record({"x": "hello"})["x"] == "hello"
    assert ak.Record({"x": b"hello"})["x"] == b"hello"

    assert ak.Array([{"x": "hello"}])[0, "x"] == "hello"
    assert ak.Array([{"x": b"hello"}])[0, "x"] == b"hello"
