# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest  # noqa: F401

import awkward as ak


def test():
    y = ak.from_iter([[{"x": 1, "y": 1.1}], [], [{"x": 2, "y": 2.2}]])
    y_tt = ak.Array(y.layout.to_typetracer())
    x_tt = ak.ravel(y_tt)

    assert str(ak.count(x_tt)) == "##"
    assert str(ak.count_nonzero(x_tt)) == "##"
    assert str(ak.any(x_tt)) == "##"
    assert str(ak.all(x_tt)) == "##"
    assert str(ak.prod(x_tt)) == "##"
    assert str(ak.sum(x_tt)) == "##"
    assert str(ak.max(x_tt)) == "?##"
    assert str(ak.min(x_tt)) == "?##"
    assert str(ak.argmax(x_tt)) == "?##"
    assert str(ak.argmin(x_tt)) == "?##"

    assert str(ak.count(y_tt.x)) == "##"
    assert str(ak.count_nonzero(y_tt.x)) == "##"
    assert str(ak.any(y_tt.x)) == "##"
    assert str(ak.all(y_tt.x)) == "##"
    assert str(ak.prod(y_tt.x)) == "##"
    assert str(ak.prod(y_tt.y)) == "##"
    assert str(ak.sum(y_tt.x)) == "##"
    assert str(ak.sum(y_tt.y)) == "##"
    assert str(ak.max(y_tt.x)) == "?##"
    assert str(ak.max(y_tt.y)) == "?##"
    assert str(ak.min(y_tt.x)) == "?##"
    assert str(ak.min(y_tt.y)) == "?##"
    assert str(ak.argmax(y_tt.x)) == "?##"
    assert str(ak.argmax(y_tt.y)) == "?##"
    assert str(ak.argmin(y_tt.x)) == "?##"
    assert str(ak.argmin(y_tt.y)) == "?##"

    assert str(ak.mean(x_tt)) == "##"
    assert str(ak.mean(y_tt.x)) == "##"
    assert str(ak.mean(y_tt.y)) == "##"
