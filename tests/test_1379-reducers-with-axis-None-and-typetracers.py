# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401

import awkward as ak  # noqa: F401


def test():
    x = ak.from_iter([[{"x": 1, "y": 1.1}], [], [{"x": 2, "y": 2.2}]])
    x_tt = ak.Array(x.layout.typetracer)

    assert str(ak.count(x_tt, flatten_records=True)) == "unknown-int64"
    assert str(ak.count_nonzero(x_tt, flatten_records=True)) == "unknown-int64"
    assert str(ak.any(x_tt, flatten_records=True)) == "unknown-bool"
    assert str(ak.all(x_tt, flatten_records=True)) == "unknown-bool"
    assert str(ak.prod(x_tt, flatten_records=True)) == "unknown-float64"
    assert str(ak.sum(x_tt, flatten_records=True)) == "unknown-float64"
    assert str(ak.max(x_tt, flatten_records=True)) == "maybe-unknown-float64"
    assert str(ak.min(x_tt, flatten_records=True)) == "maybe-unknown-float64"
    assert str(ak.argmax(x_tt, flatten_records=True)) == "maybe-unknown-int64"
    assert str(ak.argmin(x_tt, flatten_records=True)) == "maybe-unknown-int64"

    assert str(ak.count(x_tt.x)) == "unknown-int64"
    assert str(ak.count_nonzero(x_tt.x)) == "unknown-int64"
    assert str(ak.any(x_tt.x)) == "unknown-bool"
    assert str(ak.all(x_tt.x)) == "unknown-bool"
    assert str(ak.prod(x_tt.x, flatten_records=True)) == "unknown-int64"
    assert str(ak.prod(x_tt.y, flatten_records=True)) == "unknown-float64"
    assert str(ak.sum(x_tt.x, flatten_records=True)) == "unknown-int64"
    assert str(ak.sum(x_tt.y, flatten_records=True)) == "unknown-float64"
    assert str(ak.max(x_tt.x)) == "maybe-unknown-int64"
    assert str(ak.max(x_tt.y)) == "maybe-unknown-float64"
    assert str(ak.min(x_tt.x)) == "maybe-unknown-int64"
    assert str(ak.min(x_tt.y)) == "maybe-unknown-float64"
    assert str(ak.argmax(x_tt.x)) == "maybe-unknown-int64"
    assert str(ak.argmax(x_tt.y)) == "maybe-unknown-int64"
    assert str(ak.argmin(x_tt.x)) == "maybe-unknown-int64"
    assert str(ak.argmin(x_tt.y)) == "maybe-unknown-int64"

    assert str(ak.mean(x_tt, flatten_records=True)) == "unknown-float64"
    assert str(ak.mean(x_tt.x)) == "unknown-float64"
    assert str(ak.mean(x_tt.y)) == "unknown-float64"
