# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    x = ak._v2.from_iter([[{"x": 1, "y": 1.1}], [], [{"x": 2, "y": 2.2}]])
    x_tt = ak._v2.Array(x.layout.typetracer)

    assert str(ak._v2.count(x_tt, flatten_records=True)) == "unknown-int64"
    assert str(ak._v2.count_nonzero(x_tt, flatten_records=True)) == "unknown-int64"
    assert str(ak._v2.any(x_tt, flatten_records=True)) == "unknown-bool"
    assert str(ak._v2.all(x_tt, flatten_records=True)) == "unknown-bool"
    assert str(ak._v2.prod(x_tt, flatten_records=True)) == "unknown-float64"
    assert str(ak._v2.sum(x_tt, flatten_records=True)) == "unknown-float64"
    assert str(ak._v2.max(x_tt, flatten_records=True)) == "maybe-unknown-float64"
    assert str(ak._v2.min(x_tt, flatten_records=True)) == "maybe-unknown-float64"
    assert str(ak._v2.argmax(x_tt, flatten_records=True)) == "maybe-unknown-int64"
    assert str(ak._v2.argmin(x_tt, flatten_records=True)) == "maybe-unknown-int64"

    assert str(ak._v2.count(x_tt.x)) == "unknown-int64"
    assert str(ak._v2.count_nonzero(x_tt.x)) == "unknown-int64"
    assert str(ak._v2.any(x_tt.x)) == "unknown-bool"
    assert str(ak._v2.all(x_tt.x)) == "unknown-bool"
    assert str(ak._v2.prod(x_tt.x, flatten_records=True)) == "unknown-int64"
    assert str(ak._v2.prod(x_tt.y, flatten_records=True)) == "unknown-float64"
    assert str(ak._v2.sum(x_tt.x, flatten_records=True)) == "unknown-int64"
    assert str(ak._v2.sum(x_tt.y, flatten_records=True)) == "unknown-float64"
    assert str(ak._v2.max(x_tt.x)) == "maybe-unknown-int64"
    assert str(ak._v2.max(x_tt.y)) == "maybe-unknown-float64"
    assert str(ak._v2.min(x_tt.x)) == "maybe-unknown-int64"
    assert str(ak._v2.min(x_tt.y)) == "maybe-unknown-float64"
    assert str(ak._v2.argmax(x_tt.x)) == "maybe-unknown-int64"
    assert str(ak._v2.argmax(x_tt.y)) == "maybe-unknown-int64"
    assert str(ak._v2.argmin(x_tt.x)) == "maybe-unknown-int64"
    assert str(ak._v2.argmin(x_tt.y)) == "maybe-unknown-int64"

    assert str(ak._v2.mean(x_tt, flatten_records=True)) == "unknown-float64"
    assert str(ak._v2.mean(x_tt.x)) == "unknown-float64"
    assert str(ak._v2.mean(x_tt.y)) == "unknown-float64"
