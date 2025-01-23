# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
# ruff: noqa: E402

from __future__ import annotations

import awkward as ak


def test_attrs():
    x = ak.Array([1], attrs={"me": "ta"})
    y = ak.Array([1], attrs={"da": "ta"})

    rec = ak.Array({"x": x, "y": y})
    assert rec.attrs == {"me": "ta", "da": "ta"}

    # overwrite attrs:
    rec = ak.Array({"x": x, "y": y}, attrs={"foo": "bar"})
    assert rec.attrs == {"foo": "bar"}


def test_behaviors():
    behavior = {"foo": object()}
    x = ak.Array([1], behavior=behavior)
    y = ak.Array([1], behavior=behavior)

    rec = ak.Array({"x": x, "y": y})
    assert rec.behavior == behavior

    # overwrite behavior:
    behavior2 = {"bar": object()}
    rec = ak.Array({"x": x, "y": y}, behavior=behavior2)
    assert rec.behavior == behavior2
