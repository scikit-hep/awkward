# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test_example():
    x = ak.operations.from_iter([[1, 2, 3, None], [], [4, 5]])
    y = ak.operations.from_iter([100, 200, 300])

    ttx = ak.highlevel.Array(x.layout.to_typetracer())
    tty = ak.highlevel.Array(y.layout.to_typetracer())

    assert (x + y).layout.form == (ttx + tty).layout.form
    assert (x + np.sin(y)).layout.form == (ttx + np.sin(tty)).layout.form

    x = ak.highlevel.Array(
        ak.contents.ListArray(x.layout.starts, x.layout.stops, x.layout.content)
    )
    ttx = ak.highlevel.Array(x.layout.to_typetracer())

    assert (x + x).layout.form == (ttx + ttx).layout.form
