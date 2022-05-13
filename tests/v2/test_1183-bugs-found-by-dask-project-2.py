# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_example():
    x = ak._v2.operations.from_iter([[1, 2, 3, None], [], [4, 5]])
    y = ak._v2.operations.from_iter([100, 200, 300])

    ttx = ak._v2.highlevel.Array(x.layout.typetracer)
    tty = ak._v2.highlevel.Array(y.layout.typetracer)

    assert (x + y).layout.form == (ttx + tty).layout.form
    assert (x + np.sin(y)).layout.form == (ttx + np.sin(tty)).layout.form

    x = ak._v2.highlevel.Array(
        ak._v2.contents.ListArray(x.layout.starts, x.layout.stops, x.layout.content)
    )
    ttx = ak._v2.highlevel.Array(x.layout.typetracer)

    assert (x + x).layout.form == (ttx + ttx).layout.form
