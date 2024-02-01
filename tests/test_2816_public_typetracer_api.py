from __future__ import annotations

import numpy as np

import awkward as ak


def test_unknown_scalar():
    scalar = ak.typetracer.create_unknown_scalar(np.int64)
    assert scalar.dtype == np.dtype(np.int64)
    assert scalar.shape == ()
    assert ak.typetracer.is_unknown_scalar(scalar)

    scalar = ak.typetracer.create_unknown_scalar("int64")
    assert scalar.dtype == np.dtype(np.int64)
    assert scalar.shape == ()
    assert ak.typetracer.is_unknown_scalar(scalar)
