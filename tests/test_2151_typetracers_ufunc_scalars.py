# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np

import awkward as ak


def test_mixed():
    a = ak._typetracer.TypeTracerArray.from_array(np.array([[1, 2], [3, 4], [5, 6]]))
    b = ak._typetracer.TypeTracerArray.from_array(np.array([2.2]))
    assert isinstance(b[0], ak._typetracer.UnknownScalar)
    c = a + b[0]
    assert c.dtype == np.dtype(np.float64)
    assert c.shape[1] == 2


def test_mixed_maybe_none():
    a = ak._typetracer.TypeTracerArray.from_array(np.array([[1, 2], [3, 4], [5, 6]]))
    b = ak._typetracer.TypeTracerArray.from_array(np.array([2.2]))
    assert isinstance(b[9], ak._typetracer.UnknownScalar)
    c = a + ak._typetracer.MaybeNone(b[0])
    assert c.dtype == np.dtype(np.float64)
    assert c.shape[1] == 2
