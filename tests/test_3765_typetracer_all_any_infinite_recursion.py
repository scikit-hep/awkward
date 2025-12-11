# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

from awkward._nplikes.typetracer import TypeTracer, TypeTracerArray


def test_all():
    buffer = TypeTracerArray._new(np.dtype("float32"), (4,))
    nplike = TypeTracer.instance()
    result = nplike.all(buffer, axis=None)
    assert isinstance(result, TypeTracerArray)
    assert result.dtype == np.dtype("bool")
    assert result.shape == ()

    buffer = TypeTracerArray._new(np.dtype("float32"), (4, 3))
    nplike = TypeTracer.instance()
    result = nplike.all(buffer, axis=None)
    assert isinstance(result, TypeTracerArray)
    assert result.dtype == np.dtype("bool")
    assert result.shape == ()


def test_any():
    buffer = TypeTracerArray._new(np.dtype("float32"), (4,))
    nplike = TypeTracer.instance()
    result = nplike.any(buffer, axis=None)
    assert isinstance(result, TypeTracerArray)
    assert result.dtype == np.dtype("bool")
    assert result.shape == ()

    buffer = TypeTracerArray._new(np.dtype("float32"), (4, 3))
    nplike = TypeTracer.instance()
    result = nplike.any(buffer, axis=None)
    assert isinstance(result, TypeTracerArray)
    assert result.dtype == np.dtype("bool")
    assert result.shape == ()
