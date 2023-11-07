# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np  # noqa: F401
import pytest

import awkward as ak
from awkward._backends.numpy import NumpyBackend
from awkward._backends.typetracer import TypeTracerBackend


def test():
    layout = ak.to_layout({"x": 1, "y": 1})
    assert ak.backend(layout) == "cpu"
    assert layout.backend is NumpyBackend.instance()

    layout_tt = layout.to_backend("typetracer")
    assert ak.backend(layout_tt) == "typetracer"
    assert layout_tt.backend is TypeTracerBackend.instance()

    with pytest.raises(TypeError):
        layout_tt.to_backend("cpu")
