# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest  # noqa: F401

import awkward as ak
from awkward._backends.typetracer import TypeTracerBackend


def test():
    backend = TypeTracerBackend.instance()
    layout = ak.contents.ListOffsetArray(
        ak.index.Index64(
            backend.index_nplike.asarray([0, 1, 3, 7], dtype=np.dtype("int64"))
        ),
        ak.contents.NumpyArray(backend.nplike.asarray([1, 2, 3, 4, 5, 6, 7])),
    )
    assert layout.to_packed().length == 3
