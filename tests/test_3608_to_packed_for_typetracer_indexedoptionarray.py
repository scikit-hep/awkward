# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak


@pytest.mark.parametrize("forget_length", [True, False])
@pytest.mark.parametrize("recursive", [True, False])
def test(forget_length, recursive):
    index = np.array([0, 1, 2])
    content = np.array([1, 2, 3])

    layout = ak.contents.IndexedOptionArray(
        ak.index.Index(index), ak.contents.NumpyArray(content)
    )
    assert (
        layout.to_typetracer(forget_length).to_packed(recursive).form
        == layout.to_packed(recursive).form
    )
