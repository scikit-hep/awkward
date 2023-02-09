# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak
from awkward._nplikes.shape import unknown_length


def test():
    x = ak.to_regular(
        ak.Array([[[-1, 1.0], [-2, 2], [-3, 3]], [[-4, 4]], []], backend="typetracer"),
        axis=-1,
    )
    result = x[:, :-1]
    assert ak.backend(result) == "typetracer"
    layout = ak.to_layout(result)
    assert layout.length == 3
    assert layout.content.length is unknown_length
    assert layout.content.content.length is unknown_length
    assert layout.content.content.dtype == np.dtype(np.float64)
