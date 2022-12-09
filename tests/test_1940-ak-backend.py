# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak


def test_typetracer():
    array = ak.Array([[0, 1, 2, 3], [8, 9, 10, 11]])
    typetracer = ak.Array(array.layout.to_typetracer())

    with pytest.raises(ValueError, match="internal backend"):
        ak.backend(typetracer)


def test_typetracer_mixed():
    array = ak.Array([[0, 1, 2, 3], [8, 9, 10, 11]])
    typetracer = ak.Array(array.layout.to_typetracer())

    with pytest.raises(ValueError, match="internal backend"):
        ak.backend(typetracer, array)
