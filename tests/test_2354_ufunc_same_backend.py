# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest

import awkward as ak


def test():
    array = ak.Array([[0, 1, 2, 3], [8, 9, 10, 11]], backend="typetracer")
    other = ak.Array([1, 2], backend="cpu")
    result = array + other
    assert ak.backend(result) == "typetracer"


def test_mixed():
    pytest.importorskip("jax")
    ak.jax.register_and_check()

    array = ak.Array([[0, 1, 2, 3], [8, 9, 10, 11]], backend="cpu")
    other = ak.Array([1, 2], backend="jax")
    with pytest.raises(ValueError):
        array + other
