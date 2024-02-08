# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
from __future__ import annotations

import pytest

import awkward as ak

jax = pytest.importorskip("jax")


def test():
    ak.jax.register_and_check()

    arr = ak.Array([[1.0, 2, 3], [5, 6]], backend="jax")
    grad_arr = ak.Array([[34.0, 34.0, 34.0], [34.0, 34.0]], backend="jax")

    def f(x):
        return ak.sum(ak.sum(x) * x)

    assert ak.all(grad_arr == jax.grad(f)(arr))
