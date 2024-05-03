# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest

import awkward as ak

jax = pytest.importorskip("jax")


def test():
    ak.jax.register_and_check()

    array = ak.Array([[1.0, 2.0, 3.0], [], [4.0, 5.0]], backend="jax")

    val_mean, grad_mean = jax.value_and_grad(ak.mean, argnums=0)(array)
    _, grad_sum = jax.value_and_grad(ak.sum, argnums=0)(array)
    val_count, grad_count = jax.value_and_grad(ak.count, argnums=0)(array)

    assert val_mean == 3
    assert ak.all(
        grad_mean == ak.Array([[0.2, 0.2, 0.2], [], [0.2, 0.2]], backend="jax")
    )

    # mean is treated as scaled sum
    assert ak.all(grad_mean == grad_sum / val_count)

    assert val_count == 5
    assert ak.all(
        grad_count == ak.Array([[0.0, 0.0, 0.0], [], [0.0, 0.0]], backend="jax")
    )
