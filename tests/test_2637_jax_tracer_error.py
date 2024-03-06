# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
from __future__ import annotations

import pytest

import awkward as ak

jax = pytest.importorskip("jax")


def test():
    ak.jax.register_and_check()

    jets = ak.Array(
        [
            [
                {"pt": 1.0, "eta": 1.1, "phi": 0.1, "mass": 0.01},
                {"pt": 2, "eta": 2.2, "phi": 0.2, "mass": 0.02},
            ],
            [
                {"pt": 4.0, "eta": 4.4, "phi": 0.4, "mass": 0.04},
                {"pt": 5.0, "eta": 5.5, "phi": 0.5, "mass": 0.05},
                {"pt": 6.0, "eta": 6.6, "phi": 0.6, "mass": 0.06},
            ],
        ],
        backend="jax",
    )

    def correct_jets(jets, alpha):
        new_pt = jets["pt"] + 25.0 * alpha
        jets["pt"] = new_pt
        return ak.sum(jets["pt"])

    val, grad = jax.value_and_grad(correct_jets, argnums=1)(jets, 0.1)

    assert val == 30.5
    assert grad == 125.0
