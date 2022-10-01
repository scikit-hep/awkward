# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

jax = pytest.importorskip("jax")
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

ak.jax.register_and_check()


class GradBehavior(ak.Array):
    ...


def test():
    def square_fifth_entry(x):
        return x[4] ** 2

    primal = ak.Array(np.arange(10, dtype=np.float64), backend="jax")
    tangent = ak.Array(np.arange(10, dtype=np.float64), backend="jax")

    behavior = {"grad": GradBehavior}
    primal_grad = ak.with_parameter(primal, "__array__", "grad", behavior=behavior)
    tangent_grad = ak.with_parameter(tangent, "__array__", "grad", behavior=behavior)
    value_jvp_grad, jvp_grad_grad = jax.jvp(
        square_fifth_entry, (primal_grad,), (tangent_grad,)
    )

    assert value_jvp_grad == pytest.approx(16.0)
    assert jvp_grad_grad == pytest.approx(32.0)
