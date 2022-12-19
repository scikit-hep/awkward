# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

jax = pytest.importorskip("jax")
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

ak.jax.register_and_check()

# #### ak.contents.NumpyArray ####

test_numpyarray = ak.Array(np.arange(10, dtype=np.float64), backend="jax")
test_numpyarray_tangent = ak.Array(np.arange(10, dtype=np.float64), backend="jax")

test_numpyarray_jax = jax.numpy.arange(10, dtype=np.float64)
test_numpyarray_tangent_jax = jax.numpy.arange(10, dtype=np.float64)


@pytest.mark.parametrize("axis", [-1, None])
@pytest.mark.parametrize(
    "func_ak", [ak.sum, ak.prod, ak.min, ak.max, ak.any, ak.all, ak.sum]
)
def test_reducer(func_ak, axis):
    func_jax = getattr(jax.numpy, func_ak.__name__)

    def func_ak_with_axis(x):
        return func_ak(x, axis=axis)

    def func_jax_with_axis(x):
        return func_jax(x, axis=axis)

    value_jvp, jvp_grad = jax.jvp(
        func_ak_with_axis, (test_numpyarray,), (test_numpyarray_tangent,)
    )
    value_jvp_jax, jvp_grad_jax = jax.jvp(
        func_jax_with_axis, (test_numpyarray_jax,), (test_numpyarray_tangent_jax,)
    )

    value_vjp, vjp_func = jax.vjp(func_ak_with_axis, test_numpyarray)
    value_vjp_jax, vjp_func_jax = jax.vjp(func_jax_with_axis, test_numpyarray_jax)

    assert value_jvp == value_jvp_jax
    assert value_vjp == value_vjp_jax
    assert jvp_grad == jvp_grad_jax
    assert (
        ak.to_list(vjp_func(value_vjp)[0]) == (vjp_func_jax(value_vjp_jax)[0]).tolist()
    )
