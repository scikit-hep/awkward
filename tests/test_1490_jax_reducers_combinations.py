# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import numpy.testing
import pytest

import awkward as ak

jax = pytest.importorskip("jax")
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

ak.jax.register_and_check()

# #### ak.contents.NumpyArray ####


test_regulararray = ak.Array(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], backend="jax"
)
test_regulararray_tangent = ak.Array(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], backend="jax"
)

test_regulararray_jax = jax.numpy.array(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64
)
test_regulararray_tangent_jax = jax.numpy.array(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64
)


@pytest.mark.parametrize("axis", [0, 1, None])
@pytest.mark.parametrize("func_ak", [ak.sum, ak.prod, ak.min, ak.max])
def test_reducer(func_ak, axis):
    func_jax = getattr(jax.numpy, func_ak.__name__)

    def func_ak_with_axis(x):
        return func_ak(x, axis=axis)

    def func_jax_with_axis(x):
        return func_jax(x, axis=axis)

    value_jvp, jvp_grad = jax.jvp(
        func_ak_with_axis, (test_regulararray,), (test_regulararray_tangent,)
    )
    value_jvp_jax, jvp_grad_jax = jax.jvp(
        func_jax_with_axis, (test_regulararray_jax,), (test_regulararray_tangent_jax,)
    )

    value_vjp, vjp_func = jax.vjp(func_ak_with_axis, test_regulararray)
    value_vjp_jax, vjp_func_jax = jax.vjp(func_jax_with_axis, test_regulararray_jax)

    numpy.testing.assert_allclose(
        ak.to_list(value_jvp), value_jvp_jax.tolist(), rtol=1e-9, atol=np.inf
    )
    numpy.testing.assert_allclose(
        ak.to_list(value_vjp), value_vjp_jax.tolist(), rtol=1e-9, atol=np.inf
    )
    numpy.testing.assert_allclose(
        ak.to_list(jvp_grad), jvp_grad_jax.tolist(), rtol=1e-9, atol=np.inf
    )
    numpy.testing.assert_allclose(
        ak.to_list(vjp_func(value_vjp)[0]),
        (vjp_func_jax(value_vjp_jax)[0]).tolist(),
        rtol=1e-9,
        atol=np.inf,
    )


@pytest.mark.parametrize("axis", [None])
@pytest.mark.parametrize("func_ak", [ak.any, ak.all])
def test_bool_returns(func_ak, axis):
    func_jax = getattr(jax.numpy, func_ak.__name__)

    def func_ak_with_axis(x):
        return func_ak(x, axis=axis)

    def func_jax_with_axis(x):
        return func_jax(x, axis=axis)

    value_jvp, jvp_grad = jax.jvp(
        func_ak_with_axis, (test_regulararray,), (test_regulararray_tangent,)
    )
    value_jvp_jax, jvp_grad_jax = jax.jvp(
        func_jax_with_axis, (test_regulararray_jax,), (test_regulararray_tangent_jax,)
    )

    value_vjp, vjp_func = jax.vjp(func_ak_with_axis, test_regulararray)
    value_vjp_jax, vjp_func_jax = jax.vjp(func_jax_with_axis, test_regulararray_jax)

    assert jvp_grad.dtype == jvp_grad_jax.dtype

    assert value_jvp.tolist() == value_jvp_jax.tolist()
    assert value_vjp.tolist() == value_vjp_jax.tolist()

    numpy.testing.assert_allclose(
        ak.to_list(vjp_func(value_vjp)[0]),
        (vjp_func_jax(value_vjp_jax)[0]).tolist(),
        rtol=1e-9,
        atol=np.inf,
    )


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("func_ak", [ak.any, ak.all])
def test_bool_raises(func_ak, axis):
    def func_with_axis(x):
        return func_ak(x, axis=axis)

    with pytest.raises(
        TypeError, match=".*Make sure that you are not computing the derivative.*"
    ):
        jax.jvp(func_with_axis, (test_regulararray,), (test_regulararray_tangent,))
