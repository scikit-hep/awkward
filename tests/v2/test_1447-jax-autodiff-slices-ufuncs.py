# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak
import numpy as np
import pytest


jax = pytest.importorskip("jax")
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

# #### ak.layout.NumpyArray ####

test_numpyarray = ak._v2.Array(np.arange(10, dtype=np.float64), backend="jax")
test_numpyarray_tangent = ak._v2.Array(np.arange(10, dtype=np.float64), backend="jax")

test_numpyarray_jax = jax.numpy.arange(10, dtype=np.float64)
test_numpyarray_tangent_jax = jax.numpy.arange(10, dtype=np.float64)


def test_numpyarray_grad_1():
    def func_numpyarray_1(x):
        return x[4] ** 2

    value_jvp, jvp_grad = jax.jvp(
        func_numpyarray_1, (test_numpyarray,), (test_numpyarray_tangent,)
    )
    value_jvp_jax, jvp_grad_jax = jax.jvp(
        func_numpyarray_1, (test_numpyarray_jax,), (test_numpyarray_tangent_jax,)
    )

    value_vjp, vjp_func = jax.vjp(func_numpyarray_1, test_numpyarray)
    value_vjp_jax, vjp_func_jax = jax.vjp(func_numpyarray_1, test_numpyarray_jax)

    assert value_jvp == value_jvp_jax
    assert value_vjp == value_vjp_jax
    assert jvp_grad == jvp_grad_jax
    assert (
        ak._v2.to_list(vjp_func(value_vjp)[0])
        == (vjp_func_jax(value_vjp_jax)[0]).tolist()
    )


def test_numpyarray_grad_2():
    def func_numpyarray_2(x):
        return x[2:5] ** 2 + x[1:4] ** 2

    value_jvp, jvp_grad = jax.jvp(
        func_numpyarray_2, (test_numpyarray,), (test_numpyarray_tangent,)
    )
    value_jvp_jax, jvp_grad_jax = jax.jvp(
        func_numpyarray_2, (test_numpyarray_jax,), (test_numpyarray_tangent_jax,)
    )

    value_vjp, vjp_func = jax.vjp(func_numpyarray_2, test_numpyarray)
    value_vjp_jax, vjp_func_jax = jax.vjp(func_numpyarray_2, test_numpyarray_jax)

    assert ak._v2.to_list(value_jvp) == value_jvp_jax.tolist()
    assert ak._v2.to_list(value_vjp) == value_vjp_jax.tolist()
    assert ak._v2.to_list(jvp_grad) == jvp_grad_jax.tolist()
    assert (
        ak._v2.to_list(vjp_func(value_vjp)[0])
        == (vjp_func_jax(value_vjp_jax)[0]).tolist()
    )


def test_numpyarray_grad_3():
    def func_numpyarray_3(x):
        return x[::-1]

    value_jvp, jvp_grad = jax.jvp(
        func_numpyarray_3, (test_numpyarray,), (test_numpyarray_tangent,)
    )
    value_jvp_jax, jvp_grad_jax = jax.jvp(
        func_numpyarray_3, (test_numpyarray_jax,), (test_numpyarray_tangent_jax,)
    )

    value_vjp, vjp_func = jax.vjp(func_numpyarray_3, test_numpyarray)
    value_vjp_jax, vjp_func_jax = jax.vjp(func_numpyarray_3, test_numpyarray_jax)

    assert ak._v2.to_list(value_jvp) == value_jvp_jax.tolist()
    assert ak._v2.to_list(value_vjp) == value_vjp_jax.tolist()
    assert ak._v2.to_list(jvp_grad) == jvp_grad_jax.tolist()
    assert (
        ak._v2.to_list(vjp_func(value_vjp)[0])
        == (vjp_func_jax(value_vjp_jax)[0]).tolist()
    )


def test_numpyarray_grad_4():
    def func_numpyarray_4(x):
        return x[2:5] ** 2 * x[1:4] ** 2

    value_jvp, jvp_grad = jax.jvp(
        func_numpyarray_4, (test_numpyarray,), (test_numpyarray_tangent,)
    )
    value_jvp_jax, jvp_grad_jax = jax.jvp(
        func_numpyarray_4, (test_numpyarray_jax,), (test_numpyarray_tangent_jax,)
    )

    value_vjp, vjp_func = jax.vjp(func_numpyarray_4, test_numpyarray)
    value_vjp_jax, vjp_func_jax = jax.vjp(func_numpyarray_4, test_numpyarray_jax)

    assert ak._v2.to_list(value_jvp) == value_jvp_jax.tolist()
    assert ak._v2.to_list(value_vjp) == value_vjp_jax.tolist()
    assert ak._v2.to_list(jvp_grad) == jvp_grad_jax.tolist()
    assert (
        ak._v2.to_list(vjp_func(value_vjp)[0])
        == (vjp_func_jax(value_vjp_jax)[0]).tolist()
    )


test_regulararray = ak._v2.Array(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], backend="jax"
)
test_regulararray_tangent = ak._v2.Array(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], backend="jax"
)

test_regulararray_jax = jax.numpy.array(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64
)
test_regulararray_tangent_jax = jax.numpy.array(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64
)


def test_regular_array_1():
    def func_regulararray_1(x):
        return x[2] * 2

    value_jvp, jvp_grad = jax.jvp(
        func_regulararray_1, (test_regulararray,), (test_regulararray_tangent,)
    )
    value_jvp_jax, jvp_grad_jax = jax.jvp(
        func_regulararray_1, (test_regulararray_jax,), (test_regulararray_tangent_jax,)
    )

    value_vjp, vjp_func = jax.vjp(func_regulararray_1, test_regulararray)
    value_vjp_jax, vjp_func_jax = jax.vjp(func_regulararray_1, test_regulararray_jax)

    assert ak._v2.to_list(value_jvp) == value_jvp_jax.tolist()
    assert ak._v2.to_list(value_vjp) == value_vjp_jax.tolist()
    assert ak._v2.to_list(jvp_grad) == jvp_grad_jax.tolist()
    assert (
        ak._v2.to_list(vjp_func(value_vjp)[0])
        == (vjp_func_jax(value_vjp_jax)[0]).tolist()
    )


def test_regular_array_2():
    def func_regulararray_2(x):
        return x * x

    value_jvp, jvp_grad = jax.jvp(
        func_regulararray_2, (test_regulararray,), (test_regulararray_tangent,)
    )
    value_jvp_jax, jvp_grad_jax = jax.jvp(
        func_regulararray_2, (test_regulararray_jax,), (test_regulararray_tangent_jax,)
    )

    value_vjp, vjp_func = jax.vjp(func_regulararray_2, test_regulararray)
    value_vjp_jax, vjp_func_jax = jax.vjp(func_regulararray_2, test_regulararray_jax)

    assert ak._v2.to_list(value_jvp) == value_jvp_jax.tolist()
    assert ak._v2.to_list(value_vjp) == value_vjp_jax.tolist()
    assert ak._v2.to_list(jvp_grad) == jvp_grad_jax.tolist()
    assert (
        ak._v2.to_list(vjp_func(value_vjp)[0])
        == (vjp_func_jax(value_vjp_jax)[0]).tolist()
    )


def test_regular_array_3():
    def func_regular_array_3(x):
        return x[0, 0] * x[2, 1]

    value_jvp, jvp_grad = jax.jvp(
        func_regular_array_3, (test_regulararray,), (test_regulararray_tangent,)
    )
    value_jvp_jax, jvp_grad_jax = jax.jvp(
        func_regular_array_3, (test_regulararray_jax,), (test_regulararray_tangent_jax,)
    )

    value_vjp, vjp_func = jax.vjp(func_regular_array_3, test_regulararray)
    value_vjp_jax, vjp_func_jax = jax.vjp(func_regular_array_3, test_regulararray_jax)

    assert ak._v2.to_list(value_jvp) == value_jvp_jax.tolist()
    assert ak._v2.to_list(value_vjp) == value_vjp_jax.tolist()
    assert ak._v2.to_list(jvp_grad) == jvp_grad_jax.tolist()
    assert (
        ak._v2.to_list(vjp_func(value_vjp)[0])
        == (vjp_func_jax(value_vjp_jax)[0]).tolist()
    )


def test_regular_array_4():
    def func_regular_array_4(x):
        return x[::-1] ** 2

    value_jvp, jvp_grad = jax.jvp(
        func_regular_array_4, (test_regulararray,), (test_regulararray_tangent,)
    )
    value_jvp_jax, jvp_grad_jax = jax.jvp(
        func_regular_array_4, (test_regulararray_jax,), (test_regulararray_tangent_jax,)
    )

    value_vjp, vjp_func = jax.vjp(func_regular_array_4, test_regulararray)
    value_vjp_jax, vjp_func_jax = jax.vjp(func_regular_array_4, test_regulararray_jax)

    assert ak._v2.to_list(value_jvp) == value_jvp_jax.tolist()
    assert ak._v2.to_list(value_vjp) == value_vjp_jax.tolist()
    assert ak._v2.to_list(jvp_grad) == jvp_grad_jax.tolist()
    assert (
        ak._v2.to_list(vjp_func(value_vjp)[0])
        == (vjp_func_jax(value_vjp_jax)[0]).tolist()
    )


def test_regular_array_5():
    def func_regular_array_5(x):
        return 2 * x[:-1]

    value_jvp, jvp_grad = jax.jvp(
        func_regular_array_5, (test_regulararray,), (test_regulararray_tangent,)
    )
    value_jvp_jax, jvp_grad_jax = jax.jvp(
        func_regular_array_5, (test_regulararray_jax,), (test_regulararray_tangent_jax,)
    )

    value_vjp, vjp_func = jax.vjp(func_regular_array_5, test_regulararray)
    value_vjp_jax, vjp_func_jax = jax.vjp(func_regular_array_5, test_regulararray_jax)

    assert ak._v2.to_list(value_jvp) == value_jvp_jax.tolist()
    assert ak._v2.to_list(value_vjp) == value_vjp_jax.tolist()
    assert ak._v2.to_list(jvp_grad) == jvp_grad_jax.tolist()
    assert (
        ak._v2.to_list(vjp_func(value_vjp)[0])
        == (vjp_func_jax(value_vjp_jax)[0]).tolist()
    )


def test_regular_array_6():
    def func_regular_array_6(x):
        return x[0][0] * x[2][1]

    value_jvp, jvp_grad = jax.jvp(
        func_regular_array_6, (test_regulararray,), (test_regulararray_tangent,)
    )
    value_jvp_jax, jvp_grad_jax = jax.jvp(
        func_regular_array_6, (test_regulararray_jax,), (test_regulararray_tangent_jax,)
    )

    value_vjp, vjp_func = jax.vjp(func_regular_array_6, test_regulararray)
    value_vjp_jax, vjp_func_jax = jax.vjp(func_regular_array_6, test_regulararray_jax)

    assert ak._v2.to_list(value_jvp) == value_jvp_jax.tolist()
    assert ak._v2.to_list(value_vjp) == value_vjp_jax.tolist()
    assert ak._v2.to_list(jvp_grad) == jvp_grad_jax.tolist()
    assert (
        ak._v2.to_list(vjp_func(value_vjp)[0])
        == (vjp_func_jax(value_vjp_jax)[0]).tolist()
    )
