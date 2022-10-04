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


def test_numpyarray_grad_sum_1():
    def func_numpyarray_1(x):
        return ak.sum(x)

    def func_nummpyarray_1_jax(x):
        return jax.numpy.sum(x)

    value_jvp, jvp_grad = jax.jvp(
        func_numpyarray_1, (test_numpyarray,), (test_numpyarray_tangent,)
    )
    value_jvp_jax, jvp_grad_jax = jax.jvp(
        func_nummpyarray_1_jax, (test_numpyarray_jax,), (test_numpyarray_tangent_jax,)
    )

    value_vjp, vjp_func = jax.vjp(func_numpyarray_1, test_numpyarray)
    value_vjp_jax, vjp_func_jax = jax.vjp(func_nummpyarray_1_jax, test_numpyarray_jax)

    assert value_jvp == value_jvp_jax
    assert value_vjp == value_vjp_jax
    assert jvp_grad == jvp_grad_jax
    assert (
        ak.to_list(vjp_func(value_vjp)[0]) == (vjp_func_jax(value_vjp_jax)[0]).tolist()
    )


def test_numpyarray_grad_prod_1():
    def func_numpyarray_1(x):
        return ak.prod(x)

    def func_nummpyarray_1_jax(x):
        return jax.numpy.prod(x)

    value_jvp, jvp_grad = jax.jvp(
        func_numpyarray_1, (test_numpyarray,), (test_numpyarray_tangent,)
    )
    value_jvp_jax, jvp_grad_jax = jax.jvp(
        func_nummpyarray_1_jax, (test_numpyarray_jax,), (test_numpyarray_tangent_jax,)
    )

    value_vjp, vjp_func = jax.vjp(func_numpyarray_1, test_numpyarray)
    value_vjp_jax, vjp_func_jax = jax.vjp(func_nummpyarray_1_jax, test_numpyarray_jax)

    assert value_jvp == value_jvp_jax
    assert value_vjp == value_vjp_jax
    assert jvp_grad == jvp_grad_jax
    assert (
        ak.to_list(vjp_func(value_vjp)[0]) == (vjp_func_jax(value_vjp_jax)[0]).tolist()
    )


def test_numpyarray_grad_max_1():
    def func_numpyarray_1(x):
        return ak.max(x)

    def func_nummpyarray_1_jax(x):
        return jax.numpy.max(x)

    value_jvp, jvp_grad = jax.jvp(
        func_numpyarray_1, (test_numpyarray,), (test_numpyarray_tangent,)
    )
    value_jvp_jax, jvp_grad_jax = jax.jvp(
        func_nummpyarray_1_jax, (test_numpyarray_jax,), (test_numpyarray_tangent_jax,)
    )

    value_vjp, vjp_func = jax.vjp(func_numpyarray_1, test_numpyarray)
    value_vjp_jax, vjp_func_jax = jax.vjp(func_nummpyarray_1_jax, test_numpyarray_jax)

    assert value_jvp == value_jvp_jax
    assert value_vjp == value_vjp_jax
    assert jvp_grad == jvp_grad_jax
    assert (
        ak.to_list(vjp_func(value_vjp)[0]) == (vjp_func_jax(value_vjp_jax)[0]).tolist()
    )


def test_numpyarray_grad_min_1():
    def func_numpyarray_1(x):
        return ak.min(x)

    def func_nummpyarray_1_jax(x):
        return jax.numpy.min(x)

    value_jvp, jvp_grad = jax.jvp(
        func_numpyarray_1, (test_numpyarray,), (test_numpyarray_tangent,)
    )
    value_jvp_jax, jvp_grad_jax = jax.jvp(
        func_nummpyarray_1_jax, (test_numpyarray_jax,), (test_numpyarray_tangent_jax,)
    )

    value_vjp, vjp_func = jax.vjp(func_numpyarray_1, test_numpyarray)
    value_vjp_jax, vjp_func_jax = jax.vjp(func_nummpyarray_1_jax, test_numpyarray_jax)

    assert value_jvp == value_jvp_jax
    assert value_vjp == value_vjp_jax
    assert jvp_grad == jvp_grad_jax
    assert (
        ak.to_list(vjp_func(value_vjp)[0]) == (vjp_func_jax(value_vjp_jax)[0]).tolist()
    )


def test_numpyarray_grad_all_1():
    def func_numpyarray_1(x):
        return ak.all(x)

    def func_nummpyarray_1_jax(x):
        return jax.numpy.all(x)

    value_jvp, jvp_grad = jax.jvp(
        func_numpyarray_1, (test_numpyarray,), (test_numpyarray_tangent,)
    )
    value_jvp_jax, jvp_grad_jax = jax.jvp(
        func_nummpyarray_1_jax, (test_numpyarray_jax,), (test_numpyarray_tangent_jax,)
    )

    value_vjp, vjp_func = jax.vjp(func_numpyarray_1, test_numpyarray)
    value_vjp_jax, vjp_func_jax = jax.vjp(func_nummpyarray_1_jax, test_numpyarray_jax)

    assert value_jvp == value_jvp_jax
    assert value_vjp == value_vjp_jax
    assert jvp_grad == jvp_grad_jax
    assert (
        ak.to_list(vjp_func(value_vjp)[0]) == (vjp_func_jax(value_vjp_jax)[0]).tolist()
    )


def test_numpyarray_grad_any_1():
    def func_numpyarray_1(x):
        return ak.any(x)

    value_jvp, jvp_grad = jax.jvp(
        func_numpyarray_1, (test_numpyarray,), (test_numpyarray_tangent,)
    )
    assert jvp_grad.dtype == np.dtype([("float0", "V")])


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


def test_regular_array_sum_0():
    def func_regulararray_sum_0(x):
        return ak.sum(x, 0)

    def func_regulararray_sum_0_jax(x):
        return jax.numpy.sum(x, 0)

    value_jvp, jvp_grad = jax.jvp(
        func_regulararray_sum_0, (test_regulararray,), (test_regulararray_tangent,)
    )
    value_jvp_jax, jvp_grad_jax = jax.jvp(
        func_regulararray_sum_0_jax,
        (test_regulararray_jax,),
        (test_regulararray_tangent_jax,),
    )

    value_vjp, vjp_func = jax.vjp(func_regulararray_sum_0, test_regulararray)
    value_vjp_jax, vjp_func_jax = jax.vjp(
        func_regulararray_sum_0_jax, test_regulararray_jax
    )

    assert ak.to_list(value_jvp) == value_jvp_jax.tolist()
    assert ak.to_list(value_vjp) == value_vjp_jax.tolist()
    assert ak.to_list(jvp_grad) == jvp_grad_jax.tolist()
    assert (
        ak.to_list(vjp_func(value_vjp)[0]) == (vjp_func_jax(value_vjp_jax)[0]).tolist()
    )


def test_regular_array_sum_1():
    def func_regulararray_sum_1(x):
        return ak.sum(x, 1)

    def func_regulararray_sum_1_jax(x):
        return jax.numpy.sum(x, 1)

    value_jvp, jvp_grad = jax.jvp(
        func_regulararray_sum_1, (test_regulararray,), (test_regulararray_tangent,)
    )
    value_jvp_jax, jvp_grad_jax = jax.jvp(
        func_regulararray_sum_1_jax,
        (test_regulararray_jax,),
        (test_regulararray_tangent_jax,),
    )

    value_vjp, vjp_func = jax.vjp(func_regulararray_sum_1, test_regulararray)
    value_vjp_jax, vjp_func_jax = jax.vjp(
        func_regulararray_sum_1_jax, test_regulararray_jax
    )

    assert ak.to_list(value_jvp) == value_jvp_jax.tolist()
    assert ak.to_list(value_vjp) == value_vjp_jax.tolist()
    assert ak.to_list(jvp_grad) == jvp_grad_jax.tolist()
    assert (
        ak.to_list(vjp_func(value_vjp)[0]) == (vjp_func_jax(value_vjp_jax)[0]).tolist()
    )


def test_regular_array_sum_none():
    def func_regulararray_sum_none(x):
        return ak.sum(x, 1)

    def func_regulararray_sum_none_jax(x):
        return jax.numpy.sum(x, 1)

    value_jvp, jvp_grad = jax.jvp(
        func_regulararray_sum_none, (test_regulararray,), (test_regulararray_tangent,)
    )
    value_jvp_jax, jvp_grad_jax = jax.jvp(
        func_regulararray_sum_none_jax,
        (test_regulararray_jax,),
        (test_regulararray_tangent_jax,),
    )

    value_vjp, vjp_func = jax.vjp(func_regulararray_sum_none, test_regulararray)
    value_vjp_jax, vjp_func_jax = jax.vjp(
        func_regulararray_sum_none_jax, test_regulararray_jax
    )

    assert ak.to_list(value_jvp) == value_jvp_jax.tolist()
    assert ak.to_list(value_vjp) == value_vjp_jax.tolist()
    assert ak.to_list(jvp_grad) == jvp_grad_jax.tolist()
    assert (
        ak.to_list(vjp_func(value_vjp)[0]) == (vjp_func_jax(value_vjp_jax)[0]).tolist()
    )


def test_regular_array_prod_0():
    def func_regulararray_prod_0(x):
        return ak.prod(x, 0)

    def func_regulararray_prod_0_jax(x):
        return jax.numpy.prod(x, 0)

    value_jvp, jvp_grad = jax.jvp(
        func_regulararray_prod_0, (test_regulararray,), (test_regulararray_tangent,)
    )
    value_jvp_jax, jvp_grad_jax = jax.jvp(
        func_regulararray_prod_0_jax,
        (test_regulararray_jax,),
        (test_regulararray_tangent_jax,),
    )

    value_vjp, vjp_func = jax.vjp(func_regulararray_prod_0, test_regulararray)
    value_vjp_jax, vjp_func_jax = jax.vjp(
        func_regulararray_prod_0_jax, test_regulararray_jax
    )

    assert ak.to_list(value_jvp) == pytest.approx(value_jvp_jax.tolist())
    assert ak.to_list(value_vjp) == pytest.approx(value_vjp_jax.tolist())
    assert ak.to_list(jvp_grad) == pytest.approx(jvp_grad_jax.tolist())
    np.testing.assert_array_almost_equal(
        ak.to_list(vjp_func(value_vjp)[0]), vjp_func_jax(value_vjp_jax)[0].tolist()
    )


def test_regular_array_prod_1():
    def func_regulararray_prod_1(x):
        return ak.prod(x, 1)

    def func_regulararray_prod_1_jax(x):
        return jax.numpy.prod(x, 1)

    value_jvp, jvp_grad = jax.jvp(
        func_regulararray_prod_1, (test_regulararray,), (test_regulararray_tangent,)
    )
    value_jvp_jax, jvp_grad_jax = jax.jvp(
        func_regulararray_prod_1_jax,
        (test_regulararray_jax,),
        (test_regulararray_tangent_jax,),
    )

    value_vjp, vjp_func = jax.vjp(func_regulararray_prod_1, test_regulararray)
    value_vjp_jax, vjp_func_jax = jax.vjp(
        func_regulararray_prod_1_jax, test_regulararray_jax
    )

    assert ak.to_list(value_jvp) == pytest.approx(value_jvp_jax.tolist())
    assert ak.to_list(value_vjp) == pytest.approx(value_vjp_jax.tolist())
    assert ak.to_list(jvp_grad) == pytest.approx(jvp_grad_jax.tolist())
    np.testing.assert_array_almost_equal(
        ak.to_list(vjp_func(value_vjp)[0]), vjp_func_jax(value_vjp_jax)[0].tolist()
    )


def test_regular_array_prod_none():
    def func_regulararray_prod_none(x):
        return ak.prod(x)

    def func_regulararray_prod_none_jax(x):
        return jax.numpy.prod(x)

    value_jvp, jvp_grad = jax.jvp(
        func_regulararray_prod_none, (test_regulararray,), (test_regulararray_tangent,)
    )
    value_jvp_jax, jvp_grad_jax = jax.jvp(
        func_regulararray_prod_none_jax,
        (test_regulararray_jax,),
        (test_regulararray_tangent_jax,),
    )

    value_vjp, vjp_func = jax.vjp(func_regulararray_prod_none, test_regulararray)
    value_vjp_jax, vjp_func_jax = jax.vjp(
        func_regulararray_prod_none_jax, test_regulararray_jax
    )

    assert ak.to_list(value_jvp) == pytest.approx(value_jvp_jax.tolist())
    assert ak.to_list(value_vjp) == pytest.approx(value_vjp_jax.tolist())
    assert ak.to_list(jvp_grad) == pytest.approx(jvp_grad_jax.tolist())
    np.testing.assert_array_almost_equal(
        ak.to_list(vjp_func(value_vjp)[0]), vjp_func_jax(value_vjp_jax)[0].tolist()
    )


def test_regular_array_max_0():
    def func_regulararray_max_0(x):
        return ak.max(x, 0)

    def func_regulararray_max_0_jax(x):
        return jax.numpy.max(x, 0)

    value_jvp, jvp_grad = jax.jvp(
        func_regulararray_max_0, (test_regulararray,), (test_regulararray_tangent,)
    )
    value_jvp_jax, jvp_grad_jax = jax.jvp(
        func_regulararray_max_0_jax,
        (test_regulararray_jax,),
        (test_regulararray_tangent_jax,),
    )

    value_vjp, vjp_func = jax.vjp(func_regulararray_max_0, test_regulararray)
    value_vjp_jax, vjp_func_jax = jax.vjp(
        func_regulararray_max_0_jax, test_regulararray_jax
    )

    assert ak.to_list(value_jvp) == value_jvp_jax.tolist()
    assert ak.to_list(value_vjp) == value_vjp_jax.tolist()
    assert ak.to_list(jvp_grad) == jvp_grad_jax.tolist()
    assert (
        ak.to_list(vjp_func(value_vjp)[0]) == (vjp_func_jax(value_vjp_jax)[0]).tolist()
    )


def test_regular_array_max_1():
    def func_regulararray_max_1(x):
        return ak.max(x, 1)

    def func_regulararray_max_1_jax(x):
        return jax.numpy.max(x, 1)

    value_jvp, jvp_grad = jax.jvp(
        func_regulararray_max_1, (test_regulararray,), (test_regulararray_tangent,)
    )
    value_jvp_jax, jvp_grad_jax = jax.jvp(
        func_regulararray_max_1_jax,
        (test_regulararray_jax,),
        (test_regulararray_tangent_jax,),
    )

    value_vjp, vjp_func = jax.vjp(func_regulararray_max_1, test_regulararray)
    value_vjp_jax, vjp_func_jax = jax.vjp(
        func_regulararray_max_1_jax, test_regulararray_jax
    )

    assert ak.to_list(value_jvp) == value_jvp_jax.tolist()
    assert ak.to_list(value_vjp) == value_vjp_jax.tolist()
    assert ak.to_list(jvp_grad) == jvp_grad_jax.tolist()
    assert (
        ak.to_list(vjp_func(value_vjp)[0]) == (vjp_func_jax(value_vjp_jax)[0]).tolist()
    )


def test_regular_array_max_none():
    def func_regulararray_max_none(x):
        return ak.max(x)

    def func_regulararray_max_none_jax(x):
        return jax.numpy.max(x)

    value_jvp, jvp_grad = jax.jvp(
        func_regulararray_max_none, (test_regulararray,), (test_regulararray_tangent,)
    )
    value_jvp_jax, jvp_grad_jax = jax.jvp(
        func_regulararray_max_none_jax,
        (test_regulararray_jax,),
        (test_regulararray_tangent_jax,),
    )

    value_vjp, vjp_func = jax.vjp(func_regulararray_max_none, test_regulararray)
    value_vjp_jax, vjp_func_jax = jax.vjp(
        func_regulararray_max_none_jax, test_regulararray_jax
    )

    assert ak.to_list(value_jvp) == value_jvp_jax.tolist()
    assert ak.to_list(value_vjp) == value_vjp_jax.tolist()
    assert ak.to_list(jvp_grad) == jvp_grad_jax.tolist()
    assert (
        ak.to_list(vjp_func(value_vjp)[0]) == (vjp_func_jax(value_vjp_jax)[0]).tolist()
    )


def test_regular_array_min_0():
    def func_regulararray_min_0(x):
        return ak.min(x, 0)

    def func_regulararray_min_0_jax(x):
        return jax.numpy.min(x, 0)

    value_jvp, jvp_grad = jax.jvp(
        func_regulararray_min_0, (test_regulararray,), (test_regulararray_tangent,)
    )
    value_jvp_jax, jvp_grad_jax = jax.jvp(
        func_regulararray_min_0_jax,
        (test_regulararray_jax,),
        (test_regulararray_tangent_jax,),
    )

    value_vjp, vjp_func = jax.vjp(func_regulararray_min_0, test_regulararray)
    value_vjp_jax, vjp_func_jax = jax.vjp(
        func_regulararray_min_0_jax, test_regulararray_jax
    )

    assert ak.to_list(value_jvp) == value_jvp_jax.tolist()
    assert ak.to_list(value_vjp) == value_vjp_jax.tolist()
    assert ak.to_list(jvp_grad) == jvp_grad_jax.tolist()
    assert (
        ak.to_list(vjp_func(value_vjp)[0]) == (vjp_func_jax(value_vjp_jax)[0]).tolist()
    )


def test_regular_array_min_1():
    def func_regulararray_min_1(x):
        return ak.min(x, 1)

    def func_regulararray_min_1_jax(x):
        return jax.numpy.min(x, 1)

    value_jvp, jvp_grad = jax.jvp(
        func_regulararray_min_1, (test_regulararray,), (test_regulararray_tangent,)
    )
    value_jvp_jax, jvp_grad_jax = jax.jvp(
        func_regulararray_min_1_jax,
        (test_regulararray_jax,),
        (test_regulararray_tangent_jax,),
    )

    value_vjp, vjp_func = jax.vjp(func_regulararray_min_1, test_regulararray)
    value_vjp_jax, vjp_func_jax = jax.vjp(
        func_regulararray_min_1_jax, test_regulararray_jax
    )

    assert ak.to_list(value_jvp) == value_jvp_jax.tolist()
    assert ak.to_list(value_vjp) == value_vjp_jax.tolist()
    assert ak.to_list(jvp_grad) == jvp_grad_jax.tolist()
    assert (
        ak.to_list(vjp_func(value_vjp)[0]) == (vjp_func_jax(value_vjp_jax)[0]).tolist()
    )


def test_regular_array_min_none():
    def func_regulararray_min_none(x):
        return ak.min(x)

    def func_regulararray_min_none_jax(x):
        return jax.numpy.min(x)

    value_jvp, jvp_grad = jax.jvp(
        func_regulararray_min_none, (test_regulararray,), (test_regulararray_tangent,)
    )
    value_jvp_jax, jvp_grad_jax = jax.jvp(
        func_regulararray_min_none_jax,
        (test_regulararray_jax,),
        (test_regulararray_tangent_jax,),
    )

    value_vjp, vjp_func = jax.vjp(func_regulararray_min_none, test_regulararray)
    value_vjp_jax, vjp_func_jax = jax.vjp(
        func_regulararray_min_none_jax, test_regulararray_jax
    )

    assert ak.to_list(value_jvp) == value_jvp_jax.tolist()
    assert ak.to_list(value_vjp) == value_vjp_jax.tolist()
    assert ak.to_list(jvp_grad) == jvp_grad_jax.tolist()
    assert (
        ak.to_list(vjp_func(value_vjp)[0]) == (vjp_func_jax(value_vjp_jax)[0]).tolist()
    )


def test_regular_array_all_0():
    def func_regulararray_all_0(x):
        return ak.all(x, 0)

    with pytest.raises(
        TypeError, match=".*Make sure that you are not computing the derivative.*"
    ):
        jax.jvp(
            func_regulararray_all_0, (test_regulararray,), (test_regulararray_tangent,)
        )


def test_regular_array_all_1():
    def func_regulararray_all_1(x):
        return ak.all(x, 1)

    with pytest.raises(
        TypeError, match=".*Make sure that you are not computing the derivative.*"
    ):
        jax.jvp(
            func_regulararray_all_1, (test_regulararray,), (test_regulararray_tangent,)
        )


def test_regular_array_all_none():
    def func_regulararray_all_none(x):
        return ak.all(x)

    def func_regulararray_all_none_jax(x):
        return jax.numpy.all(x)

    value_jvp, jvp_grad = jax.jvp(
        func_regulararray_all_none, (test_regulararray,), (test_regulararray_tangent,)
    )
    value_jvp_jax, jvp_grad_jax = jax.jvp(
        func_regulararray_all_none_jax,
        (test_regulararray_jax,),
        (test_regulararray_tangent_jax,),
    )

    value_vjp, vjp_func = jax.vjp(func_regulararray_all_none, test_regulararray)
    value_vjp_jax, vjp_func_jax = jax.vjp(
        func_regulararray_all_none_jax, test_regulararray_jax
    )

    assert ak.to_list(value_jvp) == value_jvp_jax.tolist()
    assert ak.to_list(value_vjp) == value_vjp_jax.tolist()
    assert ak.to_list(jvp_grad) == jvp_grad_jax.tolist()
    assert (
        ak.to_list(vjp_func(value_vjp)[0]) == (vjp_func_jax(value_vjp_jax)[0]).tolist()
    )


def test_regular_array_any_0():
    def func_regulararray_any_0(x):
        return ak.any(x, 0)

    with pytest.raises(
        TypeError, match=".*Make sure that you are not computing the derivative.*"
    ):
        jax.jvp(
            func_regulararray_any_0, (test_regulararray,), (test_regulararray_tangent,)
        )


def test_regular_array_any_1():
    def func_regulararray_any_1(x):
        return ak.any(x, 1)

    with pytest.raises(
        TypeError, match=".*Make sure that you are not computing the derivative.*"
    ):
        jax.jvp(
            func_regulararray_any_1, (test_regulararray,), (test_regulararray_tangent,)
        )


def test_regular_array_any_none():
    def func_regulararray_any_none(x):
        return ak.any(x)

    def func_regulararray_any_none_jax(x):
        return jax.numpy.any(x)

    value_jvp, jvp_grad = jax.jvp(
        func_regulararray_any_none, (test_regulararray,), (test_regulararray_tangent,)
    )
    value_jvp_jax, jvp_grad_jax = jax.jvp(
        func_regulararray_any_none_jax,
        (test_regulararray_jax,),
        (test_regulararray_tangent_jax,),
    )

    value_vjp, vjp_func = jax.vjp(func_regulararray_any_none, test_regulararray)
    value_vjp_jax, vjp_func_jax = jax.vjp(
        func_regulararray_any_none_jax, test_regulararray_jax
    )

    assert ak.to_list(value_jvp) == value_jvp_jax.tolist()
    assert ak.to_list(value_vjp) == value_vjp_jax.tolist()
    assert ak.to_list(jvp_grad) == jvp_grad_jax.tolist()
    assert (
        ak.to_list(vjp_func(value_vjp)[0]) == (vjp_func_jax(value_vjp_jax)[0]).tolist()
    )
