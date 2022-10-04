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
        ak.to_list(vjp_func(value_vjp)[0]) == (vjp_func_jax(value_vjp_jax)[0]).tolist()
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

    assert ak.to_list(value_jvp) == value_jvp_jax.tolist()
    assert ak.to_list(value_vjp) == value_vjp_jax.tolist()
    assert ak.to_list(jvp_grad) == jvp_grad_jax.tolist()
    assert (
        ak.to_list(vjp_func(value_vjp)[0]) == (vjp_func_jax(value_vjp_jax)[0]).tolist()
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

    assert ak.to_list(value_jvp) == value_jvp_jax.tolist()
    assert ak.to_list(value_vjp) == value_vjp_jax.tolist()
    assert ak.to_list(jvp_grad) == jvp_grad_jax.tolist()
    assert (
        ak.to_list(vjp_func(value_vjp)[0]) == (vjp_func_jax(value_vjp_jax)[0]).tolist()
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

    assert ak.to_list(value_jvp) == value_jvp_jax.tolist()
    assert ak.to_list(value_vjp) == value_vjp_jax.tolist()
    assert ak.to_list(jvp_grad) == jvp_grad_jax.tolist()
    assert (
        ak.to_list(vjp_func(value_vjp)[0]) == (vjp_func_jax(value_vjp_jax)[0]).tolist()
    )


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

    assert ak.to_list(value_jvp) == value_jvp_jax.tolist()
    assert ak.to_list(value_vjp) == value_vjp_jax.tolist()
    assert ak.to_list(jvp_grad) == jvp_grad_jax.tolist()
    assert (
        ak.to_list(vjp_func(value_vjp)[0]) == (vjp_func_jax(value_vjp_jax)[0]).tolist()
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

    assert ak.to_list(value_jvp) == value_jvp_jax.tolist()
    assert ak.to_list(value_vjp) == value_vjp_jax.tolist()
    assert ak.to_list(jvp_grad) == jvp_grad_jax.tolist()
    assert (
        ak.to_list(vjp_func(value_vjp)[0]) == (vjp_func_jax(value_vjp_jax)[0]).tolist()
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

    assert ak.to_list(value_jvp) == value_jvp_jax.tolist()
    assert ak.to_list(value_vjp) == value_vjp_jax.tolist()
    assert ak.to_list(jvp_grad) == jvp_grad_jax.tolist()
    assert (
        ak.to_list(vjp_func(value_vjp)[0]) == (vjp_func_jax(value_vjp_jax)[0]).tolist()
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

    assert ak.to_list(value_jvp) == value_jvp_jax.tolist()
    assert ak.to_list(value_vjp) == value_vjp_jax.tolist()
    assert ak.to_list(jvp_grad) == jvp_grad_jax.tolist()
    assert (
        ak.to_list(vjp_func(value_vjp)[0]) == (vjp_func_jax(value_vjp_jax)[0]).tolist()
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

    assert ak.to_list(value_jvp) == value_jvp_jax.tolist()
    assert ak.to_list(value_vjp) == value_vjp_jax.tolist()
    assert ak.to_list(jvp_grad) == jvp_grad_jax.tolist()
    assert (
        ak.to_list(vjp_func(value_vjp)[0]) == (vjp_func_jax(value_vjp_jax)[0]).tolist()
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

    assert ak.to_list(value_jvp) == value_jvp_jax.tolist()
    assert ak.to_list(value_vjp) == value_vjp_jax.tolist()
    assert ak.to_list(jvp_grad) == jvp_grad_jax.tolist()
    assert (
        ak.to_list(vjp_func(value_vjp)[0]) == (vjp_func_jax(value_vjp_jax)[0]).tolist()
    )


test_recordarray = ak.Array(
    [
        [{"x": 1.1, "y": [1.0]}, {"x": 2.2, "y": [1.0, 2.2]}],
        [],
        [{"x": 3.3, "y": [1.0, 2.0, 3.0]}],
    ],
    backend="jax",
)
test_recordarray_tangent = ak.Array(
    [
        [{"x": 0.0, "y": [1.0]}, {"x": 2.0, "y": [1.5, 0.0]}],
        [],
        [{"x": 1.5, "y": [2.0, 0.5, 1.0]}],
    ],
    backend="jax",
)


def test_recordarray_1():
    def func_recordarray_1(x):
        return 2 * x.y[2][0][1] + 10

    value_jvp, jvp_grad = jax.jvp(
        func_recordarray_1, (test_recordarray,), (test_recordarray_tangent,)
    )
    value_vjp, vjp_func = jax.vjp(func_recordarray_1, test_recordarray)
    assert ak.to_list(value_jvp) == 14.0
    assert ak.to_list(value_vjp) == 14.0
    assert ak.to_list(jvp_grad) == 1.0
    assert ak.to_list(vjp_func(value_vjp)[0]) == [
        [{"x": 0.0, "y": [0.0]}, {"x": 0.0, "y": [0.0, 0.0]}],
        [],
        [{"x": 0.0, "y": [0.0, 28.0, 0.0]}],
    ]


def test_recordarray_2():
    def func_recordarray_2(x):
        return 2 * x.y[2][0] + 10

    value_jvp, jvp_grad = jax.jvp(
        func_recordarray_2, (test_recordarray,), (test_recordarray_tangent,)
    )
    value_vjp, vjp_func = jax.vjp(func_recordarray_2, test_recordarray)
    assert ak.to_list(value_jvp) == [12.0, 14.0, 16.0]
    assert ak.to_list(value_vjp) == [12.0, 14.0, 16.0]
    assert ak.to_list(jvp_grad) == [4.0, 1.0, 2.0]
    assert ak.to_list(vjp_func(value_vjp)[0]) == [
        [{"x": 0.0, "y": [0.0]}, {"x": 0.0, "y": [0.0, 0.0]}],
        [],
        [{"x": 0.0, "y": [24.0, 28.0, 32.0]}],
    ]


def test_recordarray_3():
    def test_recordarray_3(x):
        return 2 * x.y[0][0] ** 2

    value_jvp, jvp_grad = jax.jvp(
        test_recordarray_3, (test_recordarray,), (test_recordarray_tangent,)
    )
    value_vjp, vjp_func = jax.vjp(test_recordarray_3, test_recordarray)
    assert ak.to_list(value_jvp) == [2.0]
    assert ak.to_list(value_vjp) == [2.0]
    assert ak.to_list(jvp_grad) == [4.0]
    assert ak.to_list(vjp_func(value_vjp)[0]) == [
        [{"x": 0.0, "y": [8.0]}, {"x": 0.0, "y": [0.0, 0.0]}],
        [],
        [{"x": 0.0, "y": [0.0, 0.0, 0.0]}],
    ]


def test_recordarray_4():
    def test_recordarray_4(x):
        return 2 * x.y[2] + 10

    value_jvp, jvp_grad = jax.jvp(
        test_recordarray_4, (test_recordarray,), (test_recordarray_tangent,)
    )
    value_vjp, vjp_func = jax.vjp(test_recordarray_4, test_recordarray)
    assert ak.to_list(value_jvp) == [[12.0, 14.0, 16.0]]
    assert ak.to_list(value_vjp) == [[12.0, 14.0, 16.0]]
    assert ak.to_list(jvp_grad) == [[4.0, 1.0, 2.0]]
    assert ak.to_list(vjp_func(value_vjp)[0]) == [
        [{"x": 0.0, "y": [0.0]}, {"x": 0.0, "y": [0.0, 0.0]}],
        [],
        [{"x": 0.0, "y": [24.0, 28.0, 32.0]}],
    ]


def test_recordarray_5():
    def test_recordarray_5(x):
        return 2 * x.y

    value_jvp, jvp_grad = jax.jvp(
        test_recordarray_5, (test_recordarray,), (test_recordarray_tangent,)
    )
    value_vjp, vjp_func = jax.vjp(test_recordarray_5, test_recordarray)
    assert ak.to_list(value_jvp) == [[[2.0], [2.0, 4.4]], [], [[2.0, 4.0, 6.0]]]
    assert ak.to_list(value_vjp) == [[[2.0], [2.0, 4.4]], [], [[2.0, 4.0, 6.0]]]
    assert ak.to_list(jvp_grad) == [[[2.0], [3.0, 0.0]], [], [[4.0, 1.0, 2.0]]]
    assert ak.to_list(vjp_func(value_vjp)[0]) == [
        [{"x": 0.0, "y": [4.0]}, {"x": 0.0, "y": [4.0, 8.8]}],
        [],
        [{"x": 0.0, "y": [4.0, 8.0, 12.0]}],
    ]


def test_recordarray_6():
    def test_recordarray_6(x):
        return 2 * x.y**2

    value_jvp, jvp_grad = jax.jvp(
        test_recordarray_6, (test_recordarray,), (test_recordarray_tangent,)
    )
    value_vjp, vjp_func = jax.vjp(test_recordarray_6, test_recordarray)
    assert ak.to_list(value_jvp) == [
        [[2.0], [2.0, 9.680000000000001]],
        [],
        [[2.0, 8.0, 18.0]],
    ]
    assert ak.to_list(value_vjp) == [
        [[2.0], [2.0, 9.680000000000001]],
        [],
        [[2.0, 8.0, 18.0]],
    ]
    assert ak.to_list(jvp_grad) == [[[4.0], [6.0, 0.0]], [], [[8.0, 4.0, 12.0]]]
    assert ak.to_list(vjp_func(value_vjp)[0]) == [
        [{"x": 0.0, "y": [8.0]}, {"x": 0.0, "y": [8.0, 85.18400000000003]}],
        [],
        [{"x": 0.0, "y": [8.0, 64.0, 216.0]}],
    ]


def test_recordarray_7():
    def test_recordarray_7(x):
        return 2 * x.y[2, 0, 1] + 10

    value_jvp, jvp_grad = jax.jvp(
        test_recordarray_7, (test_recordarray,), (test_recordarray_tangent,)
    )
    value_vjp, vjp_func = jax.vjp(test_recordarray_7, test_recordarray)
    assert ak.to_list(value_jvp) == 14.0
    assert ak.to_list(value_vjp) == 14.0
    assert ak.to_list(jvp_grad) == 1.0
    assert ak.to_list(vjp_func(value_vjp)[0]) == [
        [{"x": 0.0, "y": [0.0]}, {"x": 0.0, "y": [0.0, 0.0]}],
        [],
        [{"x": 0.0, "y": [0.0, 28.0, 0.0]}],
    ]


def test_recordarray_8():
    def func_recordarray_8(x):
        return 2 * x.y[2, 0] + 10

    value_jvp, jvp_grad = jax.jvp(
        func_recordarray_8, (test_recordarray,), (test_recordarray_tangent,)
    )
    value_vjp, vjp_func = jax.vjp(func_recordarray_8, test_recordarray)
    assert ak.to_list(value_jvp) == [12.0, 14.0, 16.0]
    assert ak.to_list(value_vjp) == [12.0, 14.0, 16.0]
    assert ak.to_list(jvp_grad) == [4.0, 1.0, 2.0]
    assert ak.to_list(vjp_func(value_vjp)[0]) == [
        [{"x": 0.0, "y": [0.0]}, {"x": 0.0, "y": [0.0, 0.0]}],
        [],
        [{"x": 0.0, "y": [24.0, 28.0, 32.0]}],
    ]
