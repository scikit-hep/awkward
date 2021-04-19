import awkward as ak
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_platform_name", "cpu")
ak.jax.register()

# #### ak.layout.NumpyArray ####

test_numpyarray = ak.Array(np.arange(10, dtype=np.float64))
test_numpyarray_tangent = ak.Array(np.arange(10, dtype=np.float64))

test_numpyarray_jax = jax.numpy.arange(10, dtype=np.float64)
test_numpyarray_tangent_jax = jax.numpy.arange(10, dtype=np.float64)


def test_numpyarray_grad_1():
    def func_numpyarray_1(x):
        return x[4] ** 2

    value_jvp, jvp_grad = jax.jvp(
        func_numpyarray_1, (test_numpyarray,), (test_numpyarray_tangent,)
    )
    jit_value = jax.jit(func_numpyarray_1)(test_numpyarray)
    # value_vjp, vjp_func = jax.vjp(func_numpyarray_1, test_numpyarray)

    assert value_jvp == 16.0
    # assert value_vjp == 16.0
    assert jit_value == 16.0
    assert jvp_grad == 32.0
    # assert vjp_func(value_vjp)[0] == [0.0, 10.0, 72.0, 228.0, 200.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_numpyarray_grad_2():
    def func_numpyarray_2(x):
        return x[2:5] ** 2 + x[1:4] ** 2

    value_jvp, jvp_grad = jax.jvp(
        func_numpyarray_2, (test_numpyarray,), (test_numpyarray_tangent,)
    )
    jit_value = jax.jit(func_numpyarray_2)(test_numpyarray)
    value_vjp, vjp_func = jax.vjp(func_numpyarray_2, test_numpyarray)

    assert ak.to_list(value_jvp) == [5, 13, 25]
    assert ak.to_list(value_vjp) == [5, 13, 25]
    assert ak.to_list(jit_value) == [5, 13, 25]
    assert ak.to_list(jvp_grad) == [10.0, 26.0, 50.0]
    assert ak.to_list(vjp_func(value_vjp)[0]) == [
        0.0,
        10.0,
        72.0,
        228.0,
        200.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]


def test_numpyarray_grad_3():
    def func_numpyarray_3(x):
        return x[::-1]

    value_jvp, jvp_grad = jax.jvp(
        func_numpyarray_3, (test_numpyarray_jax,), (test_numpyarray_tangent_jax,)
    )
    jit_value = jax.jit(func_numpyarray_3)(test_numpyarray)
    value_vjp, vjp_func = jax.vjp(func_numpyarray_3, test_numpyarray_jax)

    assert ak.to_list(value_jvp) == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    assert ak.to_list(value_vjp) == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    assert ak.to_list(jit_value) == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    assert ak.to_list(jvp_grad) == [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    assert ak.to_list(vjp_func(value_vjp)[0]) == [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
    ]


def test_numpyarray_grad_4():
    def func_numpyarray_4(x):
        return x[2:5] ** 2 * x[1:4] ** 2

    value_jvp, jvp_grad = jax.jvp(
        func_numpyarray_4, (test_numpyarray,), (test_numpyarray_tangent,)
    )
    jit_value = jax.jit(func_numpyarray_4)(test_numpyarray)
    value_vjp, vjp_func = jax.vjp(func_numpyarray_4, test_numpyarray)

    assert ak.to_list(value_jvp) == [4.0, 36.0, 144.0]
    assert ak.to_list(value_vjp) == [4.0, 36.0, 144.0]
    assert ak.to_list(jit_value) == [4.0, 36.0, 144.0]
    assert ak.to_list(jvp_grad) == [16.0, 144.0, 576.0]
    assert ak.to_list(vjp_func(value_vjp)[0]) == [
        0.0,
        32.0,
        1312.0,
        14688.0,
        10368.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]


test_listoffsetarray = ak.Array([[1.0, 2.0, 3.0], [], [4.0, 5.0]])
test_listoffsetarray_tangent = ak.Array([[0.0, 0.0, 0.0], [], [0.0, 1.0]])


def test_listoffset_array_1():
    def func_listoffsetarray_1(x):
        return x[2] * 2

    value_jvp, jvp_grad = jax.jvp(
        func_listoffsetarray_1, (test_listoffsetarray,), (test_listoffsetarray_tangent,)
    )
    jit_value = jax.jit(func_listoffsetarray_1)(test_listoffsetarray)
    value_vjp, vjp_func = jax.vjp(func_listoffsetarray_1, test_listoffsetarray)

    assert ak.to_list(value_jvp) == [8, 10]
    assert ak.to_list(value_vjp) == [8, 10]
    assert ak.to_list(jit_value) == [8, 10]
    assert ak.to_list(jvp_grad) == [0.0, 2.0]
    assert ak.to_list(vjp_func(value_vjp)[0]) == [[0.0, 0.0, 0.0], [], [16.0, 20.0]]


def test_listoffset_array_2():
    def func_listoffsetarray_2(x):
        return x * x

    value_jvp, jvp_grad = jax.jvp(
        func_listoffsetarray_2, (test_listoffsetarray,), (test_listoffsetarray_tangent,)
    )
    jit_value = jax.jit(func_listoffsetarray_2)(test_listoffsetarray)
    value_vjp, vjp_func = jax.vjp(func_listoffsetarray_2, test_listoffsetarray)

    assert ak.to_list(value_jvp) == [[1.0, 4.0, 9.0], [], [16.0, 25.0]]
    assert ak.to_list(value_vjp) == [[1.0, 4.0, 9.0], [], [16.0, 25.0]]
    assert ak.to_list(jit_value) == [[1.0, 4.0, 9.0], [], [16.0, 25.0]]
    assert ak.to_list(jvp_grad) == [[0.0, 0.0, 0.0], [], [0.0, 10.0]]
    assert ak.to_list(vjp_func(value_vjp)[0]) == [[2.0, 16.0, 54.0], [], [128.0, 250.0]]


def test_listoffset_array_3():
    def func_listoffsetarray_3(x):
        return x[0, 0] * x[2, 1]

    value_jvp, jvp_grad = jax.jvp(
        func_listoffsetarray_3, (test_listoffsetarray,), (test_listoffsetarray_tangent,)
    )
    jit_value = jax.jit(func_listoffsetarray_3)(test_listoffsetarray)
    # value_vjp, vjp_func = jax.vjp(func_listoffsetarray_2, test_listoffsetarray)

    assert ak.to_list(value_jvp) == 5.0
    # assert ak.to_list(value_vjp) == 2.0
    assert ak.to_list(jit_value) == 5.0
    assert ak.to_list(jvp_grad) == 1.0
    # assert ak.to_list(vjp_func(value_vjp)[0]) == [[2.0, 16.0, 54.0], [], [128.0, 250.0]]


def test_listoffset_array_4():
    def func_listoffsetarray_4(x):
        return x[::-1] ** 2

    value_jvp, jvp_grad = jax.jvp(
        func_listoffsetarray_4, (test_listoffsetarray,), (test_listoffsetarray_tangent,)
    )
    jit_value = jax.jit(func_listoffsetarray_4)(test_listoffsetarray)
    value_vjp, vjp_func = jax.vjp(func_listoffsetarray_4, test_listoffsetarray)

    assert ak.to_list(value_jvp) == [[16.0, 25.0], [], [1.0, 4.0, 9.0]]
    assert ak.to_list(value_vjp) == [[16.0, 25.0], [], [1.0, 4.0, 9.0]]
    assert ak.to_list(jit_value) == [[16.0, 25.0], [], [1.0, 4.0, 9.0]]
    assert ak.to_list(jvp_grad) == [[0.0, 10.0], [], [0.0, 0.0, 0.0]]
    assert ak.to_list(vjp_func(value_vjp)[0]) == [[2.0, 16.0, 54.0], [], [128.0, 250.0]]


def test_listoffset_array_5():
    def func_listoffsetarray_5(x):
        return 2 * x[:-1]

    value_jvp, jvp_grad = jax.jvp(
        func_listoffsetarray_5, (test_listoffsetarray,), (test_listoffsetarray_tangent,)
    )
    jit_value = jax.jit(func_listoffsetarray_5)(test_listoffsetarray)
    value_vjp, vjp_func = jax.vjp(func_listoffsetarray_5, test_listoffsetarray)

    assert ak.to_list(value_jvp) == [[2.0, 4.0, 6.0], []]
    assert ak.to_list(value_vjp) == [[2.0, 4.0, 6.0], []]
    assert ak.to_list(jit_value) == [[2.0, 4.0, 6.0], []]
    assert ak.to_list(jvp_grad) == [[0.0, 0.0, 0.0], []]
    assert ak.to_list(vjp_func(value_vjp)[0]) == [[4.0, 8.0, 12.0], [], [16.0, 20.0]]


def test_listoffset_array_6():
    def func_listoffsetarray_6(x):
        return x[0][0] * x[2][1]

    value_jvp, jvp_grad = jax.jvp(
        func_listoffsetarray_6, (test_listoffsetarray,), (test_listoffsetarray_tangent,)
    )
    jit_value = jax.jit(func_listoffsetarray_6)(test_listoffsetarray)
    # value_vjp, vjp_func = jax.vjp(func_listoffsetarray_2, test_listoffsetarray)

    assert ak.to_list(value_jvp) == 5.0
    # assert ak.to_list(value_vjp) == 2.0
    assert ak.to_list(jit_value) == 5.0
    assert ak.to_list(jvp_grad) == 1.0
    # assert ak.to_list(vjp_func(value_vjp)[0]) == [[2.0, 16.0, 54.0], [], [128.0, 250.0]]


test_recordarray = ak.Array(
    [
        [{"x": 1.1, "y": [1.0]}, {"x": 2.2, "y": [1.0, 2.2]}],
        [],
        [{"x": 3.3, "y": [1.0, 2.0, 3.0]}],
    ]
)
test_recordarray_tangent = ak.Array(
    [
        [{"x": 0.0, "y": [1.0]}, {"x": 2.0, "y": [1.5, 0.0]}],
        [],
        [{"x": 1.5, "y": [2.0, 0.5, 1.0]}],
    ]
)


def test_recordarray_1():
    def func_recordarray_1(x):
        return 2 * x.y[2][0][1] + 10

    value_jvp, jvp_grad = jax.jvp(
        func_recordarray_1, (test_recordarray,), (test_recordarray_tangent,)
    )
    jit_value = jax.jit(func_recordarray_1)(test_recordarray)
    # value_vjp, vjp_func = jax.vjp(func_listoffsetarray_2, test_listoffsetarray)

    assert ak.to_list(value_jvp) == 14.0
    # assert ak.to_list(value_vjp) == 14.0
    assert ak.to_list(jit_value) == 14.0
    assert ak.to_list(jvp_grad) == 1.0
    # assert ak.to_list(vjp_func(value_vjp)[0]) == [[2.0, 16.0, 54.0], [], [128.0, 250.0]]


def test_recordarray_2():
    def func_recordarray_2(x):
        return 2 * x.y[2][0] + 10

    value_jvp, jvp_grad = jax.jvp(
        func_recordarray_2, (test_recordarray,), (test_recordarray_tangent,)
    )
    jit_value = jax.jit(func_recordarray_2)(test_recordarray)
    value_vjp, vjp_func = jax.vjp(func_recordarray_2, test_recordarray)

    assert ak.to_list(value_jvp) == [12.0, 14.0, 16.0]
    assert ak.to_list(value_vjp) == [12.0, 14.0, 16.0]
    assert ak.to_list(jit_value) == [12.0, 14.0, 16.0]
    assert ak.to_list(jvp_grad) == [4.0, 1.0, 2.0]
    assert ak.to_list(vjp_func(value_vjp)[0]) == [
        [{"x": 0.0, "y": [0.0]}, {"x": 0.0, "y": [0.0, 0.0]}],
        [],
        [{"x": 0.0, "y": [24.0, 28.0, 32.0]}],
    ]


def test_recordarray_3():
    def func_recordarray_3(x):
        return 2 * x.y[0][0] ** 2

    value_jvp, jvp_grad = jax.jvp(
        func_recordarray_3, (test_recordarray,), (test_recordarray_tangent,)
    )
    jit_value = jax.jit(func_recordarray_3)(test_recordarray)
    value_vjp, vjp_func = jax.vjp(func_recordarray_3, test_recordarray)

    assert ak.to_list(value_jvp) == [2.0]
    assert ak.to_list(value_vjp) == [2.0]
    assert ak.to_list(jit_value) == [2.0]
    assert ak.to_list(jvp_grad) == [4.0]
    assert ak.to_list(vjp_func(value_vjp)[0]) == [
        [{"x": 0.0, "y": [8.0]}, {"x": 0.0, "y": [0.0, 0.0]}],
        [],
        [{"x": 0.0, "y": [0.0, 0.0, 0.0]}],
    ]


def test_recordarray_4():
    def func_recordarray_4(x):
        return 2 * x.y[2] + 10

    value_jvp, jvp_grad = jax.jvp(
        func_recordarray_4, (test_recordarray,), (test_recordarray_tangent,)
    )
    jit_value = jax.jit(func_recordarray_4)(test_recordarray)
    value_vjp, vjp_func = jax.vjp(func_recordarray_4, test_recordarray)

    assert ak.to_list(value_jvp) == [[12.0, 14.0, 16.0]]
    assert ak.to_list(value_vjp) == [[12.0, 14.0, 16.0]]
    assert ak.to_list(jit_value) == [[12.0, 14.0, 16.0]]
    assert ak.to_list(jvp_grad) == [[4.0, 1.0, 2.0]]
    assert ak.to_list(vjp_func(value_vjp)[0]) == [
        [{"x": 0.0, "y": [24.0]}, {"x": 0.0, "y": [24.0, 28.8]}],
        [],
        [{"x": 0.0, "y": [24.0, 28.0, 32.0]}],
    ]


def test_recordarray_5():
    def func_recordarray_5(x):
        return 2 * x.y

    value_jvp, jvp_grad = jax.jvp(
        func_recordarray_5, (test_recordarray,), (test_recordarray_tangent,)
    )
    jit_value = jax.jit(func_recordarray_5)(test_recordarray)
    value_vjp, vjp_func = jax.vjp(func_recordarray_5, test_recordarray)

    assert ak.to_list(value_jvp) == [[[2.0], [2.0, 4.4]], [], [[2.0, 4.0, 6.0]]]
    assert ak.to_list(value_vjp) == [[[2.0], [2.0, 4.4]], [], [[2.0, 4.0, 6.0]]]
    assert ak.to_list(jit_value) == [[[2.0], [2.0, 4.4]], [], [[2.0, 4.0, 6.0]]]
    assert ak.to_list(jvp_grad) == [[[2.0], [3.0, 0.0]], [], [[4.0, 1.0, 2.0]]]
    assert ak.to_list(vjp_func(value_vjp)[0]) == [
        [{"x": 0.0, "y": [4.0]}, {"x": 0.0, "y": [4.0, 8.8]}],
        [],
        [{"x": 0.0, "y": [4.0, 8.0, 12.0]}],
    ]


def test_recordarray_6():
    def func_recordarray_6(x):
        return 2 * x.y ** 2

    value_jvp, jvp_grad = jax.jvp(
        func_recordarray_6, (test_recordarray,), (test_recordarray_tangent,)
    )
    jit_value = jax.jit(func_recordarray_6)(test_recordarray)
    value_vjp, vjp_func = jax.vjp(func_recordarray_6, test_recordarray)

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
    assert ak.to_list(jit_value) == [
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
    def func_recordarray_7(x):
        return 2 * x.y[2, 0, 1] + 10

    value_jvp, jvp_grad = jax.jvp(
        func_recordarray_7, (test_recordarray,), (test_recordarray_tangent,)
    )
    jit_value = jax.jit(func_recordarray_7)(test_recordarray)
    # value_vjp, vjp_func = jax.vjp(func_listoffsetarray_2, test_listoffsetarray)

    assert ak.to_list(value_jvp) == 14.0
    # assert ak.to_list(value_vjp) == 14.0
    assert ak.to_list(jit_value) == 14.0
    assert ak.to_list(jvp_grad) == 1.0
    # assert ak.to_list(vjp_func(value_vjp)[0]) == [[2.0, 16.0, 54.0], [], [128.0, 250.0]]


def test_recordarray_8():
    def func_recordarray_8(x):
        return 2 * x.y[2, 0] + 10

    value_jvp, jvp_grad = jax.jvp(
        func_recordarray_8, (test_recordarray,), (test_recordarray_tangent,)
    )
    jit_value = jax.jit(func_recordarray_8)(test_recordarray)
    value_vjp, vjp_func = jax.vjp(func_recordarray_8, test_recordarray)

    assert ak.to_list(value_jvp) == [12.0, 14.0, 16.0]
    assert ak.to_list(value_vjp) == [12.0, 14.0, 16.0]
    assert ak.to_list(jit_value) == [12.0, 14.0, 16.0]
    assert ak.to_list(jvp_grad) == [4.0, 1.0, 2.0]
    assert ak.to_list(vjp_func(value_vjp)[0]) == [
        [{"x": 0.0, "y": [0.0]}, {"x": 0.0, "y": [0.0, 0.0]}],
        [],
        [{"x": 0.0, "y": [24.0, 28.0, 32.0]}],
    ]
