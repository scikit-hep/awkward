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

    value_jvp, jvp_grad = jax.jvp(func_numpyarray_1, (test_numpyarray,), (test_numpyarray_tangent,))
    jit_value = jax.jit(func_numpyarray_1)(test_numpyarray)
    value_vjp, vjp_func = jax.vjp(func_numpyarray_1, test_numpyarray)
    
    assert value_jvp == 16.0
    assert value_vjp == 16.0
    assert jit_value == 16.0
    assert jvp_grad == 32.0
    assert vjp_func(value_vjp)[0] == [0.0, 10.0, 72.0, 228.0, 200.0, 0.0, 0.0, 0.0, 0.0, 0.0]

def test_numpyarray_grad_2():
    def func_numpyarray_2(x):
        return x[2:5] ** 2 + x[1:4] ** 2

    value_jvp, jvp_grad = jax.jvp(func_numpyarray_2, (test_numpyarray,), (test_numpyarray_tangent,))
    jit_value = jax.jit(func_numpyarray_2)(test_numpyarray)
    value_vjp, vjp_func = jax.vjp(func_numpyarray_2, test_numpyarray)
    
    assert ak.to_list(value_jvp) == [5, 13, 25]
    assert ak.to_list(value_vjp) == [5, 13, 25]
    assert ak.to_list(jit_value) == [5, 13, 25]
    assert ak.to_list(jvp_grad) == [10.0, 26.0, 50.0]
    assert ak.to_list(vjp_func(value_vjp)[0]) == [0.0, 10.0, 72.0, 228.0, 200.0, 0.0, 0.0, 0.0, 0.0, 0.0]

def test_numpyarray_grad_3():
    def func_numpyarray_3(x):
        return x[::-1]
    
    value_jvp, jvp_grad = jax.jvp(func_numpyarray_3, (test_numpyarray,), (test_numpyarray_tangent,))
    jit_value = jax.jit(func_numpyarray_3)(test_numpyarray)
    value_vjp, vjp_func = jax.vjp(func_numpyarray_3, test_numpyarray)
    
    assert ak.to_list(value_jvp) == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    assert ak.to_list(value_vjp) == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0] 
    assert ak.to_list(jit_value) == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    assert ak.to_list(jvp_grad) == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    assert ak.to_list(vjp_func(test_numpyarray)[0]) == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]


def test_numpyarray_grad_4():
    def func_numpyarray_4(x):
        return x[2:5] ** 2 * x[1:4] ** 2

    value_jvp, jvp_grad = jax.jvp(func_numpyarray_4, (test_numpyarray,), (test_numpyarray_tangent,))
    jit_value = jax.jit(func_numpyarray_4)(test_numpyarray)
    value_vjp, vjp_func = jax.vjp(func_numpyarray_4, test_numpyarray)
    
    assert ak.to_list(value_jvp) == [4.0, 36.0, 144.0]
    assert ak.to_list(value_vjp) == [4.0, 36.0, 144.0]
    assert ak.to_list(jit_value) == [4.0, 36.0, 144.0]
    assert ak.to_list(jvp_grad) == [16.0, 144.0, 576.0]
    assert ak.to_list(vjp_func(value_vjp)[0]) == [0.0, 32.0, 1312.0, 14688.0, 10368.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# value_jvp, jvp_grad = jax.jvp(func_numpyarray_3, (test_numpyarray,), (test_numpyarray_tangent,))
# # value_jvp, jvp_grad = jax.jvp(func_numpyarray_3, (test_numpyarray_jax,), (test_numpyarray_tangent_jax,))

# jit_value = jax.jit(func_numpyarray_3)(test_numpyarray)
# value_vjp, vjp_func = jax.vjp(func_numpyarray_1, test_numpyarray)

# # print(value_vjp)
# # print(vjp_func(test_numpyarray))
# # # value, grad = jax.value_and_grad(func_numpyarray_2)(test_nparray)

# print("Value and Grad are {0} and {1}".format(value_jvp, jvp_grad))
# print("JIT value is {0}".format(jit_value))
# assert 1 == 2
