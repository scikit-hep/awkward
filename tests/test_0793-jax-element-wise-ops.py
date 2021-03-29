import awkward as ak
import numpy as np
import jax

#### ak.layout.NumpyArray ####

test_numpyarray = ak.Array(np.arange(10, dtype=np.float64))
test_numpyarray_tangent = ak.Array(np.arange(10, dtype=np.float64))

def func_numpyarray_1(x):
    return x[4] ** 2

def func_numpyarray_2(x):
    return x[2:5] ** 2 + x[1:4] ** 2

def func_numpyarray_3(x):
    return x[::-1]



# value_jvp, jvp_grad = jax.jvp(func_numpyarray_1, (test_numpyarray,), (test_numpyarray_tangent,))
# jit_value = jax.jit(func_numpyarray_3)(test_numpyarray)
value_vjp, vjp_func = jax.vjp(func_numpyarray_1, test_numpyarray)

print(value_vjp)
print(vjp_func(test_numpyarray))
# value, grad = jax.value_and_grad(func_numpyarray_2)(test_nparray)

# print("Value and Grad are {0} and {1}".format(value_jvp, jvp_grad))
# print("JIT value is {0}".format(jit_value))
