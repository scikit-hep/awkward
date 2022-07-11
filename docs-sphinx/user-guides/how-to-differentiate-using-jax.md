---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

Differentiation using JAX
=========================

Currently, all the functions which contain slicing or numpy ufuncs are supported by Awkward Arrays and can be differentiated by JAX. We do not support any specialized funtions like `ak.sum()` or `ak.prod()`. These are planned to be implemented in the near future. Since, the GPU support for Awkward Arrays is only partially complete, we have to configure JAX to use CPU only. We can do this by:

```{code-cell}
import jax
jax.config.update("jax_platform_name", "cpu")
```
How to differentiate Awkward Arrays?
------------------------------------

Before using JAX on functions which deal with Awkward Arrays we need to call the `ak.jax.register()`. This makes `JAX` aware of Awkward Arrays. Here's an example:

```{code-cell}
import awkward as ak
import numpy as np

import jax
jax.config.update("jax_platform_name", "cpu")
ak.jax.register()


listoffsetarray = ak.Array([[1.0, 2.0, 3.0], [], [4.0, 5.0]])
listoffsetarray_tangent = ak.Array([[0.0, 0.0, 0.0], [], [0.0, 1.0]])

def func(x):
    return x[::-1] ** 2

value_jvp, jvp_grad = jax.jvp(
    func, (listoffsetarray,), (listoffsetarray_tangent,)
)
jit_value = jax.jit(func)(listoffsetarray)
value_vjp, vjp_func = jax.vjp(func, listoffsetarray)

print(value_jvp, jvp_grad)
print(jit_value)
print(value_vjp, vjp_func(value_vjp))
```

Here's how we can use numpy ufuncs on Awkward Arrays and have them differentiated by JAX:

```{code-cell}
import awkward as ak
import numpy as np

import jax
jax.config.update("jax_platform_name", "cpu")
ak.jax.register()


listoffsetarray = ak.Array([[1.0, 2.0, 3.0], [], [4.0, 5.0]])
listoffsetarray_tangent = ak.Array([[0.0, 0.0, 0.0], [], [0.0, 1.0]])

def func(x):
    return np.square(np.sin(x))

value_jvp, jvp_grad = jax.jvp(
    func, (listoffsetarray,), (listoffsetarray_tangent,)
)
jit_value = jax.jit(func)(listoffsetarray)
value_vjp, vjp_func = jax.vjp(func, listoffsetarray)

print(value_jvp, jvp_grad)
print(jit_value)
print(value_vjp, vjp_func(value_vjp))
```
Please note that we can't use `jax.numpy` ufuncs on Awkward Arrays.

What JAX functions are currently supported?
-------------------------------------------

Till now, we have tested and support three JAX functions, `jax.vjp`, `jax.jvp` and `jax.jit`. While, `jax.jvp` and `jax.jit` work for all elementwise differntiation cases, `jax.vjp` has a limitation with Awkward Arrays where you can't differentiate functions which output a scalar. Instead a workaround is to use slices to output get the scalar in the form of an Awkward Array. Here's an example:

```{code-cell}
import awkward as ak
import numpy as np

import jax
jax.config.update("jax_platform_name", "cpu")
ak.jax.register()


listoffsetarray = ak.Array([[1.0, 2.0, 3.0], [], [4.0, 5.0]])
listoffsetarray_tangent = ak.Array([[0.0, 0.0, 0.0], [], [0.0, 1.0]])

def func(x):
    return np.sin(x)[2][1:]

value_jvp, jvp_grad = jax.jvp(
    func, (listoffsetarray,), (listoffsetarray_tangent,)
)
jit_value = jax.jit(func)(listoffsetarray)
value_vjp, vjp_func = jax.vjp(func, listoffsetarray)

print(value_jvp, jvp_grad)
print(jit_value)
print(value_vjp, vjp_func(value_vjp))

```

In this example, we intended to fetch the last element of the third sublist, but instead of using `np.sin(x)[2][1]`, which would output a scalar, we had to settle down for `np.sin(x)[2][1:]` which makes it an Awkward Array type.
