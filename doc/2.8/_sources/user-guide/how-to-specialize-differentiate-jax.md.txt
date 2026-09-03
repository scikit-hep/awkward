---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

Differentiation using JAX
=========================

JAX, amongst other things, is a powerful tool for computing derivatives of native Python and NumPy code. Awkward Array implements support for the {func}`jax.grad`, {func}`jax.jvp` and {func}`jax.vjp` JAX functions for computing gradients and forward/reverse-mode Jacobian-vector/vector-Jacobian products of functions that operate upon Awkward Arrays. Only a subset of Awkward Array operations can be differentiated through, including:
- ufunc operations like `x + y`
- reducers like {func}`ak.sum`
- slices like `x[1:]`

+++

How to differentiate Awkward Arrays?
------------------------------------

For this notebook (which is evaluated on a CPU), we need to configure JAX to use only the CPU.

```{code-cell}
import jax
jax.config.update("jax_platform_name", "cpu")
```

Next, we must call {func}`ak.jax.register_and_check()` to register Awkward's JAX integration.

```{code-cell}
import awkward as ak
ak.jax.register_and_check()
```

Let's define a simple function that accepts an Awkward Array.

```{code-cell}
def reverse_sum(array):
    return ak.sum(array[::-1], axis=0)
```

We can then create an array with which to evaluate `reverse_sum`. The `backend` argument ensures that we build an Awkward Array that is backed by {class}`jax.Array` ({class}`jaxlib.xla_extension.ArrayImpl`) buffers, which power JAX's automatic differentiation and JIT compiling features. However, Awkward Array's JAX backend does not support JIT compilation on reducers as XLA requires array sizes to not be dependent on data values at compile-time.

```{code-cell}
array = ak.Array([[1.0, 2.0, 3.0], [], [4.0, 5.0]], backend="jax")
```

```{code-cell}
reverse_sum(array)
```

Computing the JVP of `reverse_sum` requires a _tangent_ vector, which can also be defined as an Awkward Array:

```{code-cell}
tangent = ak.Array([[0.0, 0.0, 0.0], [], [0.0, 1.0]], backend="jax")
```

```{code-cell}
value_jvp, jvp_grad = jax.jvp(reverse_sum, (array,), (tangent,))
```

{func}`jax.jvp` returns both the value of `reverse_sum` evaluated at `array`:

```{code-cell}
value_jvp
```

```{code-cell}
assert value_jvp.to_list() == reverse_sum(array).to_list()
```

and the JVP evaluted at `array` for the given `tangent`:

```{code-cell}
jvp_grad
```

Similarly, VJP of `reverse_sum` can be computed as:

```{code-cell}
value_vjp, func_vjp = jax.vjp(reverse_sum, array)
```

where `value_vjp` is the function (`reverse_sum`) evaluated at `array` (forward pass):

```{code-cell}
assert value_vjp.to_list() == reverse_sum(array).to_list()
```

and `func_vjp` is a function that takes a _cotangent_ vector as an argument and returns the VJP (backward pass):

```{code-cell}
cotanget = ak.Array([0., 1., 0.], backend="jax")
```

```{code-cell}
func_vjp(value_vjp)
```

JAX's own documentation encourages the user to use {mod}`jax.numpy` instead of the canonical {mod}`numpy` module when operating upon JAX arrays. However, {mod}`jax.numpy` does not understand Awkward Arrays, so for {class}`ak.Array`s you should use the normal {mod}`ak` and {mod}`numpy` functions instead.
