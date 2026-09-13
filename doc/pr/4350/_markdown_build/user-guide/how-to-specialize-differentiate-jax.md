# Differentiation using JAX

JAX, amongst other things, is a powerful tool for computing derivatives of native Python and NumPy code. Awkward Array implements support for the [`jax.grad()`](https://docs.jax.dev/en/latest/_autosummary/jax.grad.html#jax.grad), [`jax.jvp()`](https://docs.jax.dev/en/latest/_autosummary/jax.jvp.html#jax.jvp) and [`jax.vjp()`](https://docs.jax.dev/en/latest/_autosummary/jax.vjp.html#jax.vjp) JAX functions for computing gradients and forward/reverse-mode Jacobian-vector/vector-Jacobian products of functions that operate upon Awkward Arrays. Only a subset of Awkward Array operations can be differentiated through, including:

- ufunc operations like `x + y`
- reducers like [`ak.sum()`](sphinx-llm:08d8d4556ebd40868b7762f4e0ae9d1a#ak.sum)
- slices like `x[1:]`

## How to differentiate Awkward Arrays?

First, import JAX.

```ipython3
import jax
```

Next, we must call [`ak.jax.register_and_check()`](sphinx-llm:2a37bdd542db475b8948228f4d58059c#ak.jax.register_and_check) to register Awkward’s JAX integration.

```ipython3
import awkward as ak
ak.jax.register_and_check()
```

```myst-ansi
/tmp/ipykernel_5623/3763586035.py:2: DeprecationWarning: The JAX backend is deprecated and will be removed in a future release of Awkward Array. Please plan to migrate your code accordingly.
  ak.jax.register_and_check()
```

Let’s define a simple function that accepts an Awkward Array.

```ipython3
def reverse_sum(array):
    return ak.sum(array[::-1], axis=0)
```

We can then create an array with which to evaluate `reverse_sum`. The `backend` argument ensures that we build an Awkward Array that is backed by [`jax.Array`](https://docs.jax.dev/en/latest/_autosummary/jax.Array.html#jax.Array) (`jaxlib.xla_extension.ArrayImpl`) buffers, which power JAX’s automatic differentiation and JIT compiling features. However, Awkward Array’s JAX backend does not support JIT compilation on reducers as XLA requires array sizes to not be dependent on data values at compile-time.

```ipython3
array = ak.Array([[1.0, 2.0, 3.0], [], [4.0, 5.0]], backend="jax")
```

```myst-ansi
/home/runner/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_singleton.py:18: DeprecationWarning: The JAX backend is deprecated and will be removed in a future release of Awkward Array. Please plan to migrate your code accordingly.
  self.__init__()  # pylint: disable=unnecessary-dunder-call
```

```ipython3
reverse_sum(array)
```

Computing the JVP of `reverse_sum` requires a *tangent* vector, which can also be defined as an Awkward Array:

```ipython3
tangent = ak.Array([[0.0, 0.0, 0.0], [], [0.0, 1.0]], backend="jax")
```

```ipython3
value_jvp, jvp_grad = jax.jvp(reverse_sum, (array,), (tangent,))
```

[`jax.jvp()`](https://docs.jax.dev/en/latest/_autosummary/jax.jvp.html#jax.jvp) returns both the value of `reverse_sum` evaluated at `array`:

```ipython3
value_jvp
```

```ipython3
assert value_jvp.to_list() == reverse_sum(array).to_list()
```

and the JVP evaluted at `array` for the given `tangent`:

```ipython3
jvp_grad
```

Similarly, VJP of `reverse_sum` can be computed as:

```ipython3
value_vjp, func_vjp = jax.vjp(reverse_sum, array)
```

where `value_vjp` is the function (`reverse_sum`) evaluated at `array` (forward pass):

```ipython3
assert value_vjp.to_list() == reverse_sum(array).to_list()
```

and `func_vjp` is a function that takes a *cotangent* vector as an argument and returns the VJP (backward pass):

```ipython3
cotanget = ak.Array([0., 1., 0.], backend="jax")
```

```ipython3
func_vjp(value_vjp)
```

JAX’s own documentation encourages the user to use [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html#module-jax.numpy) instead of the canonical [`numpy`](https://numpy.org/doc/stable/reference/index.html#module-numpy) module when operating upon JAX arrays. However, [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html#module-jax.numpy) does not understand Awkward Arrays, so for [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array)s you should use the normal `ak` and [`numpy`](https://numpy.org/doc/stable/reference/index.html#module-numpy) functions instead.
