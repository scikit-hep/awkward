# ak.to_backend

Defined in [awkward.operations.ak_to_backend](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_to_backend.py) on [line 15](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_to_backend.py#L15).

#### ak.to_backend(array, backend, \*, highlevel=True, behavior=None, attrs=None)

Returns an array on a different backend (kernel set).

Any components that are already in the desired backend are viewed,
rather than copied, so this operation can be an inexpensive way to ensure
that an array is ready for a particular library.

To use `"cuda"`, the `cupy` package must be installed, either with:

```default
pip install cupy
```

or:

```default
conda install -c conda-forge cupy
```

To use `"jax"`, the `jax` package must be installed, either with:

```default
pip install jax
```

or:

```default
conda install -c conda-forge jax
```

See `ak.kernels`.

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **backend** (`"cpu"`, `"cuda"`, `"jax"`, or `"typetracer"`) – If `"cpu"`, the array structure is
    recursively copied (if need be) to main memory for use with
    the default Numpy backend; if `"cuda"`, the structure is copied
    to the GPU(s) for use with CuPy. If `"jax"`, the structure is
    copied to the CPU for use with JAX.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An array with the same data as the input, moved to the requested backend
  (kernel set).
