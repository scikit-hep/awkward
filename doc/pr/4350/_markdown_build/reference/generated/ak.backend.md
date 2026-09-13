# ak.backend

Defined in [awkward.operations.ak_backend](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_backend.py) on [line 11](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_backend.py#L11).

#### ak.backend(\*arrays)

Returns the name of the backend used by the given arrays.

This name may be

* `"cpu"` for arrays backed by NumPy;
* `"cuda"` for arrays backed by CuPy;
* `"jax"` for arrays backed by JAX;
* `"typetracer"` for arrays without any data;
* None if the objects are not Awkward, NumPy, JAX, CuPy, or typetracer
  arrays (e.g. Python numbers, booleans, strings).

If there are multiple, compatible backends (e.g. NumPy & typetracer)
amongst the given arrays, the coercible backend is returned.

See [`ak.to_backend`](sphinx-llm:fb9bd58d1c894eed98bb17af16147d71#ak.to_backend).

* **Parameters:**
  **arrays** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
* **Returns:**
  The name of the backend used by `arrays`.
