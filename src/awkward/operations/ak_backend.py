# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak


def backend(*arrays) -> str:
    """
    Returns the names of the backend used by `arrays`. May be

       * `"cpu"` for arrays backed by NumPy;
       * `"cuda"` for arrays backed by CuPy;
       * `"jax"` for arrays backed by JAX;
       * None if the objects are not Awkward, NumPy, JAX, or CuPy arrays (e.g.
         Python numbers, booleans, strings).

    See #ak.to_backend.
    """
    with ak._errors.OperationErrorContext(
        "ak.backend",
        {"*arrays": arrays},
    ):
        return _impl(arrays)


def _impl(arrays) -> str:
    backend_impl = ak._backends.backend_of(*arrays, default=None)
    return backend_impl.name
