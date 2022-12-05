# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak


def backend(*arrays) -> str:
    """
    Args:
        arrays: Array-like data (anything #ak.to_layout recognizes).

    Returns the names of the backend used by `arrays`. This name may be

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
    if isinstance(backend_impl, ak._backends.TypeTracerBackend):
        raise ak._errors.wrap_error(
            ValueError(
                "at least one of the given arrays was a typetracer array. "
                "This is an internal backend that you should not have encountered. "
                "Please file a bug report at https://github.com/scikit-hep/awkward/issues/"
            )
        )
    return backend_impl.name
