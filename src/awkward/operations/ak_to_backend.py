# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()


def to_backend(array, backend, *, highlevel=True, behavior=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        backend (`"cpu"`, `"cuda"`, or `"jax"`): If `"cpu"`, the array structure is
            recursively copied (if need be) to main memory for use with
            the default Numpy backend; if `"cuda"`, the structure is copied
            to the GPU(s) for use with CuPy. If `"jax"`, the structure is
            copied to the CPU for use with JAX.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Converts an array from `"cpu"`, `"cuda"`, or `"jax"` kernels to `"cpu"`,
    `"cuda"`, or `"jax"`.

    Any components that are already in the desired backend are viewed,
    rather than copied, so this operation can be an inexpensive way to ensure
    that an array is ready for a particular library.

    To use `"cuda"`, the `cupy` package must be installed, either with

        pip install cupy

    or

        conda install -c conda-forge cupy

    To use `"jax"`, the `jax` package must be installed, either with

        pip install jax

    or

        conda install -c conda-forge jax

    See #ak.kernels.
    """
    with ak._errors.OperationErrorContext(
        "ak.to_backend",
        dict(array=array, backend=backend, highlevel=highlevel, behavior=behavior),
    ):
        return _impl(array, backend, highlevel, behavior)


def _impl(array, backend, highlevel, behavior):
    layout = ak.operations.to_layout(
        array,
        allow_record=True,
        allow_other=True,
    )
    behavior = ak._util.behavior_of(array, behavior=behavior)
    backend_layout = layout.to_backend(ak._backends.regularize_backend(backend))
    return ak._util.wrap(backend_layout, behavior, highlevel)
