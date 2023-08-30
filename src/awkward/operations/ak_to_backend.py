# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
__all__ = ("to_backend",)
import awkward as ak
from awkward._backends.dispatch import regularize_backend
from awkward._behavior import behavior_of
from awkward._dispatch import high_level_function
from awkward._layout import wrap_layout
from awkward._nplikes.numpylike import NumpyMetadata

np = NumpyMetadata.instance()


@high_level_function()
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

    Converts an array from `"cpu"`, `"cuda"`, `"jax"` kernels to `"cpu"`,
    `"cuda"`, `"jax"`, or `"typetracer"` .

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
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, backend, highlevel, behavior)


def _impl(array, backend, highlevel, behavior):
    layout = ak.operations.to_layout(
        array,
        allow_record=True,
        allow_other=False,
    )
    behavior = behavior_of(array, behavior=behavior)
    backend_layout = layout.to_backend(regularize_backend(backend))
    return wrap_layout(backend_layout, behavior, highlevel)
