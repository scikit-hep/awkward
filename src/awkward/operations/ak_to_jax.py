# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
__all__ = ("to_jax",)
import awkward as ak
from awkward._backends.jax import JaxBackend
from awkward._dispatch import high_level_function


@high_level_function()
def to_jax(array):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).

    Converts `array` (many types supported) into a JAX Device Array, if possible.

    If the data are numerical and regular (nested lists have equal lengths
    in each dimension, as described by the #ak.Array.type), they can be losslessly
    converted to a JAX array and this function returns without an error.

    Otherwise, the function raises an error.

    If `array` is a scalar, it is converted into a JAX scalar.

    See also #ak.from_jax and #ak.to_numpy.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array)


def _impl(array):
    layout = ak.to_layout(array, allow_record=False)

    backend = JaxBackend.instance()
    numpy_layout = layout.to_backend(backend)

    return numpy_layout.to_backend_array(allow_missing=False)
