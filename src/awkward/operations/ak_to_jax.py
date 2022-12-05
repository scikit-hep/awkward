# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from awkward import _errors, _util, jax


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
    with _errors.OperationErrorContext(
        "ak.to_jax",
        dict(array=array),
    ):
        return _util.to_arraylib(jax.import_jax().numpy, array, True)
