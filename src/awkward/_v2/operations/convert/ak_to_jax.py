# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak


def to_jax(array):
    """
    Converts `array` (many types supported) into a JAX Device Array, if possible.

    If the data are numerical and regular (nested lists have equal lengths
    in each dimension, as described by the #type), they can be losslessly
    converted to a JAX array and this function returns without an error.

    Otherwise, the function raises an error.

    If `array` is a scalar, it is converted into a JAX scalar.

    See also #ak.from_jax and #ak.to_numpy.
    """
    with ak._v2._util.OperationErrorContext(
        "ak._v2.to_jax",
        dict(array=array),
    ):
        from awkward._v2._connect.jax import import_jax

        jax = import_jax().numpy
        return ak._v2._util.to_arraylib(jax, array, True)
