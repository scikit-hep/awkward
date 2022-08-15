# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak


def from_cupy(array, regulararray=False, highlevel=True, behavior=None):
    """
    Args:
        array (cp.ndarray): The CuPy array to convert into an Awkward Array.
        regulararray (bool): If True and the array is multidimensional,
            the dimensions are represented by nested #ak.layout.RegularArray
            nodes; if False and the array is multidimensional, the dimensions
            are represented by a multivalued #ak.layout.NumpyArray.shape.
            If the array is one-dimensional, this has no effect.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Converts a CuPy array into an Awkward Array.

    The resulting layout may involve the following #ak.layout.Content types
    (only):

       * #ak.layout.NumpyArray
       * #ak.layout.RegularArray if `regulararray=True`.

    See also #ak.to_cupy, #ak.from_numpy and #ak.from_jax.
    """
    with ak._v2._util.OperationErrorContext(
        "ak._v2.from_cupy",
        dict(
            array=array,
            regulararray=regulararray,
            highlevel=highlevel,
            behavior=behavior,
        ),
    ):
        return ak._v2._util.from_arraylib(
            array, regulararray, False, highlevel, behavior
        )
