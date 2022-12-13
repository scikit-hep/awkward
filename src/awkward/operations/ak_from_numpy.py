# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak


def from_numpy(
    array, *, regulararray=False, recordarray=True, highlevel=True, behavior=None
):
    """
    Args:
        array (np.ndarray): The NumPy array to convert into an Awkward Array.
            This array can be a np.ma.MaskedArray.
        regulararray (bool): If True and the array is multidimensional,
            the dimensions are represented by nested #ak.contents.RegularArray
            nodes; if False and the array is multidimensional, the dimensions
            are represented by a multivalued #ak.contents.NumpyArray.shape.
            If the array is one-dimensional, this has no effect.
        recordarray (bool): If True and the array is a NumPy structured array
            (dtype.names is not None), the fields are represented by an
            #ak.contents.RecordArray; if False and the array is a structured
            array, the structure is left in the #ak.contents.NumpyArray `format`,
            which some functions do not recognize.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Converts a NumPy array into an Awkward Array.

    The resulting layout can only involve the following #ak.contents.Content types:

    * #ak.contents.NumpyArray
    * #ak.contents.ByteMaskedArray or #ak.contents.UnmaskedArray if the
      `array` is an np.ma.MaskedArray.
    * #ak.contents.RegularArray if `regulararray=True`.
    * #ak.contents.RecordArray if `recordarray=True`.

    See also #ak.to_numpy and #ak.from_cupy.
    """
    with ak._errors.OperationErrorContext(
        "ak.from_numpy",
        dict(
            array=array,
            regulararray=regulararray,
            recordarray=recordarray,
            highlevel=highlevel,
            behavior=behavior,
        ),
    ):
        return ak._util.from_arraylib(
            array, regulararray, recordarray, highlevel, behavior
        )
