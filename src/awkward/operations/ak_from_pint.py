# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
__all__ = ("from_pint",)
from awkward._dispatch import high_level_function
from awkward._do import recursively_apply
from awkward._layout import from_arraylib, wrap_layout
from awkward.operations.ak_with_parameter import with_parameter


@high_level_function
def from_pint(
    array, *, regulararray=False, recordarray=True, highlevel=True, behavior=None
):
    """
    Args:
        array (pint.Quantity): The Pint array to convert into an Awkward Array.
            This Quantity can contain np.ma.MaskedArray.
        regulararray (bool): If True and the array is multidimensional,
            the dimensions are represented by nested #ak.contents.RegularArray
            nodes; if False and the array is multidimensional, the dimensions
            are represented by a multivalued #ak.contents.NumpyArray.shape.
            If the array is one-dimensional, this has no effect.
        recordarray (bool): If True and the wrapped array is a NumPy structured array
            (dtype.names is not None), the fields are represented by an
            #ak.contents.RecordArray; if False and the array is a structured
            array, the structure is left in the #ak.contents.NumpyArray `format`,
            which some functions do not recognize.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Converts a Pint.Quantity array into an Awkward Array.

    The resulting layout can only involve the following #ak.contents.Content types:

    * #ak.contents.NumpyArray
    * #ak.contents.ByteMaskedArray or #ak.contents.UnmaskedArray if the
      `array` is an np.ma.MaskedArray.
    * #ak.contents.RegularArray if `regulararray=True`.
    * #ak.contents.RecordArray if `recordarray=True`.

    See also #ak.to_numpy and #ak.from_cupy.
    """
    layout = from_arraylib(array.magnitude, regulararray, recordarray)
    units = str(array.units)

    def apply(layout, **kwargs):
        if layout.is_numpy:
            return with_parameter(layout, "__units__", units, highlevel=False)

    out = recursively_apply(layout, apply, behavior=behavior)
    return wrap_layout(
        out,
        highlevel=highlevel,
        behavior=behavior,
    )
