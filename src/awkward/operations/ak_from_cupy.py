# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._dispatch import high_level_function
from awkward._layout import from_arraylib, wrap_layout

__all__ = ("from_cupy",)


@high_level_function()
def from_cupy(array, *, regulararray=False, highlevel=True, behavior=None, attrs=None):
    """
    Args:
        array (cp.ndarray): The CuPy array to convert into an Awkward Array.
        regulararray (bool): If True and the array is multidimensional,
            the dimensions are represented by nested #ak.contents.RegularArray
            nodes; if False and the array is multidimensional, the dimensions
            are represented by a multivalued #ak.contents.NumpyArray.shape.
            If the array is one-dimensional, this has no effect.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Converts a CuPy array into an Awkward Array.

    The resulting layout may involve the following #ak.contents.Content types
    (only):

    * #ak.contents.NumpyArray
    * #ak.contents.RegularArray if `regulararray=True`.

    See also #ak.to_cupy, #ak.from_numpy and #ak.from_jax.
    """
    return wrap_layout(
        from_arraylib(array, regulararray, False),
        highlevel=highlevel,
        behavior=behavior,
    )
