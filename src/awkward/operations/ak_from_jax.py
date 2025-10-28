# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward import jax
from awkward._dispatch import high_level_function
from awkward._layout import from_arraylib, wrap_layout

__all__ = ("from_jax",)


@high_level_function()
def from_jax(
    array,
    *,
    regulararray=False,
    highlevel=True,
    behavior=None,
    attrs=None,
    primitive_policy="error",
):
    """
    Args:
        array (jax.numpy.DeviceArray): The JAX DeviceArray to convert into an Awkward Array.
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

    Converts a JAX DeviceArray array into an Awkward Array.

    The resulting layout may involve the following #ak.contents.Content types
    (only):

    * #ak.contents.NumpyArray
    * #ak.contents.RegularArray if `regulararray=True`.

    See also #ak.to_jax, #ak.from_numpy and #ak.from_jax.
    """
    jax.assert_registered()
    return wrap_layout(
        from_arraylib(array, regulararray, False, primitive_policy=primitive_policy),
        highlevel=highlevel,
        behavior=behavior,
        attrs=attrs,
    )
