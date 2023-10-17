# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext
from awkward._nplikes.numpy_like import NumpyMetadata

__all__ = ("with_parameter",)

np = NumpyMetadata.instance()


@high_level_function()
def with_parameter(
    array, parameter, value, *, highlevel=True, behavior=None, attrs=None
):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        parameter (str): Name of the parameter to set on that array.
        value (JSON): Value of the parameter to set on that array.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    This function returns a new array with a parameter set on the outermost
    node of its #ak.Array.layout.

    Note that a "new array" is a lightweight shallow copy, not a duplication
    of large data buffers.

    You can also remove a single parameter with this function, since setting
    a parameter to None is equivalent to removing it.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, parameter, value, highlevel, behavior, attrs)


def _impl(array, parameter, value, highlevel, behavior, attrs):
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=True, primitive_policy="error")

    out = layout.with_parameter(parameter, value)
    return ctx.wrap(out, highlevel=highlevel)
