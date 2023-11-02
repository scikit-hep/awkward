# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
__all__ = ("with_field_attr",)

import awkward as ak
from awkward._behavior import behavior_of
from awkward._dispatch import high_level_function
from awkward._layout import wrap_layout
from awkward._nplikes.numpylike import NumpyMetadata
from awkward._regularize import is_non_string_like_sequence

np = NumpyMetadata.instance()


@high_level_function()
def with_field_attr(array, field, parameter, value, *, highlevel=True, behavior=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        field (str or tuple of str): Field name, or series of field names
            to resolve.
        parameter (str): Name of the parameter to set on that array.
        value (JSON): Value of the parameter to set on that array.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    This function returns a new array with a parameter set on the deepest
    node with the given field name in the #ak.Array.layout.

    Note that a "new array" is a lightweight shallow copy, not a duplication
    of large data buffers.

    You can also remove a single parameter with this function, since setting
    a parameter to None is equivalent to removing it.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, field, parameter, value, highlevel, behavior)


def _impl(array, field, parameter, value, highlevel, behavior):
    behavior = behavior_of(array, behavior=behavior)
    layout = ak.operations.to_layout(
        array, allow_record=True, allow_unknown=False, primitive_policy="error"
    )
    if not (
        isinstance(field, str)
        or (
            is_non_string_like_sequence(field)
            and all(isinstance(x, str) for x in field)
        )
    ):
        raise TypeError("Field attributes may only be assigned by field name(s) ")

    def apply(layout, depth_context, continuation, **kwargs):
        path = depth_context["path"]
        if path:
            if layout.is_record:
                head, *tail = path
                return layout.copy(
                    contents=[
                        ak._do.recursively_apply(
                            content, apply, depth_context={"path": tail}
                        )
                        if field == head
                        else content
                        for field, content in zip(layout.fields, layout.contents)
                    ]
                )
        elif not path:
            if layout.is_list or layout.is_numpy or layout.is_record:
                return layout.with_parameter(parameter, value)
            if layout.is_unknown:
                raise ValueError

            if layout.parameter(parameter) is not None:
                return continuation().with_parameter(parameter, None)

    out = ak._do.recursively_apply(layout, apply, depth_context={"path": field})

    return wrap_layout(out, behavior_of(array, behavior=behavior), highlevel)
