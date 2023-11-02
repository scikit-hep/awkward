# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
__all__ = ("field_attrs",)

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._nplikes.numpylike import NumpyMetadata
from awkward._regularize import is_non_string_like_sequence

np = NumpyMetadata.instance()


@high_level_function()
def field_attrs(array, field):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        field (str or tuple of str): Field name, or series of field names
            to resolve.

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
    return _impl(array, field)


def _impl(array, field):
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

    result_entries = []

    def apply(layout, depth_context, **kwargs):
        path = depth_context["path"]
        if path:
            if layout.is_record:
                head, *tail = path
                ak._do.recursively_apply(
                    layout.content(head),
                    apply,
                    depth_context={"path": tail},
                    return_array=False,
                )
                return layout

        else:
            if layout._parameters is not None:
                result_entries.append(layout._parameters)

            if layout.is_list or layout.is_numpy or layout.is_record:
                return layout
            elif layout.is_unknown:
                raise ValueError
            else:
                return None

    ak._do.recursively_apply(
        layout, apply, depth_context={"path": field}, return_array=False
    )

    from collections import ChainMap

    return ChainMap(*reversed(result_entries))
