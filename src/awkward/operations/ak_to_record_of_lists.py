# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext, maybe_posaxis
from awkward._namedaxis import _get_named_axis, _named_axis_to_positional_axis

__all__ = ("to_record_of_lists",)


@high_level_function()
def to_record_of_lists(array, axis=0):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        axis (None or int): Keep records in a common list up to the specified axis.
            None is equivalent to `0` in this case. A list has to be present at the given axis,
            otherwise an error will be raised.

    Converts lists of records to a record of lists. Think of it as applying
    #ak.unzip, but returning an Array instead of a tuple.

    For example, consider this array of lists of lists of records:

        >>> a = ak.Array(
        ...     [
        ...         [[{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]],
        ...         [[{"a": 7, "b": 9}, {"a": 8, "b": 10}], []],
        ...     ]
        ... )
        >>> a.type.show()
        2 * var * var * {
            a: int64,
            b: int64
        }

    Using the default `axis=0`, this will become an array of records of lists of lists:

        >>> ak.to_record_of_lists(a).type.show()
        2 * {
            a: var * var * int64,
            b: var * var * int64
        }

    Using `axis=1`, the outermost list will remain common to both records:

        >>> ak.to_record_of_lists(a, axis=1).type.show()
        2 * var * {
            a: var * int64,
            b: var * int64
        }

    See also #ak.unzip, #ak.to_lists_of_records.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, axis)


def _impl(array, axis):
    with HighLevelContext() as ctx:
        layout = ctx.unwrap(array)

    if axis is not None:
        named_axis = _get_named_axis(ctx)
        axis = _named_axis_to_positional_axis(named_axis, axis)
        axis = maybe_posaxis(layout, axis, 1)

    list_found = False

    def transform(layout, depth, **kwargs):
        nonlocal list_found
        if not layout.is_list:
            return
        if axis is None or depth == axis + 1:
            list_found = True
            return ak.contents.RecordArray(
                ak.unzip(layout, highlevel=False),
                None if layout.is_tuple else layout.fields,
            )

    result = ak.transform(transform, array)
    if not list_found:
        raise ValueError(f"No list found using axis={axis} in the given array")
    return result
