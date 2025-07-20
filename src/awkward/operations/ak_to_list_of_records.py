# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function

__all__ = ("to_lists_of_records",)


@high_level_function()
def to_lists_of_records(array, depth_limit=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        depth_limit (int or None): If None, attempt to fully broadcast the
            `array` to all levels. If an int, limit the number of dimensions
            that get broadcasted. The minimum value is `1`, for no
            broadcasting.

    Combines records of lists with common lengths into lists of records.

    This operation may be thought of as applying #ak.zip, but operating on an
    Array instead of a tuple or dictionary.

    For example, consider this array of records of lists:

        >>> a = ak.Array(
        ...     [
        ...         {"a": [1, 2, 3], "b": [4, 5, 6]},
        ...         {"a": [7, 8], "b": [9, 10]},
        ...         {"a": [], "b": []},
        ...     ]
        ... )
        >>> a.type.show()
        3 * {
            a: var * int64,
            b: var * int64
        }

    The result of this operation will be an array of lists of records:

        >>> ak.to_lists_of_records(a).type.show()
        3 * var * {
            a: int64,
            b: int64
        }

    The behavior of `depth_limit` is the same as in #ak.zip, except that counting
    starts from the depth where the record is placed, e.g. in the following example
    `depth_limit=2` will fully broadcast the lists since the record is at depth 1:

        >>> b = ak.Array(
        ...     [
        ...         [{"a": [1, 2, 3], "b": [4, 5, 6]}],
        ...         [{"a": [7, 8], "b": [9, 10]}, {"a": [], "b": []}],
        ...     ]
        ... )
        >>> b.type.show()
        2 * var * {a: var * int64, b: var * int64}
        >>> ak.to_lists_of_records(b, depth_limit=2).type.show()
        2 * var * var * {a: int64, b: int64}

    Also see #ak.zip, #ak.to_record_of_lists.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, depth_limit)


def _impl(array, depth_limit):
    def transform(layout, depth, **kwargs):
        if layout.is_record:
            obj = ak.unzip(layout, highlevel=False)
            if not layout.is_tuple:
                obj = dict(zip(layout.fields, obj))
            return ak.zip(obj, depth_limit=depth_limit, highlevel=False)

    return ak.transform(transform, array)
