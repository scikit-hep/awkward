# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext
from awkward._nplikes.numpy_like import NumpyMetadata

__all__ = ("unzip",)

np = NumpyMetadata.instance()


@high_level_function()
def unzip(
    array,
    *,
    how=tuple,
    highlevel=True,
    behavior=None,
    attrs=None,
):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        how (type): The type of the returned output. This can be `tuple` or `dict`.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    If the `array` contains tuples or records, this operation splits them
    into a Python tuple (or dict) of arrays, one for each field.

    If the `array` does not contain tuples or records, the single `array`
    is placed in a length 1 Python tuple (or dict).

    For example,

        >>> array = ak.Array([{"x": 1.1, "y": [1]},
        ...                   {"x": 2.2, "y": [2, 2]},
        ...                   {"x": 3.3, "y": [3, 3, 3]}])
        >>> x, y = ak.unzip(array)
        >>> x
        <Array [1.1, 2.2, 3.3] type='3 * float64'>
        >>> y
        <Array [[1], [2, 2], [3, 3, 3]] type='3 * var * int64'>

    The `how` argument determines the structure of the output. Using `how=dict`
    returns a dictionary of arrays instead of a tuple, and let's you round-trip
    through `ak.zip`:

        >>> array = ak.Array([{"x": 1.1, "y": [1]},
        ...                   {"x": 2.2, "y": [2, 2]},
        ...                   {"x": 3.3, "y": [3, 3, 3]}])
        >>> x = ak.unzip(array, how=dict)
        >>> x
        {'x': <Array [1.1, 2.2, 3.3] type='3 * float64'>,
         'y': <Array [[1], [2, 2], [3, 3, 3]] type='3 * var * int64'>}
        >>> assert ak.zip(ak.unzip(array, how=dict), depth_limit=1).to_list() == array.to_list()  # True

    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, how, highlevel, behavior, attrs)


def _impl(array, how, highlevel, behavior, attrs):
    if how not in {tuple, dict}:
        raise ValueError(f"`how` must be `tuple` or `dict`, not {how!r}")

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=True, primitive_policy="error")

    fields = ak.operations.fields(layout)

    def check_for_union(layout, **kwargs):
        if isinstance(layout, (ak.contents.RecordArray, ak.Record)):
            return layout  # don't descend into nested records

        elif layout.is_union:
            for content in layout.contents:
                if set(ak.operations.fields(content)) != set(fields):
                    raise ValueError(
                        "union of different sets of fields, cannot ak.unzip"
                    )

    ak._do.recursively_apply(layout, check_for_union, return_array=False)

    if len(fields) == 0:
        items = (ctx.wrap(layout, highlevel=highlevel, allow_other=True),)
        fields = ["0"]
    else:
        items = (
            ctx.wrap(layout[n], highlevel=highlevel, allow_other=True) for n in fields
        )
    return tuple(items) if how is tuple else dict(zip(fields, items))
