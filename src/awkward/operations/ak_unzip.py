# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()


def unzip(array, *, highlevel=True, behavior=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    If the `array` contains tuples or records, this operation splits them
    into a Python tuple of arrays, one for each field.

    If the `array` does not contain tuples or records, the single `array`
    is placed in a length 1 Python tuple.

    For example,

        >>> array = ak.Array([{"x": 1.1, "y": [1]},
        ...                   {"x": 2.2, "y": [2, 2]},
        ...                   {"x": 3.3, "y": [3, 3, 3]}])
        >>> x, y = ak.unzip(array)
        >>> x
        <Array [1.1, 2.2, 3.3] type='3 * float64'>
        >>> y
        <Array [[1], [2, 2], [3, 3, 3]] type='3 * var * int64'>
    """
    with ak._errors.OperationErrorContext(
        "ak.unzip",
        dict(array=array, highlevel=highlevel, behavior=behavior),
    ):
        return _impl(array, highlevel, behavior)


def _impl(array, highlevel, behavior):
    behavior = ak._util.behavior_of(array, behavior=behavior)
    layout = ak.operations.to_layout(array, allow_record=True, allow_other=False)
    fields = ak.operations.fields(layout)

    def check_for_union(layout, **kwargs):
        if isinstance(layout, (ak.contents.RecordArray, ak.Record)):
            pass  # don't descend into nested records

        elif isinstance(layout, ak.contents.UnionArray):
            for content in layout.contents:
                if set(ak.operations.fields(content)) != set(fields):
                    raise ak._errors.wrap_error(
                        ValueError("union of different sets of fields, cannot ak.unzip")
                    )

        elif hasattr(layout, "content"):
            check_for_union(layout.content)

    ak._do.recursively_apply(layout, check_for_union, behavior, return_array=False)

    if len(fields) == 0:
        return (ak._util.wrap(layout, behavior, highlevel),)
    else:
        return tuple(ak._util.wrap(layout[n], behavior, highlevel) for n in fields)
