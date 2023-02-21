# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak


def from_rdataframe(
    rdf,
    columns,
    offsets_type="int64_t",
    keep_order=False,
    *,
    highlevel=True,
    behavior=None,
):
    """
    Args:
        rdf (`ROOT.RDataFrame`): ROOT RDataFrame to convert into an
            Awkward Array.
        columns (str or iterable of str): A column or multiple columns to be
            converted to Awkward Array.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        offsets_type (str): A C++ type of the ListOffsetArray offsets.
        keep_order (bool): If set to `True` the columns with Awkward type will
            keep order after filtering.

    Converts ROOT RDataFrame columns into an Awkward Array.

    If `columns` is a string, the return value represents a single RDataFrame column.

    If `columns` is any other iterable, the return value is a record array, in which
    each field corresponds to an RDataFrame column. In particular, if the `columns`
    iterable contains only one string, it is still a record array, which has only
    one field.

    See also #ak.to_rdataframe.
    """
    with ak._errors.OperationErrorContext(
        "ak.from_rdataframe",
        {
            "rdf": rdf,
            "columns": columns,
            "highlevel": highlevel,
            "behavior": behavior,
            "offsets_type": offsets_type,
            "keep_order": keep_order,
        },
    ):
        return _impl(rdf, columns, highlevel, behavior, offsets_type, keep_order)


def _impl(data_frame, columns, highlevel, behavior, offsets_type, keep_order):
    import awkward._connect.rdataframe.from_rdataframe  # noqa: F401

    if isinstance(columns, str):
        columns = (columns,)
        project = True
    else:
        columns = tuple(columns)
        project = False

    if not all(isinstance(x, str) for x in columns):
        raise ak._errors.wrap_error(
            TypeError(
                f"'columns' must be a string or an iterable of strings, not {columns!r}"
            )
        )

    out = ak._connect.rdataframe.from_rdataframe.from_rdataframe(
        data_frame,
        columns,
        highlevel=highlevel,
        behavior=behavior,
        offsets_type=offsets_type,
        keep_order=keep_order,
    )

    if project:
        return out[columns[0]]
    else:
        return out
