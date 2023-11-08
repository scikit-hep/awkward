# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._nplikes.numpy_like import NumpyMetadata

__all__ = ("from_rdataframe",)

np = NumpyMetadata.instance()


@high_level_function()
def from_rdataframe(
    rdf,
    columns,
    *,
    keep_order=False,
    offsets_type="int64",
    with_name=None,
    highlevel=True,
    behavior=None,
    attrs=None,
):
    """
    Args:
        rdf (`ROOT.RDataFrame`): ROOT RDataFrame to convert into an
            Awkward Array.
        columns (str or iterable of str): A column or multiple columns to be
            converted to Awkward Array.
        keep_order (bool): If set to `True` the columns with Awkward type will
            keep order after filtering.
        offsets_type (str): A `NumpyType.primitive` type of the ListOffsetArray
            offsets: `"int32"`, `"uint32"` or `"int64"`.
        with_name (None or str): Gives tuples and records a name that can be
            used to override their behavior (see #ak.Array).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Converts ROOT RDataFrame columns into an Awkward Array.

    If `columns` is a string, the return value represents a single RDataFrame column.

    If `columns` is any other iterable, the return value is a record array, in which
    each field corresponds to an RDataFrame column. In particular, if the `columns`
    iterable contains only one string, it is still a record array, which has only
    one field.

    See also #ak.to_rdataframe.
    """
    return _impl(rdf, columns, highlevel, behavior, with_name, offsets_type, keep_order)


def _impl(
    data_frame, columns, highlevel, behavior, with_name, offsets_type, keep_order
):
    import awkward._connect.rdataframe.from_rdataframe  # noqa: F401

    if isinstance(columns, str):
        columns = (columns,)
        project = True
    else:
        columns = tuple(columns)
        project = False

    if not all(isinstance(x, str) for x in columns):
        raise TypeError(
            f"'columns' must be a string or an iterable of strings, not {columns!r}"
        )

    if not isinstance(offsets_type, str) or offsets_type not in (
        "int32",
        "uint32",
        "int64",
    ):
        raise TypeError(
            "'offsets_type' must be a string in (int32, uint32, int64), "
            f"not {offsets_type!r}"
        )
    else:
        offsets_type = f"{offsets_type}_t"

    out = ak._connect.rdataframe.from_rdataframe.from_rdataframe(
        data_frame,
        columns,
        highlevel=highlevel,
        behavior=behavior,
        with_name=with_name,
        offsets_type=offsets_type,
        keep_order=keep_order,
    )

    if project:
        return out[columns[0]]
    else:
        return out
