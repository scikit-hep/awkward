# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function

__all__ = ("from_feather",)


@high_level_function()
def from_feather(
    path,
    *,
    columns=None,
    use_threads=True,
    memory_map=False,
    generate_bitmasks=False,
    highlevel=True,
    behavior=None,
    attrs=None,
):
    """Reads a Feather file as an Awkward Array (through pyarrow).

    Args:
        path (str or file-like object): Feather file to read as an Awkward Array,
            passed directly to [pyarrow.feather.read_table](https://arrow.apache.org/docs/python/generated/pyarrow.feather.read_table.html).
        columns (sequence): Only read a specific set of columns. If not provided,
            all columns are read.
        use_threads (bool): If True, parallelize reading using multiple threads.
        memory_map (bool): If True, use memory mapping when opening file on disk,
            when source is a string.
        generate_bitmasks (bool): If enabled and Arrow/Parquet does not have Awkward
            metadata, `generate_bitmasks=True` creates empty bitmasks for nullable
            types that don't have bitmasks in the Arrow/Parquet data, so that the
            Form (BitMaskedForm vs UnmaskedForm) is predictable.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Returns:
        An #ak.Array read from the given Feather file (through pyarrow).

    Examples:
        >>> ak.from_feather("file_name.feather")
        <Array [[1.1, 2.2, 3.3], [], [4.4, 5.5]] type='3 * var * float64'>


        See also #ak.to_feather.
    """

    return _impl(
        path,
        columns,
        use_threads,
        memory_map,
        generate_bitmasks,
        highlevel,
        behavior,
        attrs,
    )


def _impl(
    path,
    columns,
    use_threads,
    memory_map,
    generate_bitmasks,
    highlevel,
    behavior,
    attrs,
):
    import pyarrow
    import pyarrow.ipc

    # Read via the non-deprecated pyarrow.ipc API.
    # This reads the same on-disk format as the old
    # pyarrow.feather.read_table call did - a different
    # (non-deprecated) API for reading the same bytes, not a different
    # format.
    # A fundamental difference in the way we are reading all columns then dropping the non-selected ones

    with (
        pyarrow.memory_map(path, "r")
        if memory_map
        else pyarrow.OSFile(path, "rb") as source
    ):
        reader = pyarrow.ipc.open_file(
            source,
            options=pyarrow.ipc.IpcReadOptions(
                use_threads=use_threads,
            ),
        )
        arrow_table = reader.read_all()
    if columns is not None:
        arrow_table = arrow_table.select(columns)

    return ak.operations.ak_from_arrow._impl(
        arrow_table,
        generate_bitmasks,
        highlevel,
        behavior,
        attrs,
    )
