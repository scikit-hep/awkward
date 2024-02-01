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
    """
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

    Reads an Feather file as an Awkward Array (through pyarrow).

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
    import pyarrow.feather

    arrow_table = pyarrow.feather.read_table(path, columns, use_threads, memory_map)

    return ak.operations.ak_from_arrow._impl(
        arrow_table,
        generate_bitmasks,
        highlevel,
        behavior,
        attrs,
    )
