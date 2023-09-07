# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
__all__ = ("from_feather",)
import awkward as ak
from awkward._dispatch import high_level_function


@high_level_function()
def from_feather(
    path,
    *,
    columns=None,
    use_threads=True,
    memory_map=False,  # storage_options?
    generate_bitmasks=False,
    highlevel=True,
    behavior=None,
):
    """
    Args:
        path (str file path, or file-like object): Can be a MemoryMappedFile as
            source, for explicitly use memory map.
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

    Main docs go here (FIXME).

    See also #ak.to_feather
    """

    return _impl(
        path, columns, use_threads, memory_map, generate_bitmasks, highlevel, behavior
    )


def _impl(
    path, columns, use_threads, memory_map, generate_bitmasks, highlevel, behavior
):
    import pyarrow.feather

    # fsspec = awkward._connect.pyarrow.import_fsspec("ak.from_feather")

    # fs = fsspec.open(path)  # fsspec.core.url_to_fs(destination)
    # if not fs.endswith((".fea", ".feather")):
    #     # if
    #     raise ValueError(f"no *.feather or *.fea matches for path {path!r}")

    # with open(path, 'rb') as f:
    #     df = pyarrow_feather.read_feather(f, use_threads, memory_map)

    arrow_table = pyarrow.feather.read_table(path, columns, use_threads, memory_map)

    return ak.operations.ak_from_arrow._impl(
        arrow_table,
        generate_bitmasks,
        highlevel,
        behavior,
    )