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
    generate_bitmasks,
):
    """
    Args:
        path: str file path, or file-like object.
        Can be a MemoryMappedFile
        as source, for explicitly use memory map.

        columns: sequence, optional
        Only read a specific set of columns. If not provided, all columns are read.

        use_threads: bool, default True
        Whether to parallelize reading using multiple threads.

        memory_map: bool, default False
        Use memory mapping when opening file on disk, when source is a str

    Returns:
        df: pandas.dataframe
        The contents of the Feather file as a pyarrow.Table


    See also #ak.to_feather
    """

    return _impl(path, columns, use_threads, memory_map, generate_bitmasks)


def _impl(path, columns, use_threads, memory_map, generate_bitmasks):
    import pyarrow.feather

    import awkward._connect.pyarrow

    fsspec = awkward._connect.pyarrow.import_fsspec("ak.from_feather")

    fs = fsspec.open(path)  # fsspec.core.url_to_fs(destination)
    if not fs.endswith((".fea", ".feather")):
        # if
        raise ValueError(f"no *.feather or *.fea matches for path {path!r}")

    # with open(path, 'rb') as f:
    #     df = pyarrow_feather.read_feather(f, use_threads, memory_map)

    df_pandas = pyarrow.feather.read_feather(path, columns, use_threads, memory_map)
    arrow_table = pyarrow.Table.from_pandas(df_pandas)
    df = ak.operations.ak_from_arrow._impl(
        arrow_table,
        generate_bitmasks,
        # why is high-level False here?
        False,
        None,
    )
    return df
