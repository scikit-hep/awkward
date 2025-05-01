# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import collections

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._nplikes.numpy_like import NumpyMetadata

__all__ = ("metadata_from_parquet",)

np = NumpyMetadata.instance()

ParquetMetadata = collections.namedtuple(
    "ParquetMetadata",
    ["form", "fs", "paths", "metadata"],
)


@high_level_function()
def metadata_from_parquet(
    path,
    *,
    storage_options=None,
    row_groups=None,
    ignore_metadata=False,
    scan_files=True,
):
    """
    Args:
        path (str): Local filename or remote URL, passed to fsspec for resolution.
            May contain glob patterns. A list of paths is also allowed, but they
            must be data files, not directories.
        storage_options: Passed to `fsspec.parquet.open_parquet_file`.
        row_groups (None or set of int): Row groups to read; must be non-negative.
            Order is ignored: the output array is presented in the order specified
            by Parquet metadata. If None, all row groups/all rows are read.
        ignore_metadata (bool): ignore the dedicated _metadata file if found
            and instead derive metadata from the first data file.
        scan_files (bool): TODO

    This function differs from ak.from_parquet._metadata as follows:

    * this function will always use a _metadata file, if present
    * if there is no _metadata, the schema comes from _common_metadata or
      the first data file
    * the total number of rows is always known

    Returns dict containing

    * `form`: an Awkward Form representing the low-level type of the data
      (use `.type` to get a high-level type),
    * `fs`: the fsspec filesystem object,
    * `paths`: a list of matching path names,
    * `col_counts`: the number of rows in each row group,
    * `columns`: the columns defined by the schema,
    * `num_rows`: the length of the array that would be read by #ak.from_parquet,
    * `num_row_groups`: the units that can be filtered (for the #ak.from_parquet `row_groups`
      argument).

    See also #ak.from_parquet, #ak.to_parquet.
    """
    import awkward._connect.pyarrow  # noqa: F401

    return _impl(
        path,
        storage_options,
        row_groups=row_groups,
        ignore_metadata=ignore_metadata,
        scan_files=scan_files,
    )


def _impl(
    path, storage_options, row_groups=None, ignore_metadata=False, scan_files=True
):
    results = ak.operations.ak_from_parquet.metadata(
        path,
        storage_options,
        row_groups,
        None,
        ignore_metadata,
        scan_files,
        calculate_uuid=True,
    )
    parquet_columns, subform, actual_paths, fs, subrg, col_counts, metadata, uuid = (
        results
    )

    out = {
        "form": subform,
        "fs": fs,
        "paths": actual_paths,
        "col_counts": col_counts,
        "columns": parquet_columns,
        "uuid": uuid,
    }
    if col_counts:
        out["num_rows"] = sum(col_counts)
        out["num_row_groups"] = len(col_counts)
    else:
        out["num_rows"] = None
        out["num_row_groups"] = None
    return out
