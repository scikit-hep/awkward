from __future__ import annotations

from awkward._dispatch import high_level_function
import os

__all__ = ("to_parquet_dataset",)


@high_level_function()
def to_parquet_dataset(
    directory,
    filenames=None,
    storage_options=None,
):
    """
    Args:
        directory (str or Path): A directory in which to write `_common_metadata`
            and `_metadata`, making the directory of Parquet files into a dataset.
        filenames (None or list of str or Path): If None, the `directory` will be
            recursively searched for files ending in `filename_extension` and
            sorted lexicographically. Otherwise, this explicit list of files is
            taken and row-groups are concatenated in its given order. If any
            filenames are relative, they are interpreted relative to `directory`.
        filename_extension (str): Filename extension (including `.`) to use to
            search for files recursively. Ignored if `filenames` is None.

    Creates a `_common_metadata` and a `_metadata` in a directory of Parquet files.

        >>> ak.to_parquet(array1, "/directory/arr1.parquet", parquet_compliant_nested=True)
        >>> ak.to_parquet(array2, "/directory/arr2.parquet", parquet_compliant_nested=True)
        >>> ak.to_parquet_dataset("/directory")

    The `_common_metadata` contains the schema that all files share. (If the files
    have different schemas, this function raises an exception.)

    The `_metadata` contains row-group metadata used to seek to specific row-groups
    within the multi-file dataset.
    """

    return _impl(directory, filenames, storage_options)


def _impl(directory, filenames, storage_options):
    # Implementation
    import awkward._connect.pyarrow

    pyarrow_parquet = awkward._connect.pyarrow.import_pyarrow_parquet(
        "ak.to_parquet_dataset"
    )
    fsspec = awkward._connect.pyarrow.import_fsspec("ak.to_parquet")
    import fsspec.parquet

    fs, _, paths = fsspec.get_fs_token_paths(
        directory, mode="rb", storage_options=storage_options
    )
    if not fs.isdir(directory):
        raise ValueError(f"{directory!r} is not a directory" + {__file__})
    filepaths = []
    
    if filenames is not None:
        filenames = [os.path.join(directory, fname) for fname in filenames]
        for x in paths:  # paths should always be a list even if there is just one
            for f, fdata in fs.find(x, detail=True).items():
                if f.endswith((".parq", ".parquet")) and f in filenames:
                    if fdata["type"] == "file":
                        filepaths.append(f)

        if len(filepaths) == 0:
            raise ValueError(f"no *.parquet or *.parq matches for path {directory!r}")

    else:  # Ask about this...
        # Get paths:
        for x in paths:  # paths should always be a list even if there is just one
            for f, fdata in fs.find(x, detail=True).items():
                if f.endswith((".parq", ".parquet")):
                    if fdata["type"] == "file":
                        filepaths.append(f)

        if len(filepaths) == 0:
            raise ValueError(f"no *.parquet or *.parq matches for path {directory!r}")

    assert len(filepaths) != 0
    schema = None
    metadata_collector = []
    for filepath in filepaths:
        if schema is None:
            schema = pyarrow_parquet.ParquetFile(filepath).schema_arrow
            first_filepath = filepath
        elif not schema.equals(pyarrow_parquet.ParquetFile(filepath).schema_arrow):
            raise ValueError(
                "schema in {} differs from the first schema (in {})".format(
                    repr(filepath), repr(first_filepath)
                )
            )
        metadata_collector.append(pyarrow_parquet.ParquetFile(filepath).metadata)
        metadata_collector[-1].set_file_path(filepath)
    _common_metadata_path = os.path.join(directory, "_common_metadata")
    pyarrow_parquet.write_metadata(schema, _common_metadata_path)
    _metadata_path = os.path.join(directory, "_metadata")
    pyarrow_parquet.write_metadata(
        schema,
        _metadata_path,
        metadata_collector=metadata_collector,
    )
    return _common_metadata_path, _metadata_path
