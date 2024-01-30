from __future__ import annotations

from os import fsdecode, path
import awkward as ak
from awkward._dispatch import high_level_function

__all__ = ("to_parquet_dataset",)


@high_level_function()
def to_parquet_dataset(directory, filenames=None, filename_extension=".parquet", storage_options=None,):
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

    return _impl(directory, filenames, filename_extension, storage_options)


def _impl(directory, filenames, filename_extension, storage_options):
    # Implementation

    import awkward._connect.pyarrow

    pyarrow_parquet = awkward._connect.pyarrow.import_pyarrow_parquet(
        "ak.to_parquet_dataset"
    )
    fsspec = awkward._connect.pyarrow.import_fsspec("ak.to_parquet")
    import fsspec.parquet

    # fsspec.parquet.
    try:
        directory = fsdecode(directory)
    except TypeError:
        raise TypeError(
            f"'directory' argument of 'ak.to_parquet_dataset' must be a path-like, not {type(directory).__name__}"
        ) from None
    
    print(directory)
    fs, directory = fsspec.core.url_to_fs(directory, **(storage_options or {}))
    # fs, _, path = fsspec.get_fs_token_paths(
    #     directory, mode="rb", storage_options=storage_options
    print(directory) #full directory path
    # )
    if not fs.isdir(directory): # ?
        raise ValueError(
            f"{directory!r} is not a directory" + {__file__}
        )


    if filenames is None:
        import glob

        filenames = sorted(
            glob.glob(path + f"/**/*{filename_extension}", recursive=True)
        )

    else:
        filenames = [x for x in filenames]
        # filenames = [_regularize_path(x) for x in filenames]
        # filenames = [x if fs.path.isabs(x) else fs.path.combine(directory, x) for x in filenames] 
        # Combine ^^ only works if the second path is relative and there are no back references in either path...

    relpaths = [fs.path.relpath(x, directory) for x in filenames] # This sure seems to only apply to local...


    
    # schema, metadata_collector = _common_parquet_schema(
    #     pyarrow_parquet, filenames, paths
    # )

    assert len(filenames) != 0

    schema = None
    metadata_collector = []
    for filename, path in zip(filenames, relpaths):
        if schema is None:
            schema = pyarrow_parquet.ParquetFile(filename).schema_arrow
            first_filename = filename
        elif not schema.equals(pyarrow_parquet.ParquetFile(filename).schema_arrow):
            raise ValueError(
                "schema in {} differs from the first schema (in {})".format(
                    repr(filename), repr(first_filename)
                )
            )
        metadata_collector.append(pyarrow_parquet.ParquetFile(filename).metadata)
        metadata_collector[-1].set_file_path(path)

    pyarrow_parquet.write_metadata(schema, path.join(directory, "_common_metadata"))
    pyarrow_parquet.write_metadata(
        schema,
        path.join(directory, "_metadata"),
        metadata_collector=metadata_collector,
    )


# def _regularize_path(path):


#     return path


# def _common_parquet_schema(pq, filenames, paths):
#     assert len(filenames) != 0

#     schema = None
#     metadata_collector = []
#     for filename, path in zip(filenames, paths):
#         if schema is None:
#             schema = pq.ParquetFile(filename).schema_arrow
#             first_filename = filename
#         elif not schema.equals(pq.ParquetFile(filename).schema_arrow):
#             raise ValueError(
#                 "schema in {} differs from the first schema (in {})".format(
#                     repr(filename), repr(first_filename)
#                 )
#             )
#         metadata_collector.append(pyarrow_parquet.ParquetFile(f).metadata)
#         metadata_collector[-1].set_file_path(path)
#     return schema, metadata_collector
    