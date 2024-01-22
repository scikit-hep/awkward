from __future__ import annotations

__all__ = ("to_parquet_dataset",)


def to_parquet_dataset(directory, filenames=None, filename_extension=".parquet"):
    """
    Args:
        directory (str or Path): A local directory in which to write `_common_metadata`
            and `_metadata`, making the directory of Parquet files into a dataset.
        filenames (None or list of str or Path): If None, the `directory` will be
            recursively searched for files ending in `filename_extension` and
            sorted lexicographically. Otherwise, this explicit list of files is
            taken and row-groups are concatenated in its given order. If any
            filenames are relative, they are interpreted relative to `directory`.
        filename_extension (str): Filename extension (including `.`) to use to
            search for files recursively. Ignored if `filenames` is None.

    Creates a `_common_metadata` and a `_metadata` in a directory of Parquet files.

    The `_common_metadata` contains the schema that all files share. (If the files
    have different schemas, this function raises an exception.)

    The `_metadata` contains row-group metadata used to seek to specific row-groups
    within the multi-file dataset.
    """

    # Implementation

    from os import fsdecode, path

    import awkward._connect.pyarrow

    pyarrow_parquet = awkward._connect.pyarrow.import_pyarrow_parquet(
        "ak.to_parquet_dataset"
    )

    try:  # Second is probably better...
        directory = fsdecode(directory)
    except TypeError:
        raise TypeError(
            f"'directory' argument of 'ak.to_parquet_dataset' must be a path-like, not {type(directory).__name__}"
        ) from None

    directory = _regularize_path(directory)
    if not path.isdir(directory):
        raise ValueError(
            f"{directory!r} is not a local filesystem directory" + {__file__}
        )

    if filenames is None:
        import glob

        filenames = sorted(
            glob.glob(directory + f"/**/*{filename_extension}", recursive=True)
        )
    else:
        filenames = [_regularize_path(x) for x in filenames]
        filenames = [x if path.isabs(x) else path.join(directory, x) for x in filenames]

    relpaths = [path.relpath(x, directory) for x in filenames]
    schema, metadata_collector = _common_parquet_schema(
        pyarrow_parquet, filenames, relpaths
    )
    pyarrow_parquet.write_metadata(schema, path.join(directory, "_common_metadata"))
    pyarrow_parquet.write_metadata(
        schema,
        path.join(directory, "_metadata"),
        metadata_collector=metadata_collector,
    )


def _regularize_path(path):
    import os

    if isinstance(path, getattr(os, "PathLike", ())):
        path = os.fspath(path)

    elif hasattr(path, "__fspath__"):
        path = os.fspath(path)

    elif path.__class__.__module__ == "pathlib":
        import pathlib

        if isinstance(path, pathlib.Path):
            path = str(path)

    if isinstance(path, str):
        path = os.path.expanduser(path)

    return path


def _common_parquet_schema(pq, filenames, relpaths):
    assert len(filenames) != 0

    schema = None
    metadata_collector = []
    for filename, relpath in zip(filenames, relpaths):
        if schema is None:
            schema = pq.ParquetFile(filename).schema_arrow
            first_filename = filename
        elif not schema.equals(pq.ParquetFile(filename).schema_arrow):
            raise ValueError(
                "schema in {} differs from the first schema (in {})".format(
                    repr(filename), repr(first_filename)
                )
            )
        metadata_collector.append(pq.read_metadata(filename))
        metadata_collector[-1].set_file_path(relpath)
    return schema, metadata_collector
