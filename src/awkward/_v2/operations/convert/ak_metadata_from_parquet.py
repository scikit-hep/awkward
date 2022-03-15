# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import collections

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


ParquetMetadata = collections.namedtuple(
    "ParquetMetadata",
    ["form", "fs", "paths", "metadata"],
)


def metadata_from_parquet(
    path,
    storage_options=None,
    max_gap=64_000,
    max_block=256_000_000,
    footer_sample_size=1_000_000,
):
    """
    Args:
        path (str): Local filename or remote URL, passed to fsspec for resolution.
            May contain glob patterns.
        storage_options: Passed to `fsspec.parquet.open_parquet_file`.
        max_gap (int): Passed to `fsspec.parquet.open_parquet_file`.
        max_block (int): Passed to `fsspec.parquet.open_parquet_file`.
        footer_sample_size (int): Passed to `fsspec.parquet.open_parquet_file`.

    Returns a named tuple containing

      * `form`: an Awkward Form representing the low-level type of the data
         (use `.type` to get a high-level type),
      * `fs`: the fsspec filesystem object,
      * `paths`: a list of matching path names,
      * `metadata`: the Parquet metadata, which includes `.num_rows` for the length
         of the array that would be read by #ak.from_parquet and `.num_row_groups`
         for the units that can be filtered (for the #ak.from_parquet `row_groups`
         argument).

    See also #ak.from_parquet, #ak.to_parquet.
    """
    with ak._v2._util.OperationErrorContext(
        "ak._v2.metadata_from_parquet",
        dict(
            path=path,
            storage_options=storage_options,
            max_gap=max_gap,
            max_block=max_block,
            footer_sample_size=footer_sample_size,
        ),
    ):
        return _impl(
            path,
            storage_options,
            max_gap,
            max_block,
            footer_sample_size,
        )


def _impl(
    path,
    storage_options,
    max_gap,
    max_block,
    footer_sample_size,
):
    import awkward._v2._connect.pyarrow  # noqa: F401

    name = "ak._v2.from_parquet"
    pyarrow_parquet = ak._v2._connect.pyarrow.import_pyarrow_parquet(name)
    fsspec = ak._v2._connect.pyarrow.import_fsspec(name)

    import fsspec.parquet

    fs, _, paths = fsspec.get_fs_token_paths(
        path, mode="rb", storage_options=storage_options
    )

    (
        all_paths,
        path_for_metadata,
    ) = ak._v2.operations.convert.ak_from_parquet._all_and_metadata_paths(
        path, fs, paths
    )

    with fsspec.parquet.open_parquet_file(
        path_for_metadata,
        fs=fs,
        engine="pyarrow",
        row_groups=[],
        storage_options=storage_options,
        max_gap=max_gap,
        max_block=max_block,
        footer_sample_size=footer_sample_size,
    ) as file_for_metadata:
        parquetfile_for_metadata = pyarrow_parquet.ParquetFile(file_for_metadata)

        form = ak._v2._connect.pyarrow.form_handle_arrow(
            parquetfile_for_metadata.schema_arrow, pass_empty_field=True
        )

        return ParquetMetadata(form, fs, all_paths, parquetfile_for_metadata.metadata)
