# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def from_parquet(
    path,
    columns=None,
    row_groups=None,
    storage_options=None,
    max_gap=64_000,
    max_block=256_000_000,
    footer_sample_size=1_000_000,
    list_indicator="list.item",
    conservative_optiontype=False,
    highlevel=True,
    behavior=None,
):
    """
    Some awesome documentation!
    """
    with ak._v2._util.OperationErrorContext(
        "ak._v2.from_parquet",
        dict(
            path=path,
            columns=columns,
            row_groups=row_groups,
            storage_options=storage_options,
            max_gap=max_gap,
            max_block=max_block,
            footer_sample_size=footer_sample_size,
            list_indicator=list_indicator,
            conservative_optiontype=conservative_optiontype,
            highlevel=highlevel,
            behavior=behavior,
        ),
    ):
        return _impl(
            path,
            columns,
            row_groups,
            storage_options,
            max_gap,
            max_block,
            footer_sample_size,
            list_indicator,
            conservative_optiontype,
            highlevel,
            behavior,
        )


def _impl(
    path,
    columns,
    row_groups,
    storage_options,
    max_gap,
    max_block,
    footer_sample_size,
    list_indicator,
    conservative_optiontype,
    highlevel,
    behavior,
):
    import awkward._v2._connect.pyarrow  # noqa: F401

    name = "ak._v2.from_parquet"
    pyarrow_parquet = ak._v2._connect.pyarrow.import_pyarrow_parquet(name)
    fsspec = ak._v2._connect.pyarrow.import_fsspec(name)

    import fsspec.parquet

    fs, _, paths = fsspec.get_fs_token_paths(
        path, mode="rb", storage_options=storage_options
    )

    all_paths, path_for_metadata = _all_and_metadata_paths(path, fs, paths)

    if row_groups is not None:
        if len(all_paths) == 1:
            row_groups = [row_groups]
        else:
            if len(row_groups) != len(all_paths):
                matched_paths = "\n    ".join(all_paths)
                raise ak._v2._util.error(
                    ValueError(
                        f"""length of row_groups ({len(row_groups)}) does not match length of matched paths (len(all_paths)):

    {matched_paths}"""
                    )
                )
        if not all(all(ak._v2._util.isint(y) for y in x) for x in row_groups):
            raise ak._v2._util.error(
                TypeError("row_groups must be a list of lists of int")
            )

    if columns is None:
        parquet_columns = None

    else:
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
            _, parquet_columns = form.select_columns(
                columns, list_indicator=list_indicator
            )

    arrays = []
    for i, x in enumerate(all_paths):
        if row_groups is None:
            rg = None
        else:
            rg = row_groups[i]

        with fsspec.parquet.open_parquet_file(
            x,
            fs=fs,
            engine="pyarrow",
            columns=parquet_columns,
            row_groups=rg,
            storage_options=storage_options,
            max_gap=max_gap,
            max_block=max_block,
            footer_sample_size=footer_sample_size,
        ) as file:
            parquetfile = pyarrow_parquet.ParquetFile(file)

            if rg is None:
                arrow_table = parquetfile.read(parquet_columns)
            else:
                arrow_table = parquetfile.read_row_groups(rg, parquet_columns)

            arrays.append(
                ak._v2._connect.pyarrow.handle_arrow(
                    arrow_table,
                    conservative_optiontype=conservative_optiontype,
                    pass_empty_field=True,
                )
            )

    return ak._v2.operations.structure.ak_concatenate._impl(
        arrays, 0, True, True, highlevel, behavior
    )


def _all_and_metadata_paths(path, fs, paths):
    all_paths = []
    for x in paths:
        if fs.isfile(x):
            is_meta = x.split("/")[-1] == "_metadata"
            is_comm = x.split("/")[-1] == "_common_metadata"
            all_paths.append((x, is_meta, is_comm))
        elif fs.isdir(x):
            for prefix, _, files in fs.walk(x):
                for f in files:
                    is_meta = f == "_metadata"
                    is_comm = f == "_common_metadata"
                    if f.endswith((".parq", ".parquet")) or is_meta or is_comm:
                        if fs.isfile("/".join((prefix, f))):
                            all_paths.append(("/".join((prefix, f)), is_meta, is_comm))

    path_for_metadata = [x for x, is_meta, is_comm in all_paths if is_meta]
    if len(path_for_metadata) != 0:
        path_for_metadata = path_for_metadata[0]
    else:
        path_for_metadata = [x for x, is_meta, is_comm in all_paths if is_comm]
        if len(path_for_metadata) != 0:
            path_for_metadata = path_for_metadata[0]
        else:
            if len(all_paths) != 0:
                path_for_metadata = all_paths[0][0]

    all_paths = [x for x, is_meta, is_comm in all_paths if not is_meta and not is_comm]

    if len(all_paths) == 0:
        raise ak._v2._util.error(
            ValueError(f"no *.parquet or *.parq matches for path {path!r}")
        )

    return all_paths, path_for_metadata
