# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak


def from_parquet(
    path,
    columns=None,
    row_groups=None,
    storage_options=None,
    max_gap=64_000,
    max_block=256_000_000,
    footer_sample_size=1_000_000,
    generate_bitmasks=False,
    highlevel=True,
    behavior=None,
):
    """
    Args:
        path (str): Local filename or remote URL, passed to fsspec for resolution.
            May contain glob patterns.
        columns (None, str, or list of str): Glob pattern(s) with bash-like curly
            brackets for matching column names. Nested records are separated by dots.
            If a list of patterns, the logical-or is matched. If None, all columns
            are read.
        row_groups (None or set of int): Row groups to read; must be non-negative.
            Order is ignored: the output array is presented in the order specified by
            Parquet metadata. If None, all row groups/all rows are read.
        storage_options: Passed to `fsspec.parquet.open_parquet_file`.
        max_gap (int): Passed to `fsspec.parquet.open_parquet_file`.
        max_block (int): Passed to `fsspec.parquet.open_parquet_file`.
        footer_sample_size (int): Passed to `fsspec.parquet.open_parquet_file`.
        generate_bitmasks (bool): If enabled and Arrow/Parquet does not have Awkward
            metadata, `generate_bitmasks=True` creates empty bitmasks for nullable
            types that don't have bitmasks in the Arrow/Parquet data, so that the
            Form (BitMaskedForm vs UnmaskedForm) is predictable.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Reads data from a local or remote Parquet file or collection of files.

    The data are eagerly (not lazily) read and must fit into memory. Use `columns`
    and/or `row_groups` to select and filter manageable subsets of the data, and
    use #ak.metadata_from_parquet to find column names and the range of row groups
    that a dataset has.

    See also #ak.to_parquet, #ak.metadata_from_parquet.
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
            generate_bitmasks=generate_bitmasks,
            highlevel=highlevel,
            behavior=behavior,
        ),
    ):
        import awkward._v2._connect.pyarrow  # noqa: F401

        parquet_columns, subform, actual_paths, fs, subrg, meta = _metadata(
            path,
            storage_options,
            row_groups,
            columns,
        )
        return _load(
            actual_paths,
            parquet_columns,
            subrg,
            max_gap,
            max_block,
            footer_sample_size,
            generate_bitmasks,
            subform,
            highlevel,
            behavior,
            fs,
            meta,
        )


def _metadata(path, storage_options, row_groups, columns, ignore_metadata=False):
    # Test cases
    #  - list of data files, scanned
    #  - list of data files, not scanned
    #  - list of directories -> exception
    #  - directory with _metadata, used
    #  - directory with _metadata, not used, files scanned
    #  - directory with _metadata, not used, files not scanned
    #  - directory without _metadata but with _common_metadata
    #  - directory with only data files, scanned
    #  - directory with only data files, not scanned
    import pyarrow.parquet as pyarrow_parquet
    import fsspec.parquet

    if row_groups is not None:
        if not all(ak._v2._util.isint(x) and x >= 0 for x in row_groups):
            raise ak._v2._util.error(
                TypeError("row_groups must be a set of non-negative integers")
            )

    fs, _, paths = fsspec.get_fs_token_paths(
        path, mode="rb", storage_options=storage_options
    )
    all_paths, path_for_metadata = _all_and_metadata_paths(path, fs, paths)

    parquet_columns = None
    subform = None
    subrg = [None] * len(all_paths)
    actual_paths = all_paths
    with fs.open(
        path_for_metadata,
    ) as file_for_metadata:
        parquetfile_for_metadata = pyarrow_parquet.ParquetFile(file_for_metadata)

        if columns is not None:
            list_indicator = "list.item"

            form = ak._v2._connect.pyarrow.form_handle_arrow(
                parquetfile_for_metadata.schema_arrow, pass_empty_field=True
            )
            subform = form.select_columns(columns)
            parquet_columns = subform.columns(list_indicator=list_indicator)

        metadata = parquetfile_for_metadata.metadata
        if row_groups is not None:
            eoln = "\n    "
            if any(not 0 <= rg < metadata.num_row_groups for rg in row_groups):
                raise ak._v2._util.error(
                    ValueError(
                        f"one of the requested row_groups is out of range "
                        f"(must be less than {metadata.num_row_groups})"
                    )
                )

            split_paths = [p.split("/") for p in all_paths]
            prev_index = None
            prev_i = 0
            actual_paths = []
            subrg = []
            for i in range(metadata.num_row_groups):
                unsplit_path = metadata.row_group(i).column(0).file_path
                if unsplit_path == "":
                    if len(all_paths) == 1:
                        index = 0
                    else:
                        raise ak._v2._util.error(
                            LookupError(
                                f"""path from metadata is {unsplit_path!r} but more
                                than one path matches:\n\n{eoln.join(all_paths)}"""
                            )
                        )

                else:
                    split_path = unsplit_path.split("/")
                    index = None
                    for j, compare in enumerate(split_paths):
                        if split_path == compare[-len(split_path) :]:
                            index = j
                            break
                    if index is None:
                        raise ak._v2._util.error(
                            LookupError(
                                f"""path {'/'.join(split_path)!r} from metadata not found
                                in path matches:\n\n{eoln.join(all_paths)}"""
                            )
                        )

                if prev_index != index:
                    prev_index = index
                    prev_i = i
                    actual_paths.append(all_paths[index])
                    subrg.append([])

                if i in row_groups:
                    subrg[-1].append(i - prev_i)

            for k in range(len(subrg) - 1, -1, -1):
                if len(subrg[k]) == 0:
                    del actual_paths[k]
                    del subrg[k]
        if subform is None:
            subform = ak._v2._connect.pyarrow.form_handle_arrow(
                parquetfile_for_metadata.schema_arrow, pass_empty_field=True
            )
    return parquet_columns, subform, actual_paths, fs, subrg, metadata


def _load(
    actual_paths,
    parquet_columns,
    subrg,
    max_gap,
    max_block,
    footer_sample_size,
    generate_bitmasks,
    subform,
    highlevel,
    behavior,
    fs,
    meta,
):
    arrays = []
    for i, p in enumerate(actual_paths):
        arrays.append(
            _read_parquet_file(
                p,
                fs=fs,
                parquet_columns=parquet_columns,
                row_groups=subrg[i],
                max_gap=max_gap,
                max_block=max_block,
                footer_sample_size=footer_sample_size,
                generate_bitmasks=generate_bitmasks,
                metadata=meta,
            )
        )

    if len(arrays) == 0:
        numpy = ak.nplike.Numpy.instance()
        return ak._v2.operations.convert.ak_from_buffers._impl(
            subform, 0, _DictOfEmptyBuffers(), "", numpy, highlevel, behavior
        )
    else:
        return ak._v2.operations.structure.ak_concatenate._impl(
            arrays, 0, True, True, highlevel, behavior
        )


def _read_parquet_file(
    path,
    fs,
    parquet_columns,
    row_groups,
    footer_sample_size,
    max_gap,
    max_block,
    metadata,
    generate_bitmasks,
):
    import fsspec.parquet
    import pyarrow.parquet as pyarrow_parquet

    with fsspec.parquet.open_parquet_file(
        path,
        fs=fs,
        engine="pyarrow",
        columns=parquet_columns,
        row_groups=row_groups,
        max_gap=max_gap,
        max_block=max_block,
        footer_sample_size=footer_sample_size,
    ) as file:
        parquetfile = pyarrow_parquet.ParquetFile(file)

        if row_groups is None:
            arrow_table = parquetfile.read(parquet_columns)
        else:
            arrow_table = parquetfile.read_row_groups(row_groups, parquet_columns)

    # TODO: apply metadata on convert
    return ak._v2._connect.pyarrow.handle_arrow(
        arrow_table,
        generate_bitmasks=generate_bitmasks,
        pass_empty_field=True,
    )


class _DictOfEmptyBuffers:
    def __getitem__(self, where):
        return b"\x00\x00\x00\x00\x00\x00\x00\x00"


def _all_and_metadata_paths(path, fs, paths):
    all_paths = []
    for x in paths:
        if fs.isfile(x):
            is_meta = x.rsplit("/", 1)[-1] == "_metadata"
            is_comm = x.rsplit("/", 1)[-1] == "_common_metadata"
            all_paths.append((x, is_meta, is_comm))
        elif fs.isdir(x):
            for fdata in fs.find(x):
                f = fdata["name"]
                is_meta = f == "_metadata"
                is_comm = f == "_common_metadata"
                if f.endswith((".parq", ".parquet")) or is_meta or is_comm:
                    if fdata["type"] == "file":
                        all_paths.append(f, is_meta, is_comm)

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
