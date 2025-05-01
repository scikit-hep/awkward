# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import hashlib
import json

import fsspec.parquet

import awkward as ak
import awkward._connect.pyarrow
from awkward._dispatch import high_level_function
from awkward._layout import wrap_layout
from awkward._regularize import is_integer

__all__ = ("from_parquet",)


@high_level_function()
def from_parquet(
    path,
    *,
    columns=None,
    row_groups=None,
    storage_options=None,
    max_gap=64_000,
    max_block=256_000_000,
    footer_sample_size=1_000_000,
    generate_bitmasks=False,
    highlevel=True,
    behavior=None,
    attrs=None,
):
    """
    Args:
        path (str): Local filename or remote URL, passed to fsspec for resolution.
            May contain glob patterns.
        columns (None, str, or iterable of (str or iterable of str)): Glob pattern(s) including bash-like curly
            brackets for matching column names. Nested records are separated by dots.
            If a list of patterns, the logical-or is matched. If None, all columns
            are read. A list of lists can be provided to select columns with literal dots
            in their names -- The inner list provides column names or patterns.
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
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Reads data from a local or remote Parquet file or collection of files.

    The data are eagerly (not lazily) read and must fit into memory. Use `columns`
    and/or `row_groups` to select and filter manageable subsets of the data, and
    use #ak.metadata_from_parquet to find column names and the range of row groups
    that a dataset has.

    See also #ak.to_parquet, #ak.metadata_from_parquet.
    """

    parquet_columns, subform, actual_paths, fs, subrg, row_counts, meta, uuid = (
        metadata(path, storage_options, row_groups, columns, calculate_uuid=True)
    )
    return _load(
        actual_paths,
        parquet_columns if columns is not None else None,
        subrg,
        max_gap,
        max_block,
        footer_sample_size,
        generate_bitmasks,
        subform,
        highlevel,
        behavior,
        fs,
        attrs,
    )


@high_level_function()
def metadata(
    path,
    storage_options=None,
    row_groups=None,
    columns=None,
    ignore_metadata=False,
    scan_files=True,
    calculate_uuid=False,
):
    # early exit if missing deps
    pyarrow_parquet = awkward._connect.pyarrow.import_pyarrow_parquet("ak.from_parquet")

    if row_groups is not None:
        if not all(is_integer(x) and x >= 0 for x in row_groups):
            raise ValueError("row_groups must be a set of non-negative integers")
        if len(set(row_groups)) < len(row_groups):
            raise ValueError("row group indices must not repeat")

    fs, _, paths = fsspec.get_fs_token_paths(
        path, mode="rb", storage_options=storage_options
    )
    all_paths, path_for_schema, can_sub = _all_and_metadata_paths(
        path, fs, paths, ignore_metadata, scan_files
    )

    subrg = [None] * len(all_paths)
    actual_paths = all_paths
    with fs.open(
        path_for_schema,
    ) as file_for_metadata:
        parquetfile_for_metadata = pyarrow_parquet.ParquetFile(file_for_metadata)

    list_indicator = "list.item"
    for column_metadata in parquetfile_for_metadata.schema:
        if (
            column_metadata.max_repetition_level > 0
            and ".list.element" in column_metadata.path
        ):
            list_indicator = "list.element"
            break

    subform = ak._connect.pyarrow.form_handle_arrow(
        parquetfile_for_metadata.schema_arrow, pass_empty_field=True
    )
    if columns is not None:
        subform = subform.select_columns(columns)

    # Handle empty field at root
    if parquetfile_for_metadata.schema_arrow.names == [""]:
        column_prefix = ("",)
    else:
        column_prefix = ()

    metadata = parquetfile_for_metadata.metadata
    if scan_files and not path_for_schema.endswith("/_metadata"):
        if path_for_schema in all_paths:
            scan_paths = all_paths[1:]
        else:
            scan_paths = all_paths
        for apath in scan_paths:
            with fs.open(apath, "rb") as f:
                md = pyarrow_parquet.ParquetFile(f).metadata
                # TODO: not nested directory structure yet
                md.set_file_path(apath.rsplit("/", 1)[-1])
                metadata.append_row_groups(md)
    if row_groups is not None:
        if any(_ >= metadata.num_row_groups for _ in row_groups):
            raise ValueError(
                f"Row group selection out of bounds 0..{metadata.num_row_groups - 1}"
            )
        if not can_sub:
            raise TypeError(
                "Requested selection of row-groups, but not scanning metadata"
            )

        path_rgs = {}
        rgs_path = {}
        subrg = []
        col_counts = []
        for i in range(metadata.num_row_groups):
            fp = metadata.row_group(i).column(0).file_path
            path_rgs.setdefault(fp, []).append(i)
            rgs_path[i] = fp

        actual_paths = []
        for select in row_groups:
            path = rgs_path[select]
            path2 = next(_ for _ in all_paths if _.endswith(path))
            if path2 not in actual_paths:
                actual_paths.append(path2)
                subrg.append([path_rgs[path].index(select)])
            else:
                subrg[-1].append(path_rgs[path].index(select))
            col_counts.append(metadata.row_group(select).num_rows)
    else:
        if can_sub:
            col_counts = [
                metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)
            ]
        else:
            col_counts = None

    parquet_columns = subform.columns(
        list_indicator=list_indicator, column_prefix=column_prefix
    )

    # generate hash from the col_counts, first row_group and last row_group to calculate approximate parquet uuid
    uuid = None
    if calculate_uuid:
        uuids = [repr({"col_counts": col_counts})]
        for row_group_index in (0, metadata.num_row_groups - 1):
            row_group_info = metadata.row_group(row_group_index).to_dict()
            for k, v in row_group_info.items():
                # sorting columns, and columns::statistics have some version skew in underlying library
                # with latter's 'distinct_counts' showing None vs 0 for example, so they're not used
                if k in ["num_rows", "num_columns"]:
                    uuids.append(repr({k: v}))
                if k == "columns":
                    for subitem in v:
                        for subkey in subitem:
                            if subkey not in [
                                "file_offset",
                                "file_path",
                                "physical_type",
                                "path_in_schema",
                                "compression",
                                "encodings",
                                "total_compressed_size",
                            ]:
                                continue
                            uuids.append(repr({subkey: subitem[subkey]}))
        uuid = hashlib.sha256(json.dumps(",".join(uuids)).encode()).hexdigest()
        return (
            parquet_columns,
            subform,
            actual_paths,
            fs,
            subrg,
            col_counts,
            metadata,
            uuid,
        )

    return parquet_columns, subform, actual_paths, fs, subrg, col_counts, metadata


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
    attrs,
    metadata=None,
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
                metadata=metadata,
            )
        )

    assert len(arrays) != 0
    if len(arrays) == 1:
        return wrap_layout(
            arrays[0], highlevel=highlevel, attrs=attrs, behavior=behavior
        )
    else:
        # TODO: if each array is a record?
        return ak.operations.ak_concatenate._impl(
            arrays,
            axis=0,
            mergebool=True,
            highlevel=highlevel,
            behavior=behavior,
            attrs=attrs,
        )


def _open_file(
    path, fs, columns, row_groups, max_gap, max_block, footer_sample_size, metadata
):
    """Picks between fsspec.parquet and normal fs.open"""
    # condition should be if columns and ow_groups are not all the possible ones
    if (columns or row_groups) and getattr(fs, "async_impl", False):
        return fsspec.parquet.open_parquet_file(
            path,
            fs=fs,
            engine="pyarrow",
            columns=columns,
            row_groups=row_groups,
            max_gap=max_gap,
            metadata=metadata,
            max_block=max_block,
            footer_sample_size=footer_sample_size,
        )
    else:
        return fs.open(path, "rb")


def _read_parquet_file(
    path,
    fs,
    parquet_columns,
    row_groups,
    footer_sample_size,
    max_gap,
    max_block,
    generate_bitmasks,
    metadata=None,
):
    pyarrow_parquet = awkward._connect.pyarrow.import_pyarrow_parquet("ak.from_parquet")

    with _open_file(
        path,
        fs,
        parquet_columns,
        row_groups,
        max_gap,
        max_block,
        footer_sample_size,
        metadata,
    ) as file:
        parquetfile = pyarrow_parquet.ParquetFile(file)

        if row_groups is None:
            arrow_table = parquetfile.read(parquet_columns)
        else:
            arrow_table = parquetfile.read_row_groups(row_groups, parquet_columns)

    arrow_table = ak._connect.pyarrow.convert_native_arrow_table_to_awkward(arrow_table)
    return ak.operations.ak_from_arrow._impl(
        arrow_table,
        generate_bitmasks,
        # why is high-level False here?
        False,
        None,
        None,
    )


class _DictOfEmptyBuffers:
    def __getitem__(self, where):
        return b"\x00\x00\x00\x00\x00\x00\x00\x00"


def _all_and_metadata_paths(path, fs, paths, ignore_metadata=False, scan_files=True):
    all_paths = []
    for x in paths:
        if fs.isfile(x):
            is_meta = x.rsplit("/", 1)[-1] == "_metadata"
            if is_meta and ignore_metadata:
                continue
            is_comm = x.rsplit("/", 1)[-1] == "_common_metadata"
            if is_comm and scan_files:
                continue
            all_paths.append((x, is_meta, is_comm))
        elif fs.isdir(x):
            for f, fdata in fs.find(x, detail=True).items():
                is_meta = f.endswith("_metadata")
                if is_meta and ignore_metadata:
                    continue
                is_comm = f.endswith("_common_metadata")
                if is_comm and scan_files:
                    continue
                if f.endswith((".parq", ".parquet")) or is_meta or is_comm:
                    if fdata["type"] == "file":
                        all_paths.append((f, is_meta, is_comm))

    path_for_metadata = [x for x, is_meta, is_comm in all_paths if is_meta]
    if len(path_for_metadata) != 0:
        path_for_metadata = path_for_metadata[0]
        can_sub = True
    else:
        path_for_metadata = [x for x, is_meta, is_comm in all_paths if is_comm]
        if len(path_for_metadata) != 0:
            path_for_metadata = path_for_metadata[0]
        else:
            if len(all_paths) != 0:
                path_for_metadata = all_paths[0][0]
        # we will still know rew-groups and counts if we scan, so can sub-select
        can_sub = scan_files or len(all_paths) == 1

    all_paths = [x for x, is_meta, is_comm in all_paths if not is_meta and not is_comm]

    if len(all_paths) == 0:
        raise ValueError(f"no *.parquet or *.parq matches for path {path!r}")

    return all_paths, path_for_metadata, can_sub
