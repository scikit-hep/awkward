# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from collections.abc import Iterable, Sized

import awkward as ak


def to_parquet(
    data,
    destination,
    list_to32=False,
    string_to32=True,
    bytestring_to32=True,
    emptyarray_to=None,
    categorical_as_dictionary=False,
    extensionarray=True,
    count_nulls=True,
    compression="zstd",
    compression_level=None,
    compression_categorical=None,
    compression_floating=None,
    row_group_size=64 * 1024 * 1024,
    data_page_size=None,
    parquet_flavor=None,
    parquet_version="1.0",
    parquet_page_version="1.0",
    parquet_metadata_statistics=True,
    parquet_coerce_timestamps=None,
    parquet_old_int96_timestamps=None,
    parquet_compliant_nested=True,
    parquet_extra_options=None,
    hook_after_write=None,
):
    import awkward._v2._connect.pyarrow  # noqa: F401

    import pyarrow.parquet as pyarrow_parquet
    import fsspec

    if isinstance(data, Iterable) and not isinstance(data, Sized):
        iterator = iter(data)
    elif isinstance(data, Iterable):
        iterator = iter([data])
    else:
        pass  # raise TypeError

    row_group = 0
    array = next(iterator)
    layout = ak._v2.operations.convert.ak_to_layout(
        array, allow_record=False, allow_other=False
    )
    table = ak._v2.operations.convert.ak_to_arrow_table._impl(
        layout,
        list_to32,
        string_to32,
        bytestring_to32,
        emptyarray_to,
        categorical_as_dictionary,
        extensionarray,
        count_nulls,
    )

    if len(layout.fields) != 0:
        form = layout.form
        for column in form.columns():
            column_types = form.column_types(column)
            assert len(column_types) == 1
            # column_type = column_types[0]

    # HERE

    if parquet_extra_options is None:
        parquet_extra_options = {}

    with fsspec.open(destination, "wb") as file:
        with pyarrow_parquet.ParquetWriter(
            file,
            table.schema,
            filesystem=file.fs,
            flavor=parquet_flavor,
            version=parquet_version,
            use_dictionary=compression_categorical,
            compression=compression,
            write_statistics=parquet_metadata_statistics,
            use_deprecated_int96_timestamps=parquet_old_int96_timestamps,
            compression_level=compression_level,
            use_byte_stream_split=compression_floating,
            data_page_version=parquet_page_version,
            use_compliant_nested_type=parquet_compliant_nested,
            data_page_size=data_page_size,
            coerce_timestamps=parquet_coerce_timestamps,
            **parquet_extra_options,
        ) as writer:
            while True:
                writer.write_table(table, row_group_size=row_group_size)
                if hook_after_write is not None:
                    hook_after_write(
                        row_group=row_group,
                        array=array,
                        layout=layout,
                        table=table,
                        writer=writer,
                    )

                row_group += 1
                try:
                    array = next(iterator)
                except StopIteration:
                    break
                layout = ak._v2.operations.convert.ak_to_layout(
                    array, allow_record=False, allow_other=False
                )
                table = ak._v2.operations.convert.ak_to_arrow_table._impl(
                    layout,
                    list_to32,
                    string_to32,
                    bytestring_to32,
                    emptyarray_to,
                    categorical_as_dictionary,
                    extensionarray,
                    count_nulls,
                )
