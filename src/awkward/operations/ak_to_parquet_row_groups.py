from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function

__all__ = ("to_parquet_row_groups",)


@high_level_function()
def to_parquet_row_groups(
    iterator,
    destination,
    *,
    list_to32=False,
    string_to32=True,
    bytestring_to32=True,
    emptyarray_to=None,
    categorical_as_dictionary=False,
    extensionarray=True,
    count_nulls=True,
    compression="zstd",
    compression_level=None,
    row_group_size=64 * 1024 * 1024,
    data_page_size=None,
    parquet_flavor=None,
    parquet_version="2.4",
    parquet_page_version="1.0",
    parquet_metadata_statistics=True,
    parquet_dictionary_encoding=False,
    parquet_byte_stream_split=False,
    parquet_coerce_timestamps=None,
    parquet_old_int96_timestamps=None,
    parquet_compliant_nested=False,  # https://issues.apache.org/jira/browse/ARROW-16348
    parquet_extra_options=None,
    storage_options=None,
):
    """
    Args:
        iterator: Generator object that iterates over awkward arrays.
        destination (path-like): Name of the output file, file path, or
            remote URL passed to [fsspec.core.url_to_fs](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.core.url_to_fs)
            for remote writing.
        list_to32 (bool): If True, convert Awkward lists into 32-bit Arrow lists
            if they're small enough, even if it means an extra conversion. Otherwise,
            signed 32-bit #ak.types.ListType maps to Arrow `ListType`,
            signed 64-bit #ak.types.ListType maps to Arrow `LargeListType`,
            and unsigned 32-bit #ak.types.ListType picks whichever Arrow type its
            values fit into.
        string_to32 (bool): Same as the above for Arrow `string` and `large_string`.
        bytestring_to32 (bool): Same as the above for Arrow `binary` and `large_binary`.
        emptyarray_to (None or dtype): If None, #ak.types.UnknownType maps to Arrow's
            null type; otherwise, it is converted a given numeric dtype.
        categorical_as_dictionary (bool): If True, #ak.contents.IndexedArray and
            #ak.contents.IndexedOptionArray labeled with `__array__ = "categorical"`
            are mapped to Arrow `DictionaryArray`; otherwise, the projection is
            evaluated before conversion (always the case without
            `__array__ = "categorical"`).
        extensionarray (bool): If True, this function returns extended Arrow arrays
            (at all levels of nesting), which preserve metadata so that Awkward \u2192
            Arrow \u2192 Awkward preserves the array's #ak.types.Type (though not
            the #ak.forms.Form). If False, this function returns generic Arrow arrays
            that might be needed for third-party tools that don't recognize Arrow's
            extensions. Even with `extensionarray=False`, the values produced by
            Arrow's `to_pylist` method are the same as the values produced by Awkward's
            #ak.to_list.
        count_nulls (bool): If True, count the number of missing values at each level
            and include these in the resulting Arrow array, which makes some downstream
            applications faster. If False, skip the up-front cost of counting them.
        compression (None, str, or dict): Compression algorithm name, passed to
            [pyarrow.parquet.ParquetWriter](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html).
            Parquet supports `{"NONE", "SNAPPY", "GZIP", "BROTLI", "LZ4", "ZSTD"}`
            (where `"GZIP"` is also known as "zlib" or "deflate"). If a dict, the keys
            are column names (the same column names that #ak.forms.Form.columns returns
            and #ak.forms.Form.select_columns accepts) and the values are compression
            algorithm names, to compress each column differently.
        compression_level (None, int, or dict None): Compression level, passed to
            [pyarrow.parquet.ParquetWriter](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html).
            Compression levels have different meanings for different compression
            algorithms: GZIP ranges from 1 to 9, but ZSTD ranges from -7 to 22, for
            example. Generally, higher numbers provide slower but smaller compression.
        row_group_size (int or None): Maximum number of entries in each row group (except the last),
            passed to [pyarrow.parquet.ParquetWriter.write_table](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html#pyarrow.parquet.ParquetWriter.write_table).
            If None, the Parquet default of 64 MiB is used.
        data_page_size (None or int): Number of bytes in each data page, passed to
            [pyarrow.parquet.ParquetWriter](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html).
            If None, the Parquet default of 1 MiB is used.
        parquet_flavor (None or `"spark"`): If None, the output Parquet file will follow
            Arrow conventions; if `"spark"`, it will follow Spark conventions. Some
            systems, such as Spark and Google BigQuery, might need Spark conventions,
            while others might need Arrow conventions. Passed to
            [pyarrow.parquet.ParquetWriter](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html).
            as `flavor`.
        parquet_version (`"1.0"`, `"2.4"`, or `"2.6"`): Parquet file format version.
            Passed to [pyarrow.parquet.ParquetWriter](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html).
            as `version`.
        parquet_page_version (`"1.0"` or `"2.0"`): Parquet page format version.
            Passed to [pyarrow.parquet.ParquetWriter](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html).
            as `data_page_version`.
        parquet_metadata_statistics (bool or dict): If True, include summary
            statistics for each data page in the Parquet metadata, which lets some
            applications search for data more quickly (by skipping pages). If a dict
            mapping column names to bool, include summary statistics on only the
            specified columns. Passed to
            [pyarrow.parquet.ParquetWriter](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html).
            as `write_statistics`.
        parquet_dictionary_encoding (bool or dict): If True, allow Parquet to pre-compress
            with dictionary encoding. If a dict mapping column names to bool, only
            use dictionary encoding on the specified columns. Passed to
            [pyarrow.parquet.ParquetWriter](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html).
            as `use_dictionary`.
        parquet_byte_stream_split (bool or dict): If True, pre-compress floating
            point fields (`float32` or `float64`) with byte stream splitting, which
            collects all mantissas in one part of the stream and exponents in another.
            Passed to [pyarrow.parquet.ParquetWriter](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html).
            as `use_byte_stream_split`.
        parquet_coerce_timestamps (None, `"ms"`, or `"us"`): If None, any timestamps
            (`datetime64` data) are coerced to a given resolution depending on
            `parquet_version`: version `"1.0"` and `"2.4"` are coerced to microseconds,
            but later versions use the `datetime64`'s own units. If `"ms"` is explicitly
            specified, timestamps are coerced to milliseconds; if `"us"`, microseconds.
            Passed to [pyarrow.parquet.ParquetWriter](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html).
            as `coerce_timestamps`.
        parquet_old_int96_timestamps (None or bool): If True, use Parquet's INT96 format
            for any timestamps (`datetime64` data), taking priority over `parquet_coerce_timestamps`.
            If None, let the `parquet_flavor` decide. Passed to
            [pyarrow.parquet.ParquetWriter](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html).
            as `use_deprecated_int96_timestamps`.
        parquet_compliant_nested (bool): If True, use the Spark/BigQuery/Parquet
            [convention for nested lists](https://github.com/apache/parquet-format/blob/master/LogicalTypes.md#nested-types),
            in which each list is a one-field record with field name "`element`";
            otherwise, use the Arrow convention, in which the field name is "`item`".
            Passed to [pyarrow.parquet.ParquetWriter](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html).
            as `use_compliant_nested_type`.
        parquet_extra_options (None or dict): Any additional options to pass to
            [pyarrow.parquet.ParquetWriter](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html).
        storage_options (None or dict): Any additional options to pass to
            [fsspec.core.url_to_fs](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.core.url_to_fs)
            to open a remote file for writing.

    Returns:
    `pyarrow._parquet.FileMetaData` instance

    Writes an iterator over an Awkward Array to a Parquet file in batches (through pyarrow).

        >>> array1 = ak.Array([[1, 2, 3], [], [4, 5], [], [], [6, 7, 8, 9]])
        >>> ak.to_parquet_row_groups((batch for batch in array1), "array1.parquet")
        <pyarrow._parquet.FileMetaData object at 0x7f646c38ff40>
          created_by: parquet-cpp-arrow version 9.0.0
          num_columns: 1
          num_rows: 6
          num_row_groups: 1
          format_version: 2.6
          serialized_size: 0

    If the `array` does not contain records at top-level, the Arrow table will consist
    of one field whose name is `""` iff. `extensionarray` is False.

    If `extensionarray` is True`, use a custom Arrow extension to store this array.
    Otherwise, generic Arrow arrays are used, and if the `array` does not
    contain records at top-level, the Arrow table will consist of one field whose
    name is `""`. See #ak.to_arrow_table for more details.

    Parquet files can maintain the distinction between "option-type but no elements are
    missing" and "not option-type" at all levels, including the top level. However,
    there is no distinction between `?union[X, Y, Z]]` type and `union[?X, ?Y, ?Z]` type.
    Be aware of these type distinctions when passing data through Arrow or Parquet.

    See also #ak.to_arrow, which is used as an intermediate step.
    """
    # Dispatch
    yield (iterator,)

    return ak.ak_to_parquet._impl(
        iterator,
        destination,
        list_to32,
        string_to32,
        bytestring_to32,
        emptyarray_to,
        categorical_as_dictionary,
        extensionarray,
        count_nulls,
        compression,
        compression_level,
        row_group_size,
        data_page_size,
        parquet_flavor,
        parquet_version,
        parquet_page_version,
        parquet_metadata_statistics,
        parquet_dictionary_encoding,
        parquet_byte_stream_split,
        parquet_coerce_timestamps,
        parquet_old_int96_timestamps,
        parquet_compliant_nested,  # https://issues.apache.org/jira/browse/ARROW-16348
        parquet_extra_options,
        storage_options,
        write_iteratively=True,
    )
