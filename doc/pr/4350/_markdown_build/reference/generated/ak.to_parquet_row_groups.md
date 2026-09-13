# ak.to_parquet_row_groups

Defined in [awkward.operations.ak_to_parquet_row_groups](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_to_parquet_row_groups.py) on [line 8](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_to_parquet_row_groups.py#L8).

#### ak.to_parquet_row_groups(iterator, destination, \*, list_to32=False, string_to32=True, bytestring_to32=True, emptyarray_to=None, categorical_as_dictionary=False, extensionarray=True, count_nulls=True, compression='zstd', compression_level=None, row_group_size=None, data_page_size=None, parquet_flavor=None, parquet_version='2.6', parquet_page_version='1.0', parquet_metadata_statistics=True, parquet_dictionary_encoding=False, parquet_byte_stream_split=False, parquet_coerce_timestamps=None, parquet_old_int96_timestamps=None, parquet_compliant_nested=False, parquet_extra_options=None, storage_options=None)

Writes a sequence of Awkward Arrays to a Parquet file as row groups.

As in [`ak.to_parquet`](sphinx-llm:34bdf9ca41984994beeef0eed40a786b#ak.to_parquet), the arrays’ [`ak.Array.attrs`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array.attrs) are written into the file’s
metadata. The file-level `attrs` are defined by the first array, just as the
schema is, and the `attrs` of subsequent arrays are not merged into them. If
you need the `attrs` of every array to be combined, do so explicitly before
passing the iterator to this function.

If the `array` does not contain records at top-level, the Arrow table will consist
of one field whose name is `""` iff. `extensionarray` is False.

If `extensionarray` is True\`\`, use a custom Arrow extension to store this array.
Otherwise, generic Arrow arrays are used, and if the `array` does not
contain records at top-level, the Arrow table will consist of one field whose
name is `""`. See [`ak.to_arrow_table`](sphinx-llm:b370f32af9534c74b92e6e448b182ab2#ak.to_arrow_table) for more details.

Parquet files can maintain the distinction between “option-type but no elements are
missing” and “not option-type” at all levels, including the top level. However,
there is no distinction between `?union[X, Y, Z]]` type and `union[?X, ?Y, ?Z]` type.
Be aware of these type distinctions when passing data through Arrow or Parquet.

See also [`ak.to_arrow`](sphinx-llm:b7375fef0bca4ff1837b125c50ce5ad8#ak.to_arrow), which is used as an intermediate step.

* **Parameters:**
  * **iterator** – Generator object that iterates over awkward arrays.
  * **destination** (*path-like*) – Name of the output file, file path, or
    remote URL passed to [fsspec.core.url_to_fs](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.core.url_to_fs)
    for remote writing.
  * **list_to32** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, convert Awkward lists into 32-bit Arrow lists
    if they’re small enough, even if it means an extra conversion. Otherwise,
    signed 32-bit [`ak.types.ListType`](sphinx-llm:649f7f37622a4fbc999e70526300cf12#ak.types.ListType) maps to Arrow `ListType`,
    signed 64-bit [`ak.types.ListType`](sphinx-llm:649f7f37622a4fbc999e70526300cf12#ak.types.ListType) maps to Arrow `LargeListType`,
    and unsigned 32-bit [`ak.types.ListType`](sphinx-llm:649f7f37622a4fbc999e70526300cf12#ak.types.ListType) picks whichever Arrow type its
    values fit into.
  * **string_to32** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – Same as the above for Arrow `string` and `large_string`.
  * **bytestring_to32** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – Same as the above for Arrow `binary` and `large_binary`.
  * **emptyarray_to** (*None* *or* *dtype*) – If None, [`ak.types.UnknownType`](sphinx-llm:4c8a8f1814d7428e80e2d1e3a84b2dde#ak.types.UnknownType) maps to Arrow’s
    null type; otherwise, it is converted a given numeric dtype.
  * **categorical_as_dictionary** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, [`ak.contents.IndexedArray`](sphinx-llm:0ec798a3207a4746a193b3e535c37f41#ak.contents.IndexedArray) and
    [`ak.contents.IndexedOptionArray`](sphinx-llm:3f1e6f3c5e154ca39fcd977247e4a221#ak.contents.IndexedOptionArray) labeled with `__array__ = "categorical"`
    are mapped to Arrow `DictionaryArray`; otherwise, the projection is
    evaluated before conversion (always the case without
    `__array__ = "categorical"`).
  * **extensionarray** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, this function returns extended Arrow arrays
    (at all levels of nesting), which preserve metadata so that Awkward →
    Arrow → Awkward preserves the array’s [`ak.types.Type`](sphinx-llm:fc995ece0ffa4f68adbe74567c7b2ea1#ak.types.Type) (though not
    the [`ak.forms.Form`](sphinx-llm:6295086f75dd4713a4cbcfec3594d1a1#ak.forms.Form)). If False, this function returns generic Arrow arrays
    that might be needed for third-party tools that don’t recognize Arrow’s
    extensions. Even with `extensionarray=False`, the values produced by
    Arrow’s `to_pylist` method are the same as the values produced by Awkward’s
    [`ak.to_list`](sphinx-llm:c5d0843c37c34cb88bbec16becc6a6f0#ak.to_list).
  * **count_nulls** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, count the number of missing values at each level
    and include these in the resulting Arrow array, which makes some downstream
    applications faster. If False, skip the up-front cost of counting them.
  * **compression** (*None* *,* [*str*](https://docs.python.org/3/library/stdtypes.html#str) *, or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Compression algorithm name, passed to
    [pyarrow.parquet.ParquetWriter](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html).
    Parquet supports `{"NONE", "SNAPPY", "GZIP", "BROTLI", "LZ4", "ZSTD"}`
    (where `"GZIP"` is also known as “zlib” or “deflate”). If a dict, the keys
    are column names (the same column names that [`ak.forms.Form.columns`](sphinx-llm:6295086f75dd4713a4cbcfec3594d1a1#ak.forms.Form.columns) returns
    and [`ak.forms.Form.select_columns`](sphinx-llm:6295086f75dd4713a4cbcfec3594d1a1#ak.forms.Form.select_columns) accepts) and the values are compression
    algorithm names, to compress each column differently.
  * **compression_level** (*None* *,* [*int*](https://docs.python.org/3/library/functions.html#int) *, or* *dict None*) – Compression level, passed to
    [pyarrow.parquet.ParquetWriter](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html).
    Compression levels have different meanings for different compression
    algorithms: GZIP ranges from 1 to 9, but ZSTD ranges from -7 to 22, for
    example. Generally, higher numbers provide slower but smaller compression.
  * **row_group_size** ([*int*](https://docs.python.org/3/library/functions.html#int) *,* [*str*](https://docs.python.org/3/library/stdtypes.html#str) *, or* *None*) – If an integer, the maximum number of
    rows in each row group; if a string, the maximum memory size of each
    row group. The string must be a number followed by a memory unit, such
    as `"100 MB"`, and is converted into the number of rows that fit into
    that many bytes based on the in-memory size of each batch. Passed to
    [pyarrow.parquet.ParquetWriter.write_table](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html#pyarrow.parquet.ParquetWriter.write_table).
    If None, PyArrow’s default of at most 1024 \* 1024 rows is used.
  * **data_page_size** (*None* *or* [*int*](https://docs.python.org/3/library/functions.html#int)) – Number of bytes in each data page, passed to
    [pyarrow.parquet.ParquetWriter](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html).
    If None, the Parquet default of 1 MiB is used.
  * **parquet_flavor** (None or `"spark"`) – If None, the output Parquet file will follow
    Arrow conventions; if `"spark"`, it will follow Spark conventions. Some
    systems, such as Spark and Google BigQuery, might need Spark conventions,
    while others might need Arrow conventions. Passed to
    [pyarrow.parquet.ParquetWriter](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html).
    as `flavor`.
  * **parquet_version** (`"1.0"`, `"2.4"`, or `"2.6"`) – Parquet file format version.
    Passed to [pyarrow.parquet.ParquetWriter](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html).
    as `version`.
  * **parquet_page_version** (`"1.0"` or `"2.0"`) – Parquet page format version.
    Passed to [pyarrow.parquet.ParquetWriter](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html).
    as `data_page_version`.
  * **parquet_metadata_statistics** ([*bool*](https://docs.python.org/3/library/functions.html#bool) *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – If True, include summary
    statistics for each data page in the Parquet metadata, which lets some
    applications search for data more quickly (by skipping pages). If a dict
    mapping column names to bool, include summary statistics on only the
    specified columns. Passed to
    [pyarrow.parquet.ParquetWriter](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html).
    as `write_statistics`.
  * **parquet_dictionary_encoding** ([*bool*](https://docs.python.org/3/library/functions.html#bool) *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – If True, allow Parquet to pre-compress
    with dictionary encoding. If a dict mapping column names to bool, only
    use dictionary encoding on the specified columns. Passed to
    [pyarrow.parquet.ParquetWriter](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html).
    as `use_dictionary`.
  * **parquet_byte_stream_split** ([*bool*](https://docs.python.org/3/library/functions.html#bool) *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – If True, pre-compress floating
    point fields (`float32` or `float64`) with byte stream splitting, which
    collects all mantissas in one part of the stream and exponents in another.
    Passed to [pyarrow.parquet.ParquetWriter](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html).
    as `use_byte_stream_split`.
  * **parquet_coerce_timestamps** (None, `"ms"`, or `"us"`) – If None, any timestamps
    (`datetime64` data) are coerced to a given resolution depending on
    `parquet_version`: version `"1.0"` and `"2.4"` are coerced to microseconds,
    but later versions use the `datetime64`’s own units. If `"ms"` is explicitly
    specified, timestamps are coerced to milliseconds; if `"us"`, microseconds.
    Passed to [pyarrow.parquet.ParquetWriter](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html).
    as `coerce_timestamps`.
  * **parquet_old_int96_timestamps** (*None* *or* [*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, use Parquet’s INT96 format
    for any timestamps (`datetime64` data), taking priority over `parquet_coerce_timestamps`.
    If None, let the `parquet_flavor` decide. Passed to
    [pyarrow.parquet.ParquetWriter](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html).
    as `use_deprecated_int96_timestamps`.
  * **parquet_compliant_nested** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, use the Spark/BigQuery/Parquet
    [convention for nested lists](https://github.com/apache/parquet-format/blob/master/LogicalTypes.md#nested-types),
    in which each list is a one-field record with field name “`element`”;
    otherwise, use the Arrow convention, in which the field name is “`item`”.
    Passed to [pyarrow.parquet.ParquetWriter](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html).
    as `use_compliant_nested_type`.
  * **parquet_extra_options** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Any additional options to pass to
    [pyarrow.parquet.ParquetWriter](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html).
  * **storage_options** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Any additional options to pass to
    [fsspec.core.url_to_fs](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.core.url_to_fs)
    to open a remote file for writing.
* **Returns:**
  A `pyarrow._parquet.FileMetaData` describing the written Parquet file.

### Examples

```pycon
>>> array1 = ak.Array([[1, 2, 3], [], [4, 5], [], [], [6, 7, 8, 9]])
>>> ak.to_parquet_row_groups((batch for batch in array1), "array1.parquet")
<pyarrow._parquet.FileMetaData object at 0x7f646c38ff40>
  created_by: parquet-cpp-arrow version 9.0.0
  num_columns: 1
  num_rows: 6
  num_row_groups: 1
  format_version: 2.6
  serialized_size: 0
```
