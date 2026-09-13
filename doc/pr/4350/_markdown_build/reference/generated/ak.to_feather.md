# ak.to_feather

Defined in [awkward.operations.ak_to_feather](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_to_feather.py) on [line 16](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_to_feather.py#L16).

#### ak.to_feather(array, destination, \*, list_to32=False, string_to32=True, bytestring_to32=True, emptyarray_to=None, categorical_as_dictionary=False, extensionarray=True, count_nulls=True, compression='zstd', compression_level=None, chunksize=None, feather_version=2)

Writes an Awkward Array to a Feather file (through pyarrow).

If the `array` does not contain records at top-level, the Arrow table will
consist of one field whose name is `""` iff. `extensionarray` is False.

If `extensionarray` is True\`\`, use a custom Arrow extension to store this array.
Otherwise, generic Arrow arrays are used, and if the `array` does not
contain records at top-level, the Arrow table will consist of one field whose
name is `""`. See [`ak.to_arrow_table`](sphinx-llm:b370f32af9534c74b92e6e448b182ab2#ak.to_arrow_table) for more details.

See also [`ak.from_feather`](sphinx-llm:9b2693e925dd4c9f89261c061eae4168#ak.from_feather).

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **destination** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Local destination path, passed to
    [pyarrow.feather.write_feather](https://arrow.apache.org/docs/python/generated/pyarrow.feather.write_feather.html#pyarrow.feather.write_feather).
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
  * **compression** (*None* *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Can be one of {“zstd”, “lz4”, “uncompressed”}. The
    default of None uses LZ4 for `feather_version=2` files if it is available, otherwise
    uncompressed. Passed to [pyarrow.feather.write_feather](https://arrow.apache.org/docs/python/generated/pyarrow.feather.write_feather.html#pyarrow.feather.write_feather).
  * **compression_level** (*None* *or* [*int*](https://docs.python.org/3/library/functions.html#int)) – Use a compression level particular to the chosen
    compressor. If None use the default compression level. Passed to [pyarrow.feather.write_feather](https://arrow.apache.org/docs/python/generated/pyarrow.feather.write_feather.html#pyarrow.feather.write_feather).
  * **chunksize** (*None* *or* [*int*](https://docs.python.org/3/library/functions.html#int)) – For `feather_version=2` files, this is the internal maximum size of Arrow RecordBatch
    chunks when writing the Arrow IPC file format. None means use the
    default, which is currently 64K. Passed to [pyarrow.feather.write_feather](https://arrow.apache.org/docs/python/generated/pyarrow.feather.write_feather.html#pyarrow.feather.write_feather).
  * **feather_version** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Feather file version, passed to [pyarrow.feather.write_feather](https://arrow.apache.org/docs/python/generated/pyarrow.feather.write_feather.html#pyarrow.feather.write_feather).
    Version 2 is the current. Version 1 is the more limited legacy format. If not
    provided, version 2 is used.
* **Returns:**
  None. The contents of `array` are written to the given Feather file
  (through pyarrow).

### Examples

```pycon
>>> array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
>>> ak.to_feather(array, "filename.feather")
```
