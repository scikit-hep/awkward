# ak.to_arrow

Defined in [awkward.operations.ak_to_arrow](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_to_arrow.py) on [line 14](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_to_arrow.py#L14).

#### ak.to_arrow(array, \*, list_to32=False, string_to32=False, bytestring_to32=False, emptyarray_to=None, categorical_as_dictionary=False, extensionarray=True, count_nulls=True)

Converts an Awkward Array into an Apache Arrow array.

This produces arrays of type `pyarrow.Array`. You might need to further
manipulations (using the pyarrow library) to build a `pyarrow.ChunkedArray`,
a `pyarrow.RecordBatch`, or a `pyarrow.Table`. For the latter, see [`ak.to_arrow_table`](sphinx-llm:b370f32af9534c74b92e6e448b182ab2#ak.to_arrow_table).

This function always preserves the values of a dataset; i.e. the Python objects
returned by [`ak.to_list`](sphinx-llm:c5d0843c37c34cb88bbec16becc6a6f0#ak.to_list) are identical to the Python objects returned by Arrow’s
`to_pylist` method. With `extensionarray=True`, this function also preserves the
data type (high-level [`ak.types.Type`](sphinx-llm:fc995ece0ffa4f68adbe74567c7b2ea1#ak.types.Type), though not the low-level [`ak.forms.Form`](sphinx-llm:6295086f75dd4713a4cbcfec3594d1a1#ak.forms.Form)),
even through Parquet, making Parquet a good way to save Awkward Arrays for later
use. If any third-party tools don’t recognize Arrow’s extension arrays, set this
option to False for plain Arrow arrays.

See also [`ak.from_arrow`](sphinx-llm:a01617c5c7474de09fdc0ddb61b2cced#ak.from_arrow), [`ak.to_arrow_table`](sphinx-llm:b370f32af9534c74b92e6e448b182ab2#ak.to_arrow_table), [`ak.to_parquet`](sphinx-llm:34bdf9ca41984994beeef0eed40a786b#ak.to_parquet), [`ak.from_arrow_schema`](sphinx-llm:fdc866ce01444930a88d2e3ed7768d73#ak.from_arrow_schema).

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
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
* **Returns:**
  An Apache Arrow array (`pyarrow.Array`) with the same data as `array`.
