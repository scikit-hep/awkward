# ak.from_iter

Defined in [awkward.operations.ak_from_iter](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_from_iter.py) on [line 18](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_from_iter.py#L18).

#### ak.from_iter(iterable, \*, allow_record=True, highlevel=True, behavior=None, attrs=None, initial=1024, resize=8)

Converts Python data into an Awkward Array.

Any heterogeneous and deeply nested Python data can be converted, but the
output will never have regular-typed array lengths. Internally, this
function uses `ak::ArrayBuilder` (see the high-level [`ak.ArrayBuilder`](sphinx-llm:2fb9ffd5c3a441658693b9cc500d5518#ak.ArrayBuilder)
documentation for a more complete description).

The following Python types are supported.

* bool, including `np.bool_`: converted into [`ak.contents.NumpyArray`](sphinx-llm:6f8cd60f60b04ce7a9499f2967197e79#ak.contents.NumpyArray).
* int, including `np.integer`: converted into [`ak.contents.NumpyArray`](sphinx-llm:6f8cd60f60b04ce7a9499f2967197e79#ak.contents.NumpyArray).
* float, including `np.floating`: converted into [`ak.contents.NumpyArray`](sphinx-llm:6f8cd60f60b04ce7a9499f2967197e79#ak.contents.NumpyArray).
* bytes: converted into [`ak.contents.ListOffsetArray`](sphinx-llm:00659f65177e4a1db8f94cec2f25984d#ak.contents.ListOffsetArray) with parameter
  `"__array__"` equal to `"bytestring"` (unencoded bytes).
* str: converted into [`ak.contents.ListOffsetArray`](sphinx-llm:00659f65177e4a1db8f94cec2f25984d#ak.contents.ListOffsetArray) with parameter
  `"__array__"` equal to `"string"` (UTF-8 encoded string).
* tuple: converted into [`ak.contents.RecordArray`](sphinx-llm:2475335b802d4bf79c938eaf015c0299#ak.contents.RecordArray) without field names
  (i.e. homogeneously typed, uniform sized tuples).
* dict: converted into [`ak.contents.RecordArray`](sphinx-llm:2475335b802d4bf79c938eaf015c0299#ak.contents.RecordArray) with field names
  (i.e. homogeneously typed records with the same sets of fields).
* iterable, including np.ndarray: converted into
  [`ak.contents.ListOffsetArray`](sphinx-llm:00659f65177e4a1db8f94cec2f25984d#ak.contents.ListOffsetArray).

See also [`ak.to_list`](sphinx-llm:c5d0843c37c34cb88bbec16becc6a6f0#ak.to_list).

* **Parameters:**
  * **iterable** (*Python iterable*) – Data to convert into an Awkward Array.
  * **allow_record** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, the outermost element may be a record
    (returning [`ak.Record`](sphinx-llm:e158df99f80043309c758f6e933352b2#ak.Record) or [`ak.record.Record`](sphinx-llm:c769a66df3124cdc8442e55ad70f6980#ak.record.Record) type, depending on
    `highlevel`); if False, the outermost element must be an array.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
  * **initial** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Initial size (in bytes) of buffers used by the `ak::ArrayBuilder`.
  * **resize** ([*float*](https://docs.python.org/3/library/functions.html#float)) – Resize multiplier for buffers used by the `ak::ArrayBuilder`;
    should be strictly greater than 1.
* **Returns:**
  An [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) built from the given Python data.
