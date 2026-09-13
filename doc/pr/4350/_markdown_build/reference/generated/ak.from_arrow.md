# ak.from_arrow

Defined in [awkward.operations.ak_from_arrow](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_from_arrow.py) on [line 15](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_from_arrow.py#L15).

#### ak.from_arrow(array, \*, generate_bitmasks=False, highlevel=True, behavior=None, attrs=None)

Converts an Apache Arrow array into an Awkward Array.

This function always preserves the values of a dataset; i.e. the Python
objects returned by [`ak.to_list`](sphinx-llm:c5d0843c37c34cb88bbec16becc6a6f0#ak.to_list) are identical to the Python objects
returned by Arrow’s `to_pylist` method. If [`ak.to_arrow`](sphinx-llm:b7375fef0bca4ff1837b125c50ce5ad8#ak.to_arrow) was invoked with
`extensionarray=True`, this function also preserves the data type
(high-level [`ak.types.Type`](sphinx-llm:fc995ece0ffa4f68adbe74567c7b2ea1#ak.types.Type), though not the low-level [`ak.forms.Form`](sphinx-llm:6295086f75dd4713a4cbcfec3594d1a1#ak.forms.Form)), even
through Parquet, making Parquet a good way to save Awkward Arrays for later
use.

Because awkward uses numpy’s dtype system, timestamp types do not have
timezones. If encountering timestamp types with timezones in the input
arrow data, they will be silently dropped.

See also [`ak.to_arrow`](sphinx-llm:b7375fef0bca4ff1837b125c50ce5ad8#ak.to_arrow), [`ak.to_arrow_table`](sphinx-llm:b370f32af9534c74b92e6e448b182ab2#ak.to_arrow_table), [`ak.from_parquet`](sphinx-llm:ab272f6edef4436f8edba845c0477c5f#ak.from_parquet), [`ak.from_arrow_schema`](sphinx-llm:fdc866ce01444930a88d2e3ed7768d73#ak.from_arrow_schema).

* **Parameters:**
  * **array** (`pyarrow.Array`, `pyarrow.ChunkedArray`, `pyarrow.RecordBatch`, or `pyarrow.Table`) – Apache Arrow array to convert into an  Awkward Array.
  * **generate_bitmasks** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If enabled and Arrow/Parquet does not have Awkward
    metadata, `generate_bitmasks=True` creates empty bitmasks for nullable
    types that don’t have bitmasks in the Arrow/Parquet data, so that the
    Form (BitMaskedForm vs UnmaskedForm) is predictable.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) built from the given Apache Arrow array.
