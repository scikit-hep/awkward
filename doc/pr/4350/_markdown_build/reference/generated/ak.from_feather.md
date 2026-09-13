# ak.from_feather

Defined in [awkward.operations.ak_from_feather](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_from_feather.py) on [line 13](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_from_feather.py#L13).

#### ak.from_feather(path, \*, columns=None, use_threads=True, memory_map=False, generate_bitmasks=False, highlevel=True, behavior=None, attrs=None)

Reads a Feather file as an Awkward Array (through pyarrow).

See also [`ak.to_feather`](sphinx-llm:480d09861fdd4ca0ab340700b239a669#ak.to_feather).

* **Parameters:**
  * **path** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *or* *file-like object*) – Feather file to read as an Awkward Array,
    passed directly to [pyarrow.feather.read_table](https://arrow.apache.org/docs/python/generated/pyarrow.feather.read_table.html).
  * **columns** (*sequence*) – Only read a specific set of columns. If not provided,
    all columns are read.
  * **use_threads** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, parallelize reading using multiple threads.
  * **memory_map** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, use memory mapping when opening file on disk,
    when source is a string.
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
  An [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) read from the given Feather file (through pyarrow).

### Examples

```pycon
>>> ak.from_feather("file_name.feather")
<Array [[1.1, 2.2, 3.3], [], [4.4, 5.5]] type='3 * var * float64'>
```
