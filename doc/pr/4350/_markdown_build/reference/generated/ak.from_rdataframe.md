# ak.from_rdataframe

Defined in [awkward.operations.ak_from_rdataframe](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_from_rdataframe.py) on [line 14](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_from_rdataframe.py#L14).

#### ak.from_rdataframe(rdf, columns, \*, keep_order=False, offsets_type='int64', with_name=None, highlevel=True, behavior=None, attrs=None)

Converts ROOT RDataFrame columns into an Awkward Array.

The data is copied: the conversion runs the RDataFrame event loop and
writes the selected columns into newly allocated buffers.

If `columns` is a string, the return value represents a single RDataFrame
column. If `columns` is any other iterable, the return value is a record
array, in which each field corresponds to an RDataFrame column. In
particular, if the `columns` iterable contains only one string, it is still
a record array, which has only one field.

See also [`ak.to_rdataframe`](sphinx-llm:130d58a880d04dc49249c7acc0602655#ak.to_rdataframe).

* **Parameters:**
  * **rdf** (`ROOT.RDataFrame`) – ROOT RDataFrame to convert into an
    Awkward Array.
  * **columns** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *or* *iterable* *of* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – A column or multiple columns to be
    converted to Awkward Array.
  * **keep_order** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If set to `True` the columns with Awkward type will
    keep order after filtering.
  * **offsets_type** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – A `NumpyType.primitive` type of the ListOffsetArray
    offsets: `"int32"`, `"uint32"` or `"int64"`.
  * **with_name** (*None* *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Gives tuples and records a name that can be
    used to override their behavior (see [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array)).
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) built from the given ROOT RDataFrame columns.
