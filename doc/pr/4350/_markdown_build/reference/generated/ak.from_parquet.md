# ak.from_parquet

Defined in [awkward.operations.ak_from_parquet](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_from_parquet.py) on [line 19](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_from_parquet.py#L19).

#### ak.from_parquet(path, \*, columns=None, row_groups=None, storage_options=None, max_gap=64000, max_block=256000000, footer_sample_size=1000000, generate_bitmasks=False, highlevel=True, behavior=None, attrs=None)

Reads data from a local or remote Parquet file or collection of files.

The data are eagerly (not lazily) read and must fit into memory. Use
`columns` and/or `row_groups` to select and filter manageable subsets of
the data, and use [`ak.metadata_from_parquet`](sphinx-llm:75fec6bc51c44ab08fa166190078ad9c#ak.metadata_from_parquet) to find column names and the
range of row groups that a dataset has.

Any attrs that [`ak.to_parquet`](sphinx-llm:34bdf9ca41984994beeef0eed40a786b#ak.to_parquet) stored in the file are restored (as are those
written by pandas), unless overridden by the `attrs` argument.

See also [`ak.to_parquet`](sphinx-llm:34bdf9ca41984994beeef0eed40a786b#ak.to_parquet), [`ak.metadata_from_parquet`](sphinx-llm:75fec6bc51c44ab08fa166190078ad9c#ak.metadata_from_parquet).

* **Parameters:**
  * **path** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Local filename or remote URL, passed to fsspec for resolution.
    May contain glob patterns.
  * **columns** (*None* *,* [*str*](https://docs.python.org/3/library/stdtypes.html#str) *, or* *iterable* *of*  *(*[*str*](https://docs.python.org/3/library/stdtypes.html#str) *or* *iterable* *of* [*str*](https://docs.python.org/3/library/stdtypes.html#str) *)*) – Glob pattern(s) including bash-like curly
    brackets for matching column names. Nested records are separated by dots.
    If a list of patterns, the logical-or is matched. If None, all columns
    are read. A list of lists can be provided to select columns with literal dots
    in their names – The inner list provides column names or patterns.
  * **row_groups** (*None* *or* [*set*](https://docs.python.org/3/library/stdtypes.html#set) *of* [*int*](https://docs.python.org/3/library/functions.html#int)) – Row groups to read; must be non-negative.
    Order is ignored: the output array is presented in the order specified by
    Parquet metadata. If None, all row groups/all rows are read.
  * **storage_options** – Passed to `fsspec.parquet.open_parquet_file`.
  * **max_gap** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Passed to `fsspec.parquet.open_parquet_file`.
  * **max_block** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Passed to `fsspec.parquet.open_parquet_file`.
  * **footer_sample_size** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Passed to `fsspec.parquet.open_parquet_file`.
  * **generate_bitmasks** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If enabled and Arrow/Parquet does not have Awkward
    metadata, `generate_bitmasks=True` creates empty bitmasks for nullable
    types that don’t have bitmasks in the Arrow/Parquet data, so that the
    Form (BitMaskedForm vs UnmaskedForm) is predictable.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level. These take precedence over any `attrs` stored in the file.
* **Returns:**
  An [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) read from the given local or remote Parquet file(s).
