# ak.metadata_from_parquet

Defined in [awkward.operations.ak_metadata_from_parquet](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_metadata_from_parquet.py) on [line 21](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_metadata_from_parquet.py#L21).

#### ak.metadata_from_parquet(path, \*, storage_options=None, row_groups=None, ignore_metadata=False, scan_files=True)

Reads metadata from a Parquet file or dataset without reading the data.

This function differs from ak.from_parquet._metadata as follows:

* this function will always use a \_metadata file, if present
* if there is no \_metadata, the schema comes from \_common_metadata or
  the first data file
* the total number of rows is always known

A dict containing

* `form`: an Awkward Form representing the low-level type of the data
  (use `.type` to get a high-level type),
* `fs`: the fsspec filesystem object,
* `paths`: a list of matching path names,
* `col_counts`: the number of rows in each row group,
* `columns`: the columns defined by the schema,
* `num_rows`: the length of the array that would be read by [`ak.from_parquet`](sphinx-llm:ab272f6edef4436f8edba845c0477c5f#ak.from_parquet),
* `num_row_groups`: the units that can be filtered (for the [`ak.from_parquet`](sphinx-llm:ab272f6edef4436f8edba845c0477c5f#ak.from_parquet) `row_groups`
  argument).

See also [`ak.from_parquet`](sphinx-llm:ab272f6edef4436f8edba845c0477c5f#ak.from_parquet), [`ak.to_parquet`](sphinx-llm:34bdf9ca41984994beeef0eed40a786b#ak.to_parquet).

* **Parameters:**
  * **path** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Local filename or remote URL, passed to fsspec for resolution.
    May contain glob patterns. A list of paths is also allowed, but they
    must be data files, not directories.
  * **storage_options** – Passed to `fsspec.parquet.open_parquet_file`.
  * **row_groups** (*None* *or* [*set*](https://docs.python.org/3/library/stdtypes.html#set) *of* [*int*](https://docs.python.org/3/library/functions.html#int)) – Row groups to read; must be non-negative.
    Order is ignored: the output array is presented in the order specified
    by Parquet metadata. If None, all row groups/all rows are read.
  * **ignore_metadata** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – ignore the dedicated \_metadata file if found
    and instead derive metadata from the first data file.
  * **scan_files** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – TODO
* **Returns:**
  A dict of metadata describing the Parquet dataset (its form, filesystem,
  paths, row counts, and columns), read without reading the array data.
