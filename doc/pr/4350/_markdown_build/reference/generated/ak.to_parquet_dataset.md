# ak.to_parquet_dataset

Defined in [awkward.operations.ak_to_parquet_dataset](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_to_parquet_dataset.py) on [line 9](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_to_parquet_dataset.py#L9).

#### ak.to_parquet_dataset(directory, filenames=None, storage_options=None)

Creates a `_common_metadata` and a `_metadata` in a directory of Parquet files.

The `_common_metadata` contains the schema that all files share. (If the files
have different schemas, this function raises an exception.)

The `_metadata` contains row-group metadata used to seek to specific row-groups
within the multi-file dataset.

* **Parameters:**
  * **directory** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *or* *Path*) – A directory in which to write `_common_metadata`
    and `_metadata`, making the directory of Parquet files into a dataset.
  * **filenames** (*None* *or* [*list*](https://docs.python.org/3/library/stdtypes.html#list) *of* [*str*](https://docs.python.org/3/library/stdtypes.html#str) *or* *Path*) – If None, the `directory` will be
    recursively searched for files ending in `filename_extension` and
    sorted lexicographically. Otherwise, this explicit list of files is
    taken and row-groups are concatenated in its given order. If any
    filenames are relative, they are interpreted relative to `directory`.
  * **filename_extension** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Filename extension (including `.`) to use to
    search for files recursively. Ignored if `filenames` is None.
* **Returns:**
  A 2-tuple `(common_metadata_path, metadata_path)` with the paths to the
  `_common_metadata` and `_metadata` files created in the given directory
  of Parquet files.

### Examples

```pycon
>>> ak.to_parquet(array1, "/directory/arr1.parquet", parquet_compliant_nested=True)
>>> ak.to_parquet(array2, "/directory/arr2.parquet", parquet_compliant_nested=True)
>>> ak.to_parquet_dataset("/directory")
```
