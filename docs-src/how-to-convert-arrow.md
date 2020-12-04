---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.10'
    jupytext_version: 1.5.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

How to convert to/from Arrow and Parquet
========================================

The [Apache Arrow](https://arrow.apache.org/) data format is very similar to Awkward Array's, but they're not exactly the same. As such, arrays can _usually_ be shared without copying, but not always.

The [Apache Parquet](https://parquet.apache.org/) file format has strong connections to Arrow with a large overlap in available tools, and while it's also a columnar format like Awkward and Arrow, it is implemented in a different way, which emphasizes compact storage over random access.

```{code-cell} ipython3
import awkward as ak
import pyarrow as pa
import pyarrow.csv
import urllib.request
```

From Arrow to Awkward
---------------------

The function for Arrow → Awkward conversion is [ak.from_arrow](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_arrow.html).

The argument to this function can be any of the following types from the pyarrow library:

   * `pyarrow.lib.Array`
   * `pyarrow.lib.ChunkedArray`
   * `pyarrow.lib.RecordBatch`
   * `pyarrow.lib.Table`

and they are converted into non-partitioned, non-virtual Awkward Arrays. (Any disjoint chunks in the Arrow array are concatenated.)

```{code-cell} ipython3
pa_array = pa.array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
pa_array
```

```{code-cell} ipython3
ak.from_arrow(pa_array)
```

Here is an example of an Arrow Table, derived from CSV. (Printing a table shows its field types.)

```{code-cell} ipython3
pokemon = urllib.request.urlopen("https://gist.githubusercontent.com/armgilles/194bcff35001e7eb53a2a8b441e8b2c6/raw/92200bc0a673d5ce2110aaad4544ed6c4010f687/pokemon.csv")
table = pyarrow.csv.read_csv(pokemon)
table
```

Awkward Array doesn't make a deep distinction between "arrays" and "tables" the way Arrow does: the Awkward equivalent of an Arrow table is just an Awkward Array of record type.

```{code-cell} ipython3
array = ak.from_arrow(table)
array
```

The Awkward equivalent of Arrow's schemas is [ak.type](https://awkward-array.readthedocs.io/en/latest/_auto/ak.type.html).

```{code-cell} ipython3
ak.type(array)
```

```{code-cell} ipython3
ak.to_list(array[0])
```

This array is ready for data analysis.

```{code-cell} ipython3
array[array.Legendary].Attack - array[array.Legendary].Defense
```

From Awkward to Arrow
---------------------

The function for Awkward → Arrow conversion is [ak.to_arrow](https://awkward-array.readthedocs.io/en/latest/_auto/ak.to_arrow.html). This function always returns

   * `pyarrow.lib.Array`

type.

```{code-cell} ipython3
ak_array = ak.Array([{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}])
ak_array
```

```{code-cell} ipython3
pa_array = ak.to_arrow(ak_array)
pa_array
```

```{code-cell} ipython3
type(pa_array)
```

```{code-cell} ipython3
isinstance(pa_array, pa.lib.Array)
```

If you need `pyarrow.lib.RecordBatch`, you can build this using pyarrow:

```{code-cell} ipython3
pa_batch = pa.RecordBatch.from_arrays([
    ak.to_arrow(ak_array.x),
    ak.to_arrow(ak_array.y),
], ["x", "y"])
pa_batch
```

If you need `pyarrow.lib.Table`, you can build this using pyarrow:

```{code-cell} ipython3
pa_table = pa.Table.from_batches([pa_batch])
pa_table
```

The columns of this Table are `pa.lib.ChunkedArray` instances:

```{code-cell} ipython3
pa_table[0]
```

```{code-cell} ipython3
pa_table[1]
```

Arrow Tables are closely aligned with Pandas DataFrames, so

```{code-cell} ipython3
pa_table.to_pandas()
```

shares memory as much as is possible, which [can be faster than constructing Pandas directly](https://ursalabs.org/blog/fast-pandas-loading/).

+++

Reading/writing data streams and random access files
----------------------------------------------------

Arrow has several methods for interfacing to data streams and disk-bound files, [see the official documentation](https://arrow.apache.org/docs/python/ipc.html) for instructions.

When following those instructions, remember that [ak.from_arrow](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_arrow.html) can accept pyarrow Arrays, ChunkedArrays, RecordBatches, and Tables, but [ak.to_arrow](https://awkward-array.readthedocs.io/en/latest/_auto/ak.to_arrow.html) _only returns Arrow Arrays_.

For instance, when writing to an IPC stream, Arrow requires RecordBatches, so you need to build them:

```{code-cell} ipython3
ak_array = ak.Array([{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}])
ak_array
```

```{code-cell} ipython3
first_batch = pa.RecordBatch.from_arrays([
    ak.to_arrow(ak_array.x),
    ak.to_arrow(ak_array.y),
], ["x", "y"])
first_batch.schema
```

```{code-cell} ipython3
sink = pa.BufferOutputStream()
writer = pa.ipc.new_stream(sink, first_batch.schema)

writer.write_batch(first_batch)

for i in range(5):
    next_batch = pa.RecordBatch.from_arrays([
        ak.to_arrow(ak_array.x),
        ak.to_arrow(ak_array.y),
    ], ["x", "y"])
    
    writer.write_batch(next_batch)

writer.close()
```

```{code-cell} ipython3
bytes(sink.getvalue())
```

But when reading them back, we can just pass the RecordBatches (yielded by the RecordBatchStreamReader `reader`), we can just pass them to [ak.from_arrow](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_arrow.html):

```{code-cell} ipython3
reader = pa.ipc.open_stream(sink.getvalue())
reader.schema
```

```{code-cell} ipython3
for batch in reader:
    print(repr(ak.from_arrow(batch)))
```

Reading/writing the Feather file format
---------------------------------------

Feather is a lightweight file format that puts Arrow Tables in disk-bound files, [see the official documentation](https://arrow.apache.org/docs/python/feather.html) for instructions.

When following those instructions, remember that [ak.from_arrow](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_arrow.html) can accept Arrow Tables, but [ak.to_arrow](https://awkward-array.readthedocs.io/en/latest/_auto/ak.to_arrow.html) _only returns Arrow Arrays_.

For instance, when writing to a Feather file, Arrow requires Tables, so you need to build them:

```{code-cell} ipython3
ak_array = ak.Array([{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}])
ak_array
```

```{code-cell} ipython3
pa_batch = pa.RecordBatch.from_arrays([
    ak.to_arrow(ak_array.x),
    ak.to_arrow(ak_array.y),
], ["x", "y"])

pa_table = pa.Table.from_batches([pa_batch])
pa_table
```

```{code-cell} ipython3
import pyarrow.feather

pyarrow.feather.write_feather(pa_table, "/tmp/example.feather")
```

But when reading them back, we can just pass the Arrow Table to [ak.from_arrow](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_arrow.html).

```{code-cell} ipython3
from_feather = pyarrow.feather.read_table("/tmp/example.feather")
from_feather
```

```{code-cell} ipython3
type(from_feather)
```

```{code-cell} ipython3
ak.from_arrow(from_feather)
```

Reading/writing the Parquet file format
---------------------------------------

With data converted to and from Arrow, it can then be saved and loaded from Parquet files. [Arrow's official Parquet documentation](https://arrow.apache.org/docs/python/parquet.html) provides instructions for converting Arrow to and from Parquet, but Parquet is a sufficiently important file format that Awkward has specialized functions for it.

The [ak.to_parquet](https://awkward-array.readthedocs.io/en/latest/_auto/ak.to_parquet.html) function writes Awkward Arrays as Parquet files. It has relatively few options.

```{code-cell} ipython3
ak_array = ak.Array([{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}])
ak_array
```

```{code-cell} ipython3
ak.to_parquet(ak_array, "/tmp/example.parquet")
```

The [ak.from_parquet](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_parquet.html) function reads Parquet files as Awkward Arrays, with quite a few more options. Basic usage just gives you the Awkward Array back.

```{code-cell} ipython3
ak.from_parquet("/tmp/example.parquet")
```

(Note that the type is not exactly the same: all fields are nullable—potentially have missing data—in Parquet, so the numbers become numbers-or-None after passing through Parquet.)

Since the data in a Parquet file may be huge, there are `columns` and `row_groups` options to read back only _part_ of the file.

   * Parquet's "columns" correspond to Awkward's record "fields," though Parquet columns cannot be nested.
   * Parquet's "row groups" are ranges of contiguous elements, such as "`1000-2000`". They correspond to Awkward's "partitioning." Neither Parquet row groups nor Awkward partitions can be nested.

For instance, the expression

```{code-cell} ipython3
ak.from_parquet("/tmp/example.parquet", columns=["x"])
```

Doesn't read column `"y"`.

To take advantage of this, you might need to "explode" nested records when writing so that they become top-level columns.

```{code-cell} ipython3
ak_array = ak.Array([
    [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}],
    [],
    [{"x": 4.4, "y": [1, 2, 3, 4]}, {"x": 5.5, "y": [1, 2, 3, 4, 5]}],
])
ak_array
```

```{code-cell} ipython3
ak.to_parquet(ak_array, "/tmp/example-exploded.parquet", explode_records=True)
```

(At the time of writing, this array can't be written by pyarrow without exploding records anyway.)

```{code-cell} ipython3
exploded = ak.from_parquet("/tmp/example-exploded.parquet")
exploded
```

The data type is different:

```{code-cell} ipython3
ak.type(ak_array)
```

```{code-cell} ipython3
ak.type(exploded)
```

but can be reconciled with [ak.zip](https://awkward-array.readthedocs.io/en/latest/_auto/ak.zip.html):

```{code-cell} ipython3
ak.type(ak.zip({"x": exploded.x, "y": exploded.y}))
```

(apart from the `option` due to passing through Parquet).

+++

Lazy Parquet file reading
-------------------------

The reason for limiting the `columns` or `row_groups` is to reduce the amount of data that needs to be read. These parameters of the [ak.from_parquet](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_parquet.html) function are helpful if you know in advance which columns and row groups you need, but reading can also be triggered by use with `lazy=True`.

Evaluating the following line only reads the file's metadata, none of the columns or row groups:

```{code-cell} ipython3
ak_lazy = ak.from_parquet("/tmp/example.parquet", lazy=True)
```

We can see this by peeking at the array's internal representation:

```{code-cell} ipython3
ak_lazy.layout
```

Both fields are [VirtualArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.VirtualArray.html) nodes, which delay evaluation of an [ArrayGenerator](https://awkward-array.readthedocs.io/en/latest/ak.layout.ArrayGenerator.html).

The array also has an attached cache, which we can query to see that nothing has been read:

```{code-cell} ipython3
ak_lazy.caches
```

But when we actually access a field, the data are read from disk:

```{code-cell} ipython3
ak_lazy.x
```

```{code-cell} ipython3
ak_lazy.caches
```

```{code-cell} ipython3
ak_lazy.y
```

```{code-cell} ipython3
ak_lazy.caches
```

A custom `lazy_cache` can be supplied: the default is a non-evicting Python dict attached to the output [ak.Array](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Array.html) but may be any [Mapping](https://docs.python.org/3/library/collections.abc.html#collections-abstract-base-classes).

Furthermore, a `lazy_cache_key` can be supplied to ensure uniqueness of cache keys across processes (the default is unique in process).
