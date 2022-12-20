---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
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

The function for Arrow → Awkward conversion is {func}`ak.from_arrow`.

The argument to this function can be any of the following types from the pyarrow library:

   * {class}`pyarrow.lib.Array`
   * {class}`pyarrow.lib.ChunkedArray`
   * {class}`pyarrow.lib.RecordBatch`
   * {class}`pyarrow.lib.Table`

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
pokemon = urllib.request.urlopen(
    "https://gist.githubusercontent.com/armgilles/194bcff35001e7eb53a2a8b441e8b2c6/raw/92200bc0a673d5ce2110aaad4544ed6c4010f687/pokemon.csv"
)
table = pyarrow.csv.read_csv(pokemon)
table
```

Awkward Array doesn't make a deep distinction between "arrays" and "tables" the way Arrow does: the Awkward equivalent of an Arrow table is just an Awkward Array of record type.

```{code-cell} ipython3
array = ak.from_arrow(table)
array
```

The Awkward equivalent of Arrow's schemas is {func}`ak.type`.

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

The function for Awkward → Arrow conversion is {func}`ak.to_arrow`. This function always returns

   * {class}`pyarrow.lib.Array`

type.

```{code-cell} ipython3
ak_array = ak.Array(
    [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}]
)
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

If you need {class}`pyarrow.lib.RecordBatch`, you can build this using pyarrow:

```{code-cell} ipython3
pa_batch = pa.RecordBatch.from_arrays(
    [
        ak.to_arrow(ak_array.x),
        ak.to_arrow(ak_array.y),
    ],
    ["x", "y"],
)
pa_batch
```

If you need {class}`pyarrow.lib.Table`, you can build this using pyarrow:

```{code-cell} ipython3
pa_table = pa.Table.from_batches([pa_batch])
pa_table
```

The columns of this Table are {class}`pa.lib.ChunkedArray` instances:

```{code-cell} ipython3
pa_table[0]
```

```{code-cell} ipython3
pa_table[1]
```

shares memory as much as is possible, which [can be faster than constructing Pandas directly](https://ursalabs.org/blog/fast-pandas-loading/).

+++ {"tags": []}

Reading/writing data streams and random access files
----------------------------------------------------

Arrow has several methods for interfacing to data streams and disk-bound files, [see the official documentation](https://arrow.apache.org/docs/python/ipc.html) for instructions.

When following those instructions, remember that {func}`ak.from_arrow` can accept {class}`pyarrow.lib.Array`, {class}`pyarrow.lib.ChunkedArray`, {class}`pyarrow.lib.RecordBatch`, and {class}`pyarrow.lib.Table`, but {func}`ak.to_arrow` _only returns {class}`pyarrow.lib.Array`_.

For instance, when writing to an IPC stream, Arrow requires {class}`pyarrow.lib.RecordBatch`, so you need to build them:

```{code-cell} ipython3
ak_array = ak.Array(
    [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}]
)
ak_array
```

```{code-cell} ipython3
first_batch = pa.RecordBatch.from_arrays(
    [
        ak.to_arrow(ak_array.x),
        ak.to_arrow(ak_array.y),
    ],
    ["x", "y"],
)
first_batch.schema
```

```{code-cell} ipython3
sink = pa.BufferOutputStream()
writer = pa.ipc.new_stream(sink, first_batch.schema)

writer.write_batch(first_batch)

for i in range(5):
    next_batch = pa.RecordBatch.from_arrays(
        [
            ak.to_arrow(ak_array.x),
            ak.to_arrow(ak_array.y),
        ],
        ["x", "y"],
    )

    writer.write_batch(next_batch)

writer.close()
```

```{code-cell} ipython3
bytes(sink.getvalue())
```

But when reading them back, we can just pass the record batches (yielded by the {class}`pyarrow.lib.RecordBatchStreamReader` `reader`) to {func}`ak.from_arrow`:

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

When following those instructions, remember that {func}`ak.from_arrow` can accept {class}`pyarrow.lib.Table`, but {func}`ak.to_arrow` _only returns {class}`pyarrow.lib.Array`_.

For instance, when writing to a Feather file, Arrow requires {class}`pyarrow.lib.Table`, so you need to build them:

```{code-cell} ipython3
ak_array = ak.Array(
    [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}]
)
ak_array
```

```{code-cell} ipython3
pa_batch = pa.RecordBatch.from_arrays(
    [
        ak.to_arrow(ak_array.x),
        ak.to_arrow(ak_array.y),
    ],
    ["x", "y"],
)

pa_table = pa.Table.from_batches([pa_batch])
pa_table
```

```{code-cell} ipython3
import pyarrow.feather

pyarrow.feather.write_feather(pa_table, "/tmp/example.feather")
```

But when reading them back, we can just pass the Arrow Table to {func}`ak.from_arrow`.

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

The {func}`ak.to_parquet` function writes Awkward Arrays as Parquet files. It has relatively few options.

```{code-cell} ipython3
ak_array = ak.Array(
    [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}]
)
ak_array
```

```{code-cell} ipython3
ak.to_parquet(ak_array, "/tmp/example.parquet")
```

The {func}`ak.from_parquet` function reads Parquet files as Awkward Arrays, with quite a few more options. Basic usage just gives you the Awkward Array back.

```{code-cell} ipython3
ak.from_parquet("/tmp/example.parquet")
```

Since the data in a Parquet file may be huge, there are `columns` and `row_groups` options to read back only _part_ of the file.

   * Parquet's "columns" correspond to Awkward's record "fields," though Parquet columns cannot be nested.
   * Parquet's "row groups" are ranges of contiguous elements, such as "`1000-2000`". They correspond to Awkward's "partitioning." Neither Parquet row groups nor Awkward partitions can be nested.

For instance, the expression

```{code-cell} ipython3
ak.from_parquet("/tmp/example.parquet", columns=["x"])
```

Doesn't read column `"y"`.
