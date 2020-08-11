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
import awkward1 as ak
import pyarrow as pa
import pyarrow.csv
import urllib.request
```

From Arrow to Awkward
---------------------

A variety of types in the pyarrow library, including

   * `pyarrow.lib.Array`
   * `pyarrow.lib.ChunkedArray`
   * `pyarrow.lib.RecordBatch`
   * `pyarrow.lib.Table`

can all be converted to (non-partitioned, non-virtual) Awkward Arrays with [ak.from_arrow](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_arrow.html).

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

This array is ready for data analysis.

```{code-cell} ipython3
array[array.Legendary].Attack - array[array.Legendary].Defense
```

From Awkward to Arrow
---------------------
