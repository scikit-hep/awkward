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

How to convert to Pandas
========================

[Pandas](https://pandas.pydata.org/) is a data analysis library for ordered time-series and relational data. In general, Pandas does not define operations for manipulating nested data structures, but in some cases, [MultiIndex/advanced indexing](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html) can do equivalent things.

```{code-cell} ipython3
import awkward1 as ak
import pandas as pd
import pyarrow as pa
```

From Pandas to Awkward
----------------------

At the time of writing, there is no `ak.from_pandas` function, but such a thing could be useful.

However, [Apache Arrow](https://arrow.apache.org/) can be converted to and from Awkward Arrays, and Arrow can be converted to and from Pandas (sometimes zero-copy). See below for more on conversion through Arrow.

+++

From Awkward to Pandas
----------------------

The function for Awkward → Pandas conversion is [ak.to_pandas](https://awkward-array.readthedocs.io/en/latest/_auto/ak.to_pandas.html).

```{code-cell} ipython3
ak_array = ak.Array([
    {"x": 1.1, "y": 1, "z": "one"},
    {"x": 2.2, "y": 2, "z": "two"},
    {"x": 3.3, "y": 3, "z": "three"},
    {"x": 4.4, "y": 4, "z": "four"},
    {"x": 5.5, "y": 5, "z": "five"},
])
ak_array
```

```{code-cell} ipython3
ak.to_pandas(ak_array)
```

Conversion through Apache Arrow
-------------------------------

Since [Apache Arrow](https://arrow.apache.org/) can be converted to and from Awkward Arrays and Pandas, Arrow can connect Awkward and Pandas in both directions.

```{code-cell} ipython3

```
