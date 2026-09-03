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

How to convert to Pandas
========================

[Pandas](https://pandas.pydata.org/) is a data analysis library for ordered time-series and relational data. In general, Pandas does not define operations for manipulating nested data structures, but in some cases, [MultiIndex/advanced indexing](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html) can do equivalent things.

```{code-cell} ipython3
import awkward as ak
import pandas as pd
import pyarrow as pa
import urllib.request
```

From Pandas to Awkward
----------------------

At the time of writing, there is no `ak.from_dataframe` function, but such a thing could be useful.

However, [Apache Arrow](https://arrow.apache.org/) can be converted to and from Awkward Arrays, and Arrow can be converted to and from Pandas (sometimes zero-copy). See below for more on conversion through Arrow.

+++

From Awkward to Pandas
----------------------

The function for Awkward → Pandas conversion is {func}`ak.to_dataframe`.

```{code-cell} ipython3
ak_array = ak.Array(
    [
        {"x": 1.1, "y": 1, "z": "one"},
        {"x": 2.2, "y": 2, "z": "two"},
        {"x": 3.3, "y": 3, "z": "three"},
        {"x": 4.4, "y": 4, "z": "four"},
        {"x": 5.5, "y": 5, "z": "five"},
    ]
)
ak_array
```

```{code-cell} ipython3
ak.to_dataframe(ak_array)
```

Awkward record field names are converted into Pandas column names, even if nested within lists.

```{code-cell} ipython3
ak_array = ak.Array(
    [
        [
            {"x": 1.1, "y": 1, "z": "one"},
            {"x": 2.2, "y": 2, "z": "two"},
            {"x": 3.3, "y": 3, "z": "three"},
        ],
        [],
        [{"x": 4.4, "y": 4, "z": "four"}, {"x": 5.5, "y": 5, "z": "five"}],
    ]
)
ak_array
```

```{code-cell} ipython3
ak.to_dataframe(ak_array)
```

In this case, we see that the `"x"`, `"y"`, and `"z"` fields are separate columns, but also that the index is now hierarchical, a [MultiIndex](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.MultiIndex.html). Nested lists become MultiIndex rows and nested records become MultiIndex columns.

Here is an example with three levels of depth:

```{code-cell} ipython3
ak_array = ak.Array(
    [
        [[1.1, 2.2], [], [3.3]],
        [],
        [[4.4], [5.5, 6.6]],
        [[7.7]],
        [[8.8]],
    ]
)
ak_array
```

```{code-cell} ipython3
ak.to_dataframe(ak_array)
```

And here is an example with nested records/hierarchical columns:

```{code-cell} ipython3
ak_array = ak.Array(
    [
        {"I": {"a": _, "b": {"i": _}}, "II": {"x": {"y": {"z": _}}}}
        for _ in range(0, 50, 10)
    ]
)
ak_array
```

```{code-cell} ipython3
ak.to_dataframe(ak_array)
```

Although nested lists and records can be represented using Pandas's MultiIndex, different-length lists in the same data structure can only be translated without loss into multiple DataFrames. This is because a DataFrame can have only one MultiIndex, but lists of different lengths require different MultiIndexes.

```{code-cell} ipython3
ak_array = ak.Array(
    [
        {"x": [], "y": [4.4, 3.3, 2.2, 1.1]},
        {"x": [1], "y": [3.3, 2.2, 1.1]},
        {"x": [1, 2], "y": [2.2, 1.1]},
        {"x": [1, 2, 3], "y": [1.1]},
        {"x": [1, 2, 3, 4], "y": []},
    ]
)
ak_array
```

To avoid losing any data, {func}`ak.to_dataframe` can be used with `how=None` (the default is `how="inner"`) to return a _list_ of the minimum number of DataFrames needed to encode the data.

In `how=None` mode, {func}`ak.to_dataframe` always returns a list (sometimes with only one item).

```{code-cell} ipython3
ak.to_dataframe(ak_array, how=None)
```

The default `how="inner"` combines the above into a single DataFrame using [pd.merge](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.merge.html). This operation is lossy.

```{code-cell} ipython3
ak.to_dataframe(ak_array, how="inner")
```

The value of `how` is passed to [pd.merge](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.merge.html), so outer joins are possible as well.

```{code-cell} ipython3
ak.to_dataframe(ak_array, how="outer")
```

Conversion through Apache Arrow
-------------------------------

Since [Apache Arrow](https://arrow.apache.org/) can be converted to and from Awkward Arrays and Pandas, Arrow can connect Awkward and Pandas in both directions. This is an alternative to {func}`ak.to_pandas` (described above) with different behavior.

As described in the tutorial on Arrow, the {func}`ak.to_arrow` function returns a {class}`pyarrow.lib.Arrow` object. Arrow's conversion to Pandas requires a {class}`pyarrow.lib.Table`.

```{code-cell} ipython3
ak_array = ak.Array(
    [
        [
            {"x": 1.1, "y": 1, "z": "one"},
            {"x": 2.2, "y": 2, "z": "two"},
            {"x": 3.3, "y": 3, "z": "three"},
        ],
        [],
        [{"x": 4.4, "y": 4, "z": "four"}, {"x": 5.5, "y": 5, "z": "five"}],
    ]
)
ak_array
```

```{code-cell} ipython3
pa_array = ak.to_arrow(ak_array)
pa_array
```

We can build a Table manually, ensuring that we set `extensionarray=False`. The `extensionarray` flag is normally `True`, and enables Awkward to preserve metadata through Arrow transformations. However, tools like Arrow's Pandas conversion do not recognise Awkward's special extension type, so we must take care to provide Arrow with native types:

```{code-cell} ipython3
pa_table = pa.Table.from_batches(
    [
        pa.RecordBatch.from_arrays(
            [
                ak.to_arrow(ak_array.x, extensionarray=False),
                ak.to_arrow(ak_array.y, extensionarray=False),
                ak.to_arrow(ak_array.z, extensionarray=False),
            ],
            ["x", "y", "z"],
        )
    ]
)
pa_table
```

```{code-cell} ipython3
pa_table.to_pandas()
```

Note that this is different from the output of {func}`ak.to_pandas`:

```{code-cell} ipython3
ak.to_dataframe(ak_array)
```

The Awkward → Arrow → Pandas route leaves the lists as nested data within each cell, whereas {func}`ak.to_dataframe` encodes the nested structure with a [MultiIndex/advanced indexing](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html) and puts simple values in each cell. Depending on your needs, one or the other may be desirable.

Finally, the Pandas → Arrow → Awkward is currently the only means of turning Pandas DataFrames into Awkward Arrays.

```{code-cell} ipython3
pokemon = urllib.request.urlopen(
    "https://gist.githubusercontent.com/armgilles/194bcff35001e7eb53a2a8b441e8b2c6/raw/92200bc0a673d5ce2110aaad4544ed6c4010f687/pokemon.csv"
)
df = pd.read_csv(pokemon)
df
```

```{code-cell} ipython3
ak_array = ak.from_arrow(pa.Table.from_pandas(df))
ak_array
```

```{code-cell} ipython3
ak.type(ak_array)
```

```{code-cell} ipython3
ak.to_list(ak_array[0])
```

This array is ready for data analysis.

```{code-cell} ipython3
ak_array[ak_array.Legendary].Attack - ak_array[ak_array.Legendary].Defense
```
