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

How to convert to/from ROOT RDataFrame
======================================

The [ROOT RDataFrame](https://root.cern.ch/doc/master/classROOT_1_1RDataFrame.html) is a declarative, parallel framework for data analysis and manipulation. `RDataFrame` reads columnar data via a data source. The transformations can be applied to the data to select rows and/or to define new columns, and to produce results: histograms, etc.

```{code-cell} ipython3
import awkward as ak
import ROOT
```

From Awkward to RDataFrame
--------------------------

The function for Awkward → `RDataFrame` conversion is {func}`ak.to_rdataframe`.

The argument to this function requires a dictionary: `{ <column name string> : <awkwad array> }`. This function always returns

   * {class}`cppyy.gbl.ROOT.RDF.RInterface<ROOT::Detail::RDF::RLoopManager,void>`

object.

```{code-cell} ipython3
array_x = ak.Array(
    [
        {"x": [1.1, 1.2, 1.3]},
        {"x": [2.1, 2.2]},
        {"x": [3.1]},
        {"x": [4.1, 4.2, 4.3, 4.4]},
        {"x": [5.1]},
    ]
)
array_y = ak.Array([1, 2, 3, 4, 5])
array_z = ak.Array([[1.1], [2.1, 2.3, 2.4], [3.1], [4.1, 4.2, 4.3], [5.1]])
```

The arrays given for each column have to be equal length:

```{code-cell} ipython3
assert len(array_x) == len(array_y) == len(array_z)
```

The dictionary key defines a column name in RDataFrame.

```{code-cell} ipython3
df = ak.to_rdataframe({"x": array_x, "y": array_y, "z": array_z})
```

The {func}`ak.to_rdataframe` function presents a generated-on-demand Awkward Array view as an `RDataFrame` source. There is a small overhead of generating Awkward RDataSource C++ code. This operation does not execute the `RDataFrame` event loop. The array data are not copied.

The column readers are generated based on the run-time type of the views. Here is a description of the `RDataFrame` columns:

```{code-cell} ipython3
df.Describe().Print()
```

The `x` column contains an Awkward Array with a made-up type; `awkward::Record_cKnX5DyNVM`.

Awkward Arrays are dynamically typed, so in a C++ context, the type name is hashed. In practice, there is no need to know the type. The C++ code should use a placeholder type specifier `auto`. The type of the variable that is being declared will be automatically deduced from its initializer.


From RDataFrame to Awkward
--------------------------

The function for `RDataFrame`  → Awkward conversion is {func}`ak.from_rdataframe`. The argument to this function accepts a tuple of strings that are the `RDataFrame` column names. By default this function returns

   * {class}`ak.Array`

type.

```{code-cell} ipython3
array = ak.from_rdataframe(
    df,
    columns=(
        "x",
        "y",
        "z",
    ),
)
array
```

When `RDataFrame` runs multi-threaded event loops, the entry processing order is not guaranteed:

```{code-cell} ipython3
ROOT.ROOT.EnableImplicitMT()
```

+++

Let's recreate the dataframe, to reflect the new multi-threading mode

```{code-cell} ipython3
df = ak.to_rdataframe({"x": array_x, "y": array_y, "z": array_z})
```

+++

If the `keep_order` parameter set to `True`, the columns will keep order after filtering:

```{code-cell} ipython3
df = df.Filter("y % 2 == 0")

array = ak.from_rdataframe(
    df,
    columns=(
        "x",
        "y",
        "z",
    ),
    keep_order=True,
)
array
```
