---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

How to concatenate and interleave arrays
========================================

```{code-cell} ipython3
import awkward as ak
import numpy as np
import pandas as pd
```

## Simple concatenation

{func}`ak.concatenate` is an analog of [np.concatenate](https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html) (in fact, you can use [np.concatenate](https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html) where you mean {func}`ak.concatenate`). However, it applies to data of arbitrary data structures:

```{code-cell} ipython3
array1 = ak.Array([
    [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}],
    [],
    [{"x": 4.4, "y": [1, 2, 3, 4]}, {"x": 5.5, "y": [1, 2, 3, 4, 5]}],
])
array2 = ak.Array([
    [{"x": 6.6, "y": [1, 2, 3, 4, 5, 6]}],
    [{"x": 7.7, "y": [1, 2, 3, 4, 5, 6, 7]}],
])
```

```{code-cell} ipython3
ak.concatenate([array1, array2])
```

The arrays can even have different data types, in which case the output has union-type.

```{code-cell} ipython3
array3 = ak.Array([{"z": None}, {"z": 0}, {"z": 123}])
```

```{code-cell} ipython3
ak.concatenate([array1, array2, array3])
```

Keep in mind, however, that some operations can't deal with union-types (heterogeneous data), so you might want to avoid this.

## Interleaving lists with `axis > 0`

The default `axis=0` returns an array whose length is equal to the sum of the lengths of the input arrays.

Other `axis` values combine lists within the arrays, as long as the arrays have the same lengths.

```{code-cell} ipython3
array1 = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
array2 = ak.Array([[10, 20], [30], [40, 50, 60, 70]])
```

```{code-cell} ipython3
len(array1), len(array2)
```

```{code-cell} ipython3
ak.concatenate([array1, array2], axis=1)
```

This can be used in some non-trivial ways: sometimes a problem that doesn't seem to have anything to do with concatenation can be solved this way.

For instance, suppose that you have to pad some lists so that they start and stop with 0 (for some window-averaging procedure, perhaps). You can make the pad as a new array:

```{code-cell} ipython3
pad = np.zeros(len(array1))[:, np.newaxis]
pad
```

and concatenate it with `axis=1` to get the desired effect:

```{code-cell} ipython3
ak.concatenate([pad, array1, pad], axis=1)
```

Or similarly, to double the first value and double the last value (without affecting empty lists):

```{code-cell} ipython3
ak.concatenate([array1[:, :1], array1, array1[:, -1:]], axis=1)
```

The same applies for more deeply nested lists and `axis > 1`. Remember that `axis=-1` starts counting from the innermost dimension, outward.

## Emulating NumPy's "stack" functions

[np.stack](https://numpy.org/doc/stable/reference/generated/numpy.stack.html), [np.hstack](https://numpy.org/doc/stable/reference/generated/numpy.hstack.html), [np.vstack](https://numpy.org/doc/stable/reference/generated/numpy.vstack.html), and [np.dstack](https://numpy.org/doc/stable/reference/generated/numpy.dstack.html) are concatenations with [np.newaxis](https://numpy.org/doc/stable/reference/constants.html#numpy.newaxis) (reshaping to add a dimension of length 1).

```{code-cell} ipython3
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
```

```{code-cell} ipython3
np.stack([a, b])
```

```{code-cell} ipython3
np.concatenate([a[np.newaxis], b[np.newaxis]], axis=0)
```

```{code-cell} ipython3
np.stack([a, b], axis=1)
```

```{code-cell} ipython3
np.concatenate([a[:, np.newaxis], b[:, np.newaxis]], axis=1)
```

Since {func}`ak.concatenate` has the same interface as [np.concatenate](https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html) and Awkward Arrays can also be sliced with [np.newaxis](https://numpy.org/doc/stable/reference/constants.html#numpy.newaxis), they can be stacked the same way, with the addition of arbitrary data structures.

```{code-cell} ipython3
a = ak.Array([[1], [1, 2], [1, 2, 3]])
b = ak.Array([[4], [4, 5], [4, 5, 6]])
```

```{code-cell} ipython3
ak.concatenate([a[np.newaxis], b[np.newaxis]], axis=0)
```

```{code-cell} ipython3
ak.concatenate([a[:, np.newaxis], b[:, np.newaxis]], axis=1)
```

## Differences from Pandas

Concatenation in Awkward Array combines arrays lengthwise: by adding the lengths of the arrays or adding the lengths of lists within an array. It does not refer to adding fields to a record (that is, "adding columns to a table"). To add fields to a record, see {func}`ak.zip` or {func}`ak.Array.__setitem__` in {doc}`how to zip/unzip and project <how-to-restructure-zip-project>` and {doc}`how to add fields <how-to-restructure-add-fields>`. This is important to note because [pandas.concat](https://pandas.pydata.org/docs/reference/api/pandas.concat.html) does both, depending on its `axis` argument (and there's no equivalent in NumPy).

Here's a table-like example of concatenation in Awkward Array:

```{code-cell} ipython3
array1 = ak.Array({"column": [[1, 2, 3], [], [4, 5]]})
array2 = ak.Array({"column": [[1.1, 2.2, 3.3], [], [4.4, 5.5]]})
```

```{code-cell} ipython3
array1
```

```{code-cell} ipython3
array2
```

```{code-cell} ipython3
ak.concatenate([array1, array2], axis=0)
```

This is like Pandas for `axis=0`,

```{code-cell} ipython3
df1 = pd.DataFrame({"column": [[1, 2, 3], [], [4, 5]]})
df2 = pd.DataFrame({"column": [[1.1, 2.2, 3.3], [], [4.4, 5.5]]})
```

```{code-cell} ipython3
df1
```

```{code-cell} ipython3
df2
```

```{code-cell} ipython3
pd.concat([df1, df2], axis=0)
```

But for `axis=1`, they're quite different:

```{code-cell} ipython3
ak.concatenate([array1, array2], axis=1)
```

```{code-cell} ipython3
pd.concat([df1, df2], axis=1)
```

{func}`ak.concatenate` accepts any `axis` less than the number of dimensions in the arrays, but Pandas has only two choices, `axis=0` and `axis=1`.

Fields ("columns") of an Awkward Array are unrelated to array dimensions. If you want what [pandas.concat](https://pandas.pydata.org/docs/reference/api/pandas.concat.html) does with `axis=1`, you would use {func}`ak.zip`:

```{code-cell} ipython3
ak.zip({"column1": array1.column, "column2": array2.column}, depth_limit=1)
```

The `depth_limit` prevents {func}`ak.zip` from interleaving the lists further:

```{code-cell} ipython3
ak.zip({"column1": array1.column, "column2": array2.column})
```

which Pandas doesn't do because lists in Pandas cells are Python objects that it doesn't modify.
