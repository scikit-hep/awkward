---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

How to create arrays of missing data
====================================

Data at any level of an Awkward Array can be "missing," represented by `None` in Python.

This functionality is somewhat like NumPy's [masked arrays](https://numpy.org/doc/stable/reference/maskedarray.html), but masked arrays can only declare numerical values to be missing (not, for instance, a row of a 2-dimensional array) and they represent missing data with an `np.ma.masked` object instead of `None`.

Pandas also handles missing data, but in several different ways. For floating point columns, `NaN` (not a number) is used to mean "missing," and [as of version 1.0](https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html#missing-data-na), Pandas has a `pd.NA` object for missing data in other data types.

In Awkward Array, floating point `NaN` and a missing value are clearly distinct. Missing data, like all data in Awkward Arrays, are also not represented by any Python object; they are converted _to_ and _from_ `None` by {func}`ak.to_list` and {func}`ak.from_iter`.

```{code-cell} ipython3
import awkward as ak
import numpy as np
```

From Python None
----------------

The {class}`ak.Array` constructor and {func}`ak.from_iter` interpret `None` as a missing value, and {func}`ak.to_list` converts them back into `None`.

```{code-cell} ipython3
ak.Array([1, 2, 3, None, 4, 5])
```

The missing values can be deeply nested (missing integers):

```{code-cell} ipython3
ak.Array([[[[], [1, 2, None]]], [[[3]]], []])
```

They can be shallow (missing lists):

```{code-cell} ipython3
ak.Array([[[[], [1, 2]]], None, [[[3]]], []])
```

Or both:

```{code-cell} ipython3
ak.Array([[[[], [3]]], None, [[[None]]], []])
```

Records can also be missing:

```{code-cell} ipython3
ak.Array([{"x": 1, "y": 1}, None, {"x": 2, "y": 2}])
```

Potentially missing values are represented in the type string as "`?`" or "`option[...]`" (if the nested type is a list, which needs to be bracketed for clarity).

+++

From NumPy arrays
-----------------

Normal NumPy arrays can't represent missing data, but masked arrays can. Here is how one is constructed in NumPy:

```{code-cell} ipython3
numpy_array = np.ma.MaskedArray([1, 2, 3, 4, 5], [False, False, True, True, False])
numpy_array
```

It returns `np.ma.masked` objects if you try to access missing values:

```{code-cell} ipython3
numpy_array[0], numpy_array[1], numpy_array[2], numpy_array[3], numpy_array[4]
```

But it uses `None` for missing values in `tolist`:

```{code-cell} ipython3
numpy_array.tolist()
```

The {func}`ak.from_numpy` function converts masked arrays into Awkward Arrays with missing values, as does the {class}`ak.Array` constructor.

```{code-cell} ipython3
awkward_array = ak.Array(numpy_array)
awkward_array
```

The reverse, {func}`ak.to_numpy`, returns masked arrays if the Awkward Array has missing data.

```{code-cell} ipython3
ak.to_numpy(awkward_array)
```

But [np.asarray](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html), the usual way of casting data as NumPy arrays, does not. ([np.asarray](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html) is supposed to return a plain [np.ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html), which [np.ma.masked_array](https://numpy.org/doc/stable/reference/generated/numpy.ma.masked_array.html) is not.)

```{code-cell} ipython3
:tags: [raises-exception]

np.asarray(awkward_array)
```

Missing rows vs missing numbers
-------------------------------

In Awkward Array, a missing list is a different thing from a list whose values are missing. However, {func}`ak.to_numpy` converts it for you.

```{code-cell} ipython3
missing_row = ak.Array([[1, 2, 3], None, [4, 5, 6]])
missing_row
```

```{code-cell} ipython3
ak.to_numpy(missing_row)
```

NaN is not missing
------------------

Floating point `NaN` values are simply unrelated to missing values, in both Awkward Array and NumPy.

```{code-cell} ipython3
missing_with_nan = ak.Array([1.1, 2.2, np.nan, None, 3.3])
missing_with_nan
```

```{code-cell} ipython3
ak.to_numpy(missing_with_nan)
```

Missing values as empty lists
-----------------------------

Sometimes, it's useful to think about a potentially missing value as a length-1 list if it is not missing and a length-0 list if it is. (Some languages define the [option type as a kind of list](https://www.scala-lang.org/api/2.13.3/scala/Option.html).)

The Awkward functions {func}`ak.singletons` and {func}`ak.firsts` convert from "`None` form" to and from "lists form."

```{code-cell} ipython3
none_form = ak.Array([1, 2, 3, None, None, 5])
none_form
```

```{code-cell} ipython3
lists_form = ak.singletons(none_form)
lists_form
```

```{code-cell} ipython3
ak.firsts(lists_form)
```

Masking instead of slicing
--------------------------

The most common way of filtering data is to slice it with an array of booleans (usually the result of a calculation).

```{code-cell} ipython3
array = ak.Array([1, 2, 3, 4, 5])
array
```

```{code-cell} ipython3
booleans = ak.Array([True, True, False, False, True])
booleans
```

```{code-cell} ipython3
array[booleans]
```

The data can also be effectively filtered by replacing values with `None`. The following syntax does that:

```{code-cell} ipython3
array.mask[booleans]
```

(Or use the {func}`ak.mask` function.)

+++

An advantage of masking is that the length and nesting structure of the masked array is the same as the original array, so anything that broadcasts with one broadcasts with the other (so that unfiltered data can be used interchangeably with filtered data).

```{code-cell} ipython3
array + array.mask[booleans]
```

whereas

```{code-cell} ipython3
:tags: [raises-exception]

array + array[booleans]
```

With ArrayBuilder
-----------------

{class}`ak.ArrayBuilder` is described in more detail [in this tutorial](how-to-create-arraybuilder), but you can add missing values to an array using the `null` method or appending `None`.

(This is what {func}`ak.from_iter` uses internally to accumulate data.)

```{code-cell} ipython3
builder = ak.ArrayBuilder()

builder.append(1)
builder.append(2)
builder.null()
builder.append(None)
builder.append(3)

array = builder.snapshot()
array
```

In Numba
--------

Functions that Numba Just-In-Time (JIT) compiles can use {class}`ak.ArrayBuilder` or construct a boolean array for {func}`ak.mask`.

({class}`ak.ArrayBuilder` can't be constructed or converted to an array using `snapshot` inside a JIT-compiled function, but can be outside the compiled context.)

```{code-cell} ipython3
import numba as nb
```

```{code-cell} ipython3
@nb.jit
def example(builder):
    builder.append(1)
    builder.append(2)
    builder.null()
    builder.append(None)
    builder.append(3)
    return builder


builder = example(ak.ArrayBuilder())

array = builder.snapshot()
array
```

```{code-cell} ipython3
@nb.jit
def faster_example():
    data = np.empty(5, np.int64)
    mask = np.empty(5, np.bool_)
    data[0] = 1
    mask[0] = True
    data[1] = 2
    mask[1] = True
    mask[2] = False
    mask[3] = False
    data[4] = 5
    mask[4] = True
    return data, mask


data, mask = faster_example()

array = ak.Array(data).mask[mask]
array
```
