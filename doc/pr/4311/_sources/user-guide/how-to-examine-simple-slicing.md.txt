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

How to examine an array with simple slicing
===========================================

Slicing data from an array is a basic operation in array-oriented data analysis. Awkward Array extends [NumPy's slicing capabilities](https://numpy.org/doc/stable/user/basics.indexing.html) to handle nested and ragged data structures. This tutorial illustrates several ways to slice an array.

For a complete list of slicing features, see {func}`ak.Array.__getitem__`.

```{code-cell} ipython3
import awkward as ak
import numpy as np
```

## Basic slicing with ranges

Much like NumPy, you can slice Awkward Arrays using simple ranges specified with colons (`start:stop:step`). Here’s an example of a regular (non-ragged) Awkward Array:

```{code-cell} ipython3
array = ak.Array(np.arange(10)**2)  # squaring numbers for clarity
array
```

To select the first five elements:

```{code-cell} ipython3
array[:5]
```

To select from the fifth-to-last onward:

```{code-cell} ipython3
array[-5:]
```

To select every other element starting from the second:

```{code-cell} ipython3
array[1::2]
```

## Multiple ranges for multiple dimensions

Similarly, for multidimensional data,

```{code-cell} ipython3
np_array3d = np.arange(2*3*5).reshape(2, 3, 5)
np_array3d
```

```{code-cell} ipython3
array3d = ak.Array(np_array3d)
array3d
```

```{code-cell} ipython3
np_array3d[1, ::2, 1:-1]
```

```{code-cell} ipython3
array3d[1, ::2, 1:-1]
```

Just as with NumPy, a single colon (`:`) means "take everything from this dimension" and an ellipsis (`...`) expands to all dimensions between two slices.

```{code-cell} ipython3
array3d[:, :, 1:-1]
```

```{code-cell} ipython3
array3d[..., 1:-1]
```

## Boolean array slices

Like NumPy's [advanced slicing](https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing), an array of booleans filters individual items. For instance, consider an array of booleans constructed by asking which elements of `array` are greater than 20:

```{code-cell} ipython3
array > 20
```

When applied to `array` between square brackets, the boolean array eliminates all items in which `array > 20` is `False`:

```{code-cell} ipython3
array[array > 20]
```

Boolean array slicing is more powerful than range slicing because the `True` and `False` values may have any pattern. The following selects only even numbers.

```{code-cell} ipython3
array % 2 == 0
```

```{code-cell} ipython3
array[array % 2 == 0]
```

## Integer array slices

You can also use arrays of integer indices to select specific elements.

```{code-cell} ipython3
indices = ak.Array([2, 5, 3])
array[indices]
```

If you are passing indexes directly between the `array`'s square brackets, be sure that they, too, are nested within square brackets (to be a list, rather than a tuple).

```{code-cell} ipython3
array[[2, 5, 3]]
```

In addition to picking elements out of order, you can pick the same element multiple times.

```{code-cell} ipython3
array[[2, 5, 5, 5, 5, 5, 3]]
```

Any slices that could be performed by boolean arrays can be performed by integer arrays, but only integer arrays can reorder and duplicate elements.

## Ragged array slicing

One of the unique features of Awkward Array is its ability to handle ragged arrays efficiently. Here's an example of a ragged array:

```{code-cell} ipython3
ragged_array = ak.Array([[10, 20, 30], [40], [], [50, 60]])
ragged_array
```

You can slice individual sublists like this:

```{code-cell} ipython3
ragged_array[1]
```

And you can perform slices that operate across the sublists:

```{code-cell} ipython3
ragged_array[:, :2]  # get first two elements of each sublist
```

Ranges and single indices mixed with slice notation allow the complexity of ragged slicing to express selecting ranges in nested lists, a feature unique to Awkward Array beyond NumPy’s capabilities. Here's an example where we skip the first element of each sublist that has more than one element:

```{code-cell} ipython3
ragged_array[ak.num(ragged_array) > 1, 1:]
```

## Boolean array slicing with missing data

When working with boolean arrays for slicing, the arrays can include `None` (missing) values. Awkward Array handles missing data gracefully during boolean slicing:

```{code-cell} ipython3
bool_mask = ak.Array([True, None, False, True])
array[bool_mask]
```

This ability to cope with missing data without failing or needing imputation is invaluable in data analysis tasks where missing data is common.
