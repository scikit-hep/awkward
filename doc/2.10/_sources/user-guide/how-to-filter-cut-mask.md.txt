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

How to filter arrays: cutting vs. masking
=========================================

```{code-cell} ipython3
import awkward as ak
import numpy as np
```

## The problem with slicing

When you write a mathematical formula using binary operators like `+` and `*`, or [NumPy universal functions (ufuncs)](https://numpy.org/doc/stable/reference/ufuncs.html) like `np.sqrt`, the shapes of nested lists must align. If the arrays in an expression were derived from a single array, this is often automatic. For instance,

```{code-cell} ipython3
original_array = ak.Array([
    [
        {"title": "zero", "x": 0, "y": 0},
        {"title": "one", "x": 1, "y": 1.1},
        {"title": "two", "x": 2, "y": 2.2},
    ],
    [],
    [
        {"title": "three", "x": 3, "y": 3.3},
        {"title": "four", "x": 4, "y": 4.4},
    ],
    [
        {"title": "five", "x": 5, "y": 5.5},
    ],
    [
        {"title": "six", "x": 6, "y": 6.6},
        {"title": "seven", "x": 7, "y": 7.7},
        {"title": "eight", "x": 8, "y": 8.8},
        {"title": "nine", "x": 9, "y": 9.9},
    ],
])
```

```{code-cell} ipython3
array_x = original_array.x
array_y = original_array.y
```

The `array_x` and `array_y` have the same number of lists and the same numbers of items in each list because they were both slices of the `original_array`.

```{code-cell} ipython3
array_x
```

```{code-cell} ipython3
array_y
```

Thus, they can be used together in a mathematical formula.

```{code-cell} ipython3
array_x**2 + array_y**2
```

However, if one array is sliced, or if the two arrays are sliced by different criteria, they would no longer line up:

```{code-cell} ipython3
sliced_x = array_x[array_x > 3]
sliced_y = array_y[array_y > 3]
```

```{code-cell} ipython3
sliced_x
```

```{code-cell} ipython3
sliced_y
```

Notice that the first was sliced with `array_x > 3` and the second was sliced with `array_y > 3`, and as a result, the third list differs in length between the two arrays:

```{code-cell} ipython3
sliced_x[2], sliced_y[2]
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

If we try to use these together, we get a ValueError:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [raises-exception]
---
sliced_x**2 + sliced_y**2
```

Sometimes, these misalignments are overt, but sometimes they're subtle and embedded deep within a very large array. You can start investigating a problem like this with {func}`ak.num`:

```{code-cell} ipython3
ak.num(sliced_x) != ak.num(sliced_y)
```

```{code-cell} ipython3
np.nonzero(ak.to_numpy(ak.num(sliced_x) != ak.num(sliced_y)))
```

But it's also possible to avoid them in the first place.

## Masking with missing values

The problem was that the two arrays' shapes changed differently; instead, we'll slice them in such a way that their shapes don't change at all.

The {func}`ak.mask` function uses a boolean array like a slice, but takes values that line up with `False` and returns `None` instead of removing them.

```{code-cell} ipython3
ak.mask(array_x, array_x > 3)
```

It can also be accessed as an array property, with square brackets, so that it resembles a slice:

```{code-cell} ipython3
masked_x = array_x.mask[array_x > 3]
masked_y = array_y.mask[array_y > 3]
```

```{code-cell} ipython3
masked_x
```

```{code-cell} ipython3
masked_y
```

The results of these two masks can be used in a mathematical expression because they line up:

```{code-cell} ipython3
result = masked_x**2 + masked_y**2
result
```

Now only one problem remains: the `None` (missing) values might be undesirable in the output. There are several ways to get rid of them:

* {func}`ak.drop_none` eliminates `None`, like a slice, but it can be done once at the end of a calculation,
* {func}`ak.fill_none` replaces `None` with a chosen value,
* {func}`ak.flatten` removes list structure, and if the `None` values are at the level of a list (the ones in `result` aren't), they'll be removed too,
* {func}`ak.singletons` replaces `None` with `[]` and any other value `x` with `[x]`. The resulting lists all have length 0 or length 1.

```{code-cell} ipython3
ak.drop_none(result, axis=1)
```

```{code-cell} ipython3
ak.fill_none(result, -1, axis=1)
```

```{code-cell} ipython3
ak.singletons(result, axis=1)
```

As a final note, the difference between using {func}`ak.drop_none` and slicing with the result of {func}`ak.is_none` is that {func}`ak.drop_none` also removes "missingness" from the data type; a slice does not.

```{code-cell} ipython3
result[~ak.is_none(result, axis=1)]
```

(Note the `?` for "option-type" before `float64`. This could have consequences, good or bad, at a later stage in processing.)
