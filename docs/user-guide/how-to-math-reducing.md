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

How to reduce dimensions (sum/min/any/all)
==========================================

After elementwise functions, dimension-reducer functions are the most commonly used. These functions replace a list of numbers with a single, scalar number by adding, multiplying, minimizing, maximizing, or performing logical-or ("any") or logical-and ("all").

These are also called aggregation functions; in relational databases, SQL, and data-frames, aggregations are applied after a "group by" operation. Awkward Array doesn't have "group by" operations; lists are already grouped.

```{code-cell} ipython3
import awkward as ak
import numpy as np
```

## First reducer: `ak.sum`

To illustrate all of these functions, let's consider addition. Given an array:

```{code-cell} ipython3
array = ak.Array([[1, 2, 3], [4, 5], [], [6]])
```

{func}`ak.sum` with no arguments adds all of the values in the nested lists, just like `np.sum`.

```{code-cell} ipython3
ak.sum(array)
```

With Awkward Arrays, it's usually more useful to supply an `axis` argument to reduce one dimension, rather than all dimensions.

For reasons that will be explained below, `axis=-1` is the most frequently useful.

```{code-cell} ipython3
ak.sum(array, axis=-1)
```

### The `axis` argument

Before getting deeper into the `axis` argument, let's consider a NumPy array with more dimensions.

```{code-cell} ipython3
array3d = np.array([
    [
        [    1,     2,     3,     4,     5],
        [   10,    20,    30,    40,    50],
        [  100,   200,   300,   400,   500],
    ],
    [
        [0.1  , 0.2  , 0.3  , 0.4  , 0.5  ],
        [0.01 , 0.02 , 0.03 , 0.04 , 0.05 ],
        [0.001, 0.002, 0.003, 0.004, 0.005],
    ],
])

with np.printoptions(suppress=True):
    print(array3d)
```

This array has 3 dimensions, so in addition to `axis=None` (reduce everything to a scalar), there are 3 possible axis values.

The first case, `axis=0`, adds the first 3×5 block to the second 3×5 block, i.e. summing over the first (length-2) dimension. Thus, the `1` is added to `0.1`, the `2` is added to `0.2`, and so on until the `500` is added to `0.005`.

```{code-cell} ipython3
with np.printoptions(suppress=True):
    print(np.sum(array3d, axis=0))
```

The second case, `axis=1`, adds vertically within each 3×5 block, i.e. summing over the second (length-3) dimension. What's left are two lists of length 5.

```{code-cell} ipython3
with np.printoptions(suppress=True):
    print(np.sum(array3d, axis=1))
```

The third case, `axis=2`, adds horizontally within each 3×5 block, i.e. summing over the third (length-5) dimension. What's left are two lists of length 3.

```{code-cell} ipython3
with np.printoptions(suppress=True):
    print(np.sum(array3d, axis=2))
```

Since negative `axis` counts from the other end of the scale,

* `axis=0` is equivalent to `axis=-3`
* `axis=1` is equivalent to `axis=-2`
* `axis=2` is equivalent to `axis=-1`.

### The `axis` argument with ragged lists

Awkward Arrays allow the lengths of lists in an array to differ, so we can have

```{code-cell} ipython3
array_ragged = ak.Array([
    [  1,   2,   3     ],
    [ 10,  20          ],
    [100, 200, 300, 400],
])
array_ragged
```

As before, `axis=-1` sums over the innermost lists, replacing each of the 3 horizontal rows with a sum.

```{code-cell} ipython3
ak.sum(array_ragged, axis=-1)
```

And `axis=-2` sums vertically, replacing each of the 4 vertical columns with a sum. Since the list lengths differ, some of the places we might expect to see a value is an empty gap—it contributes nothing to the result.

```{code-cell} ipython3
ak.sum(array_ragged, axis=0)
```

We also have to choose a convention: should the values be left-aligned or right-aligned within their lists? Awkward Array choses left-aligned.

In ragged data from real datasets, summing over whole lists usually has more meaning than summing over parts of different lists, so `axis=-1` is usually the most meaningful choice of `axis`.

+++

### The `axis` argument with missing data

+++

Just as empty gaps contribute nothing to the sum, missing values (`None`) don't contribute anything, either.

```{code-cell} ipython3
array_ragged = ak.Array([
    [None, None,    3,    4],
    [  10, None,   30      ],
    [ 100,  200,  300,  400],
])
array_ragged
```

`axis=-1` sums over each inner list, horizontally, replacing it with a scalar.

```{code-cell} ipython3
ak.sum(array_ragged, axis=-1)
```

And `axis=-2` sums over the outer dimension, vertically.

```{code-cell} ipython3
ak.sum(array_ragged, axis=-2)
```

For {func}`ak.sum`, each `None` has the same effect as a `0` value, for {func}`ak.prod` (multiplication), each `None` has the same effect as a `1` value, etc.

## The `keepdims` argument

Sometimes, you want to replace lists with a length-1 list, rather than a scalar. `keepdims=True` does that.

```{code-cell} ipython3
ak.sum(array_ragged, axis=-1, keepdims=True)
```

```{code-cell} ipython3
ak.sum(array_ragged, axis=-2, keepdims=True)
```

The `keepdims` argument is particularly useful for {func}`ak.argmin` and {func}`ak.argmax`, which return positions in a list where the value is minimized or maximized. Those positions can only be used as slice indexes if they're at the right nesting level, which `keepdims=True` maintains.

## Other reducers

* The {func}`ak.prod` reducer multiplies, rather than adding.
* {func}`ak.min` and {func}`ak.max` minimize and maximize, returning `None` for empty lists.
* {func}`ak.argmin` and {func}`ak.argmax` return the index positions of the minimum or maximum value, with `None` for empty lists.
* {func}`ak.nansum`, {func}`ak.nanprod`, {func}`ak.nanmin`, {func}`ak.nanmax`, {func}`ak.nanargmin`, and {func}`ak.nanargmax` ignore floating-point `nan` values before operating, the way that all reducers ignore `None` values before operating.
* {func}`ak.count_nonzero` counts non-zero values.
* {func}`ak.count` simply counts values. In NumPy, there's no need for such a function because it would return constants (drawn from the NumPy array's `shape`), but for ragged arrays, it counts the number of values that enter into a reduction. {func}`ak.num` also returns lengths of lists, but in a way that's more useful for slicing; {func}`ak.count` is useful as the denominator of expressions in which another reducer (with the same `axis` and `keepdims` choices) is in the numerator.
* {func}`ak.any` and {func}`ak.all` reduce like logical-or and logical-and, which makes them particularly useful in slices (below).

## Reducing over "any" and "all"

{func}`ak.any` and {func}`ak.all` reduce boolean arrays, asking if a predicate is satisfied by "any" item or "all" items, respectively.

```{code-cell} ipython3
array_bool = ak.Array([
    [False, False,  True,  True],
    [False,  True, False,  True],
    [False,  True,  True,  True],
])
array_bool
```

```{code-cell} ipython3
ak.any(array_bool, axis=-1)
```

```{code-cell} ipython3
ak.any(array_bool, axis=-2)
```

```{code-cell} ipython3
ak.all(array_bool, axis=-1)
```

```{code-cell} ipython3
ak.all(array_bool, axis=-2)
```

Since logical-or is like addition of booleans and logical-and is like multiplication, these reducers could have been replaced with {func}`ak.sum` and {func}`ak.prod`, but they're very useful to have because they make some boolean-array slices easier to read.

```{code-cell} ipython3
array = ak.Array([[0, 1, 2], [], [-3, 4], [-5], [-6, -7, -8, -9]])
array
```

Select _whole lists_ if _any_ of their values are negative:

```{code-cell} ipython3
array[ak.any(array < 0, axis=-1)]
```

Select _whole lists_ if _all_ of their values are negative:

```{code-cell} ipython3
array[ak.all(array < 0, axis=-1)]
```

(If a list is empty, all of its elements satisfy a constraint.)

In both cases above, the selection can be read like an English sentence, "select lists if _any_..." or "select lists if _all_...".

## Heterogeneous data and records cannot be reduced

These two kinds of data types are not reducible. Heterogeneous data allows an array to have multiple numbers of dimensions, so the problem is ill-posed:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [raises-exception]
---
ak.sum(ak.Array([[1.1, 2.2, 3.3], [], 4.4, 5.5]))
```

And records are sometimes used to represent data with coordinates; applying {func}`ak.sum` to non-Cartesian coordinates would be a subtle error.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [raises-exception]
---
ak.sum(ak.Array([{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}]), axis=-1)
```
