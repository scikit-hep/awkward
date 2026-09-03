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

How Awkward broadcasting works
==============================

Functions that accept more than one array argument need to combine the elements of their array elements somehow, particularly if the input arrays have different numbers of dimensions. That combination is called "broadcasting." Broadcasting in Awkward Array is very similar to [NumPy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html), with some minor differences described at the end of this section.

```{code-cell} ipython3
import awkward as ak
import numpy as np
```

## Broadcasting in mathematical functions

Any function that takes more than one array argument has to broadcast them together; a common case of that is in binary operators of a mathematical expression:

```{code-cell} ipython3
array1 = ak.Array([[1, 2, 3], [], [4, 5]])
array2 = ak.Array([10, 20, 30])

array1 + array2
```

The single `10` in `array2` is added to every element of `[1, 2, 3]` in `array1`, and the single `30` is added to every element of `[4, 5]`. The single `20` in `array2` is not added to anything in `array1` because the corresponding list is empty.

For broadcasting to be successful, the arrays need to have the same length in all dimensions except the one being broadcasted; `array1` and `array2` both had to be length 3 in the example above. That's why this example fails:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [raises-exception]
---
array1 = ak.Array([[1, 2, 3], [4, 5]])
array2 = ak.Array([10, 20, 30])

array1 + array2
```

The same applies to functions of multiple arguments that aren't associated with any binary operator:

```{code-cell} ipython3
array1 = ak.Array([[True, False, True], [], [False, True]])
array2 = ak.Array([True, True, False])

np.logical_and(array1, array2)
```

And functions that aren't universal functions (ufuncs):

```{code-cell} ipython3
array1 = ak.Array([[1, 2, 3], [], [4, 5]])
array2 = ak.Array([10, 20, 30])

np.where(array1 % 2 == 0, array1, array2)
```

## Using `ak.broadcast_arrays`

Sometimes, you may want to broadcast arrays to a common shape without performing an additional operation. The {func}`ak.broadcast_arrays` function allows you to do this:

```{code-cell} ipython3
array1 = ak.Array([[1, 2, 3], [], [4, 5]])
array2 = ak.Array([10, 20, 30])

ak.broadcast_arrays(array1, array2)
```

This code would align `array1` and `array2` into compatible shapes that can be used in subsequent operations, effectively showing how each element corresponds between the two original arrays.

+++

## Missing data, heterogeneous data, and records

One of the ways Awkward Arrays extend beyond NumPy is by allowing the use of `None` for missing data. These `None` values are broadcasted like empty lists:

```{code-cell} ipython3
array1 = ak.Array([[1, 2, 3], None, [4, 5]])
array2 = ak.Array([10, 20, 30])

array1 + array2
```

Another difference from NumPy is that Awkward Arrays can contain data of mixed type, such as different numbers of dimensions. If numerical values _can_ be matched across such arrays, they are:

```{code-cell} ipython3
array1 = ak.Array([[1, 2, 3], 4, 5])
array2 = ak.Array([10, 20, 30])

array1 + array2
```

Arrays containing records can also be broadcasted, though most mathematical operations cannot be applied to records. Here is an example using {func}`ak.broadcast_arrays`.

```{code-cell} ipython3
array1 = ak.Array([
    [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}],
    [],
    [{"x": 4.4, "y": [1, 2, 3, 4]}, {"x": 5.5, "y": [1, 2, 3, 4, 5]}],
])
array2 = ak.Array([10, 20, 30])

ak.broadcast_arrays(array1, array2)
```

## Differences from NumPy broadcasting

Awkward Array broadcasting is identical to NumPy broadcasting in three respects:

1. arrays with the same number of dimensions must match lengths (except for length 1) exactly,
2. length-1 dimensions expand like scalars (one to many),
3. for arrays with different numbers of dimensions, the smaller number of dimensions is expanded to match the largest number of dimensions.

Awkward Arrays with fixed-length dimensions—not "variable-length" or "ragged"—broadcast exactly like NumPy.

Awkward Arrays with ragged dimensions expand the smaller number of dimensions on the left, whereas NumPy and Awkward-with-fixed-length expand the smaller number of dimensions on the right, when implementing point 3 above. This is the only difference.

Here's a demonstration of NumPy broadcasting:

```{code-cell} ipython3
x = np.array([
    [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
])
y = np.array([
    [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]],
    [[100, 200, 300, 400], [500, 600, 700, 800], [900, 1000, 1100, 1200]],
])

x + y
```

And fixed-length Awkward Arrays made from these can be broadcasted the same way:

```{code-cell} ipython3
ak.Array(x) + ak.Array(y)
```

but only because the latter have completely regular dimensions, like their NumPy counterparts.

```{code-cell} ipython3
print(x.shape)
print(y.shape)
```

```{code-cell} ipython3
print(ak.Array(x).type)
print(ak.Array(y).type)
```

In both NumPy and Awkward Array, `x` has fewer dimensions than `y`, so `x` is expanded on the _left_ from length-1 to length-2.

However, if the Awkward Array has variable-length _type_, regardless of whether the actual lists have variable lengths,

```{code-cell} ipython3
print(ak.Array(x.tolist()).type)
print(ak.Array(y.tolist()).type)
```

this broadcasting does not work:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [raises-exception]
---
ak.Array(x.tolist()) + ak.Array(y.tolist())
```

Instead of trying to add a dimension to the left of `x`'s shape, `(3, 4)`, to make `(2, 3, 4)`, the ragged broadcasting is trying to add a dimension to the right of `x`'s shape, and it doesn't line up.

### Why does ragged broadcasting have to be different?

Instead of adding a new dimension on the left, as NumPy and fixed-length Awkward Arrays do, ragged broadcasting tries to add a new dimension on the right in order to make it useful for emulating imperative code like

```{code-cell} ipython3
for x_i, y_i in zip(x, y):
    for x_ij, y_ij in zip(x_i, y_i):
        print("[", end=" ")
        for y_ijk in y_ij:
            print(x_ij + y_ijk, end=" ")
        print("]")
    print()
```

In the above, the value of `x_ij` is not varying while `y_ijk` varies in the innermost for-loop. In imperative code like this, it's natural for the outermost (left-most) dimensions of two nested lists to line up, while a scalar from the list with fewer dimensions, `x`, stays constant (is effectively duplicated) for each innermost `y` value.

This is _not_ what NumPy's left-broadcasting does:

```{code-cell} ipython3
x + y
```

Notice that the numerical values are different!

To get the behavior we expect from imperative code, we need to right-broadcast, which is what ragged broadcasting in Awkward Array does:

```{code-cell} ipython3
x = ak.Array([
    [1.1, 2.2, 3.3],
    [],
    [4.4, 5.5]
])
y = ak.Array([
    [[1], [1, 2], [1, 2, 3]],
    [],
    [[1, 2, 3, 4], [1, 2, 3, 4, 5]]
])

for x_i, y_i in zip(x, y):
    print("[")
    for x_ij, y_ij in zip(x_i, y_i):
        print("    [", end=" ")
        for y_ijk in y_ij:
            print(x_ij + y_ijk, end=" ")
        print("]")
    print("]\n")

x + y
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

In summary,

* NumPy left-broadcasts,
* Awkward Arrays with fixed-length lists left-broadcast, for consistency with NumPy,
* Awkward Arrays with variable-length lists right-broadcast, for consistency with imperative code.

One way to control this is to ensure that all arrays involved in an expression have the same number of dimensions by explicitly expanding them. Implicit broadcasting only happens for arrays of different numbers of dimensions, or if the length of a dimension is 1.

But it might also be the case that your arrays have lists of equal length, so they seem to be regular like a NumPy array, yet their data type says that the lists _can_ be variable-length. Perhaps you got the NumPy-like data from a source that doesn't enforce fixed lengths, such as Python lists ({func}`ak.from_iter`), JSON ({func}`ak.from_json`), or Parquet ({func}`ak.from_parquet`). Check the array's {func}`ak.type` to see whether all dimensions are ragged (`var *`) or regular (some number `*`).

The {func}`ak.from_regular` and {func}`ak.to_regular` functions toggle ragged (`var *`) and regular (some number `*`) dimensions, and {func}`ak.enforce_type` can be used to cast types like this in general.
