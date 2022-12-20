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

How to create arrays of lists
=============================

```{code-cell} ipython3
import awkward as ak
import numpy as np
```

From Python lists
-----------------

If you have a collection of Python lists, the easiest way to turn them into an Awkward Array is to pass them to the {class}`ak.Array` constructor, which recognizes a non-dict, non-NumPy iterable and calls {func}`ak.from_iter`.

```{code-cell} ipython3
python_lists = [[1, 2, 3], [], [4, 5], [6], [7, 8, 9, 10]]
python_lists
```

```{code-cell} ipython3
awkward_array = ak.Array(python_lists)
awkward_array
```

The lists of lists can be arbitrarily deep.

```{code-cell} ipython3
python_lists = [[[[], [1, 2, 3]]], [[[4, 5]]], []]
python_lists
```

```{code-cell} ipython3
awkward_array = ak.Array(python_lists)
awkward_array
```

The "`var *`" in the type string indicates nested levels of variable-length lists. This is an array of lists of lists of lists of integers.

+++

The advantage of the Awkward Array is that the numerical data are now all in one array buffer and calculations are vectorized across the array, such as NumPy [universal functions](https://numpy.org/doc/stable/reference/ufuncs.html).

```{code-cell} ipython3
np.sqrt(awkward_array)
```

Unlike Python lists, arrays consist of a homogeneous type. A Python list wouldn't notice if numerical data were given at two different levels of nesting, but that's a big difference to an Awkward Array.

```{code-cell} ipython3
union_array = ak.Array([[[[], [1, 2, 3]]], [[4, 5]], []])
union_array
```

In this example, the data type is a "union" of two levels deep and three levels deep.

```{code-cell} ipython3
union_array.type
```

Some operations are possible with union arrays, but not all. ([Iteration in Numba](https://github.com/scikit-hep/awkward-1.0/issues/174) is one such example.)

+++

From NumPy arrays
-----------------

The {class}`ak.Array` constructor loads NumPy arrays differently from Python lists. The inner dimensions of a NumPy array are guaranteed to have the same lengths, so they are interpreted as a fixed-length list type.

```{code-cell} ipython3
numpy_array = np.arange(2 * 3 * 5).reshape(2, 3, 5)
numpy_array
```

```{code-cell} ipython3
regular_array = ak.Array(numpy_array)
regular_array
```

The type in this case has no "`var *`" in it, only "`2 *`", "`3 *`", and "`5 *`". It's a length-2 array of length-3 lists containing length-5 lists of integers.

+++

Furthermore, if NumPy arrays are _nested within_ Python lists (or other iterables), they'll be treated as variable-length ("`var *`") because there's no guarantee at the start of a sequence that all NumPy arrays in the sequence will have the same shape.

```{code-cell} ipython3
numpy_arrays = [
    np.arange(3 * 5).reshape(3, 5),
    np.arange(3 * 5, 2 * 3 * 5).reshape(3, 5),
]
numpy_arrays
```

```{code-cell} ipython3
irregular_array = ak.Array(numpy_arrays)
irregular_array
```

Both `regular_array` and `irregular_array` have the same data values:

```{code-cell} ipython3
regular_array.to_list() == irregular_array.to_list()
```

but they have different types:

```{code-cell} ipython3
regular_array.type, irregular_array.type
```

This can make a difference in some operations, such as [broadcasting](how-to-math-broadcasting).

+++

If you want more control over this, use the explicit {func}`ak.from_iter` and {func}`ak.from_numpy` functions instead of the general-purpose {class}`ak.Array` constructor.

+++

Unflattening
------------

Another difference between {func}`ak.from_iter` and {func}`ak.from_numpy` is that iteration over Python lists is slow and necessarily copies the data, whereas ingesting a NumPy array is zero-copy. (You can see that it's zero copy by [changing the data in-place](how-to-convert-numpy.html#mutability-of-awkward-arrays-from-numpy).)

In some cases, list-making can be vectorized. If you have a flat NumPy array of data and an array of "counts" that add up to the length of the data, then you can {func}`ak.unflatten` it.

```{code-cell} ipython3
data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
counts = np.array([3, 0, 1, 4])

unflattened = ak.unflatten(data, counts)
unflattened
```

The first list has length `3`, the second has length `0`, the third has length `1`, and the last has length `4`. This is close to Awkward Array's internal representation of variable-length lists, so it can be performed quickly.

+++

This function is named {func}`ak.unflatten` because it has the opposite effect as {func}`ak.flatten` and {func}`ak.num`:

```{code-cell} ipython3
ak.flatten(unflattened)
```

```{code-cell} ipython3
ak.num(unflattened)
```

With ArrayBuilder
-----------------

{class}`ak.ArrayBuilder` is described in more detail [in this tutorial](how-to-create-arraybuilder), but you can also construct arrays of lists using the `begin_list`/`end_list` methods or the `list` context manager.

(This is what {func}`ak.from_iter` uses internally to accumulate lists.)

```{code-cell} ipython3
builder = ak.ArrayBuilder()

builder.begin_list()
builder.append(1)
builder.append(2)
builder.append(3)
builder.end_list()

builder.begin_list()
builder.end_list()

builder.begin_list()
builder.append(4)
builder.append(5)
builder.end_list()

array = builder.snapshot()
array
```

```{code-cell} ipython3
builder = ak.ArrayBuilder()

with builder.list():
    builder.append(1)
    builder.append(2)
    builder.append(3)

with builder.list():
    pass

with builder.list():
    builder.append(4)
    builder.append(5)

array = builder.snapshot()
array
```

In Numba
--------

Functions that Numba Just-In-Time (JIT) compiles can use {class}`ak.ArrayBuilder` or construct flat data and "counts" arrays for {func}`ak.unflatten`.

([At this time](https://numba.pydata.org/numba-doc/dev/reference/pysupported.html#language), Numba can't use context managers, the `with` statement, in fully compiled code. {class}`ak.ArrayBuilder` can't be constructed or converted to an array using `snapshot` inside a JIT-compiled function, but can be outside the compiled context. Similarly, `ak.*` functions like {func}`ak.unflatten` can't be called inside a JIT-compiled function, but can be outside.)

```{code-cell} ipython3
import numba as nb
```

```{code-cell} ipython3
@nb.jit
def append_list(builder, start, stop):
    builder.begin_list()
    for x in range(start, stop):
        builder.append(x)
    builder.end_list()


@nb.jit
def example(builder):
    append_list(builder, 1, 4)
    append_list(builder, 999, 999)
    append_list(builder, 4, 6)
    return builder


builder = example(ak.ArrayBuilder())

array = builder.snapshot()
array
```

```{code-cell} ipython3
@nb.jit
def append_list(i, data, j, counts, start, stop):
    for x in range(start, stop):
        data[i] = x
        i += 1
    counts[j] = stop - start
    j += 1
    return i, j


@nb.jit
def example():
    data = np.empty(5, np.int64)
    counts = np.empty(3, np.int64)
    i, j = 0, 0
    i, j = append_list(i, data, j, counts, 1, 4)
    i, j = append_list(i, data, j, counts, 999, 999)
    i, j = append_list(i, data, j, counts, 4, 6)
    return data, counts


data, counts = example()

array = ak.unflatten(data, counts)
array
```
