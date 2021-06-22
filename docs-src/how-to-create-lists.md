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

How to create arrays of lists
=============================

```{code-cell} ipython3
import awkward as ak
import numpy as np
```

From Python lists
-----------------

If you have a collection of Python lists, the easiest way to turn them into an Awkward Array is to pass them to the [ak.Array](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Array.html) constructor, which recognizes a non-dict, non-NumPy iterable and calls [ak.from_iter](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_iter.html).

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

The [ak.Array](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Array.html) constructor loads NumPy arrays differently from Python lists. The inner dimensions of a NumPy array are guaranteed to have the same lengths, so they are interpreted as a fixed-length list type.

```{code-cell} ipython3
numpy_array = np.arange(2*3*5).reshape(2, 3, 5)
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
numpy_arrays = [np.arange(3*5).reshape(3, 5), np.arange(3*5, 2*3*5).reshape(3, 5)]
numpy_arrays
```

```{code-cell} ipython3
irregular_array = ak.Array(numpy_arrays)
irregular_array
```

Both `regular_array` and `irregular_array` have the same data values:

```{code-cell} ipython3
regular_array.tolist() == irregular_array.tolist()
```

but they have different types:

```{code-cell} ipython3
regular_array.type, irregular_array.type
```

This can make a difference in some operations, such as [broadcasting](how-to-math-broadcasting).

+++

If you want more control over this, use the explicit [ak.from_iter](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_iter.html) and [ak.from_numpy](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_numpy.html) functions instead of the general-purpose [ak.Array](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Array.html) constructor.

+++

Unflattening
------------

Another difference between [ak.from_iter](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_iter.html) and [ak.from_numpy](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_numpy.html) is that iteration over Python lists is slow and necessarily copies the data, whereas ingesting a NumPy array is zero-copy. (You can see that it's zero copy by [changing the data in-place](how-to-convert-numpy.html#mutability-of-awkward-arrays-from-numpy).)

In some cases, list-making can be vectorized. If you have a flat NumPy array of data and an array of "counts" that add up to the length of the data, then you can [ak.unflatten](https://awkward-array.readthedocs.io/en/latest/_auto/ak.unflatten.html) it.

```{code-cell} ipython3
data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
counts = np.array([3, 0, 1, 4])

unflattened = ak.unflatten(data, counts)
unflattened
```

The first list has length `3`, the second has length `0`, the third has length `1`, and the last has length `4`. This is close to Awkward Array's internal representation of variable-length lists, so it can be performed quickly.

+++

This function is named [ak.unflatten](https://awkward-array.readthedocs.io/en/latest/_auto/ak.unflatten.html) because it has the opposite effect as [ak.flatten](https://awkward-array.readthedocs.io/en/latest/_auto/ak.flatten.html) and [ak.num](https://awkward-array.readthedocs.io/en/latest/_auto/ak.num.html):

```{code-cell} ipython3
ak.flatten(unflattened)
```

```{code-cell} ipython3
ak.num(unflattened)
```
