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

If you have a collection of Python lists, the easiest way to turn them into an Awkward Array is to pass them to the [ak.Array](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Array.html) constructor, which recognizes a non-dict, non-NumPy iterable and calls [ak.from_iter](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_iter.html).

```{code-cell} ipython3
import awkward as ak
import numpy as np
```

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

Some operations are possible with union arrays, but not all. (Iteration in Numba is one such example.)

+++

**TODO!** Should eventually cover [ak.unflatten](https://awkward-array.readthedocs.io/en/latest/_auto/ak.unflatten.html).
