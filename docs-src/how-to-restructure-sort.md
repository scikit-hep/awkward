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

How to sort arrays and inner arrays
===================================

Sorting puts Awkward Array elements in an ordered sequence: numeric or lexicographical, ascending or descending.

By default, `ak.sort` sorts the elements in non-descending order. The order of equivalent elements is guaranteed to be preserved.

The back-end is an `std::stable_sort` with complexity `O(N·log(N)2)`, where `N = std::distance(first, last)` applications of a comparator. If an additional memory is available, then the complexity is `O(N·log(N))`.

```{code-cell} ipython3
import awkward as ak
ak.sort(ak.Array([[7, 5, 7], [], [2], [8, 2]]))
```

Sorting in axis
---------------

An optional `axis` parameter defines an axis along which to sort. The default is `axis=-1`, which is the last axis.

Awkward Array axes are defined similar to Numpy axes. In a two-dimentional Awkward Array they are the directions along the rows and columns.

[![](img/axis.jpg)]

In Akward Array, axis 0 is the "first" axis.
