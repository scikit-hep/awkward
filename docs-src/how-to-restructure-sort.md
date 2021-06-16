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
ak.sort(array = ak.Array([[2, 5, 3], [], [None, 0], [1]]))
```

Sorting in axis
---------------

An optional `axis` parameter defines an axis along which to sort. The default is `axis=-1`, which is the last axis.

Awkward Array axes are defined similar to Numpy axes. Unlike other operations, sorting does not support `axis=None`.

[![](img/sorting-axis.svg)]

In Akward Array, `axis=0` is the "first" axis and `axis=-1` is the last axis.

```{code-cell} ipython3
import awkward as ak
ak.sort(array = ak.Array([[2, 5, 3], [], [None, 0], [1]]), axis=0)
```

and

```{code-cell} ipython3
import awkward as ak
ak.sort(array = ak.Array([[2, 5, 3], [], [None, 0], [1]]), axis=1)
```
An empty range is not sorted.

The results make a lot of sense when you understand how the Awkward Array axis work.

Sorting `None` values
---------------------

Sorting places `None` values at the end of an ordered sequence.

```{code-cell} ipython3
import awkward as ak
array = ak.Array([[None, None, 3, 2], [1]])
ak.sort(array)
```

`None` values are dropped when sorting a one dimentinal option type array:

```{code-cell} ipython3
import awkward as ak
array = ak.Array([None, None, 1, -1, 30])
ak.sort(array)
```

Sorting order
-------------

The default sorting order is ascending. To switch to a descending order, change the paramerer to `ascending=False`:

```{code-cell} ipython3
import awkward as ak
ak.sort(array = ak.Array([[2, 5, 3], [], [None, 0], [1]]), ascending=False)
```

Stable sort
-----------

By default, the order of equivalent elements is guaranteed to be preserved. To switch to a quick sort, set a parameter `stable=False`.
