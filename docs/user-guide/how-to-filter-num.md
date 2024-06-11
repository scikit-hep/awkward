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

How to filter arrays by number of items
=======================================

```{code-cell} ipython3
import awkward as ak
```

In general, arrays are filtered using NumPy-like slicing. Numerical values can be filtered by numerical expressions in a way that is very similar to NumPy:

```{code-cell} ipython3
array = ak.Array([
    [[0, 1.1, 2.2], []], [[3.3, 4.4]], [], [[5.5], [6.6, 7.7, 8.8, 9.9]]
])
```

```{code-cell} ipython3
array[array > 4]
```

but it's also common to want to filter arrays by the number of items in each list, for two reasons:

* to exclude empty lists so that subsequent slices can select the item at index `0`,
* to make the list lengths rectangular for computational steps that require rectangular array (such as most forms of machine learning).

There are two functions that provide the lengths of lists: {func}`ak.num` and {func}`ak.count`. To filter arrays, you'll most likely want {func}`ak.num`.

## Use `ak.num`

{func}`ak.num` can be applied at any `axis`, and it returns the number of items in lists at that `axis` with the same shape for all levels above that `axis`.

```{code-cell} ipython3
ak.num(array, axis=0)
```

```{code-cell} ipython3
ak.num(array, axis=1)   # default
```

```{code-cell} ipython3
ak.num(array, axis=2)
```

Thus, if you want to select outer lists of `array` with length 2, you would use `axis=1`:

```{code-cell} ipython3
array[ak.num(array) == 2]
```

And if you want to select inner lists of `array` with length greater than 2, you would use `axis=2`:

```{code-cell} ipython3
array[ak.num(array, axis=2) > 2]
```

The ragged array of booleans that you get from comparing {func}`ak.num` with a number is exactly what is needed to slice the array.

## Don't use `ak.count`

By contrast, {func}`ak.count` returns structures that you can't use this way (for all but `axis=-1`):

```{code-cell} ipython3
ak.count(array, axis=None)   # default
```

```{code-cell} ipython3
ak.count(array, axis=0)
```

```{code-cell} ipython3
ak.count(array, axis=1)
```

```{code-cell} ipython3
ak.count(array, axis=2)   # equivalent to axis=-1 for this array
```

Also, {func}`ak.num` can be used on arrays that contain records, whereas {func}`ak.count` (like other reducers), can't.

As a reducer, {func}`ak.count` is intended to be used in a mathematical formula with other reducers, like {func}`ak.sum`, {func}`ak.max`, etc. (usually as a denominator). Its `axis` behavior matches that of other reducers, which is important for the shapes of nested lists to align.
