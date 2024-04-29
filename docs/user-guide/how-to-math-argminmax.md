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

Min/max/sort one array by another
=================================

A common task in data analysis is to select items from one array that minimizes or maximizes another, or to sort one array by the values of another.

```{code-cell} ipython3
import awkward as ak
```

## Naive attempt goes wrong

For instance, in

```{code-cell} ipython3
data = ak.Array([
    [
        {"title": "zero", "x": 0, "y": 0},
        {"title": "two", "x": 2, "y": 2.2},
        {"title": "one", "x": 1, "y": 1.1},
    ],
    [],
    [
        {"title": "four", "x": 4, "y": 4.4},
        {"title": "three", "x": 3, "y": 3.3},
    ],
    [
        {"title": "five", "x": 5, "y": 5.5},
    ],
    [
        {"title": "eight", "x": 8, "y": 8.8},
        {"title": "six", "x": 6, "y": 6.6},
        {"title": "nine", "x": 9, "y": 9.9},
        {"title": "seven", "x": 7, "y": 7.7},
    ],
])
```

you may want to score each record with a computed value, such as `x**2 + y**2`, and then select the record with the highest score from each list.

```{code-cell} ipython3
score = data.x**2 + data.y**2
score
```

At first, it would seem that {func}`ak.argmax` is what you need to identify the item with the highest score from each list and select it from `data`.

```{code-cell} ipython3
best_index = ak.argmax(score, axis=1)
best_index
```

However, if you attempt to slice the `data` with this, you'll either get an indexing error or lists instead of records:

```{code-cell} ipython3
data[best_index]
```

## What happend?

Following the logic for {doc}`reducers <how-to-math-reducing>`, the {func}`ak.argmin` function returns an array with one fewer dimension than the input: the `data` is an array of lists of records, but `best_index` is an array of integers. We want an array of lists of integers.

The `keepdims=True` parameter can ensure that the output has the same number of dimensions as the input:

```{code-cell} ipython3
best_index = ak.argmax(score, axis=1, keepdims=True)
best_index
```

Now these integers are at the same level of depth as the records that we want to select:

```{code-cell} ipython3
result = data[best_index]
result
```

In the above, each length-1 list contains the record with the highest `score`. Even the empty list, for which the {func}`ak.argmax` is missing (`None`), is now a length-1 list containing `None`. We can remove this length-1 list structure with a slice:

```{code-cell} ipython3
result[:, 0]
```

To summarize this as a handy idiom, the way to get the record with maximum `data.x**2 + data.y**2` from an array of lists of records named `data` is

```{code-cell} ipython3
data[ak.argmax(data.x**2 + data.y**2, axis=1, keepdims=True)][:, 0]
```

For an array of lists of lists of records, `axis=2` and the final slice would be `[:, :, 0]`, and so on.

## Sorting by another array

In addition to selecting items corresponding to the minimum or maximum of some other array, we may want to sort by another array. Just as {func}`ak.argmin` and {func}`ak.argmax` are the functions that would convey indexes from one array to another, {func}`ak.argsort` conveys sorted indexes from one array to another array. However, {func}`ak.argsort` always maintains the total number of dimensions, so we don't need to worry about `keepdims`.

```{code-cell} ipython3
sorted_indexes = ak.argsort(score)
sorted_indexes
```

```{code-cell} ipython3
data[sorted_indexes]
```

This sorted data has the same type as `data`:

```{code-cell} ipython3
data.type.show()
```

It's exactly what we want. {func}`ak.argsort` is easier to use than {func}`ak.argmin` and {func}`ak.argmax`.

## Getting the top _n_ items

The {func}`ak.min`, {func}`ak.max`, {func}`ak.argmin`, and {func}`ak.argmax` functions select one extreme value. If you want the top _n_ items (with _n ≠ 1_), you can use {func}`ak.sort` or {func}`ak.argsort`, followed by a slice:

```{code-cell} ipython3
top2 = data[ak.argsort(score)][:, :2]
top2
```

Notice, though, that not all of these lists have length 2. The lists with 0 or 1 input items have 0 or 1 output items: these lists have _up to_ length 2. That may be fine, but the example with {func}`ak.argmax`, above, resulted in `None` for an empty list. We could emulate that with {func}`ak.pad_none`.

```{code-cell} ipython3
padded = ak.pad_none(top2, 2, axis=1)
padded
```

The data type still says "`var *`", meaning that the lists are allowed to be variable-length, even though they happen to all have length 2. At this point, we might not care because that's all we need in order to convert these fields into NumPy arrays (e.g. for some machine learning process):

```{code-cell} ipython3
ak.to_numpy(padded.x)
```

```{code-cell} ipython3
ak.to_numpy(padded.y)
```

Or we might want to force the data type to ensure that the lists have length 2, using {func}`ak.to_regular`, {func}`ak.enforce_type`, or just by passing `clip=True` in the original {func}`ak.pad_none`.

```{code-cell} ipython3
ak.to_regular(padded, axis=1)
```

(Now the list lengths are "`2 *`", rather than "`var *`".)
