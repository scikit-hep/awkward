---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

How to pad and clip arrays, particularly for machine learning
=============================================================

Most applications of arrays expect them to be rectilinear—a rectangular table of numbers in _N_ dimensions. Machine learning frameworks refer to these blocks of numbers as "tensors," but they are equivalent to _N_-dimensional NumPy arrays. Awkward Array handles more general data than these, but if your intention is to pass the data to a framework that wants a tensor, you have to reduce your data to a tensor.

This tutorial presents several ways of doing that. Like [flattening for plots](how-to-restructure-flatten), the method you choose will depend on what your data mean and what you want to get out of the machine learning process. For instance, if you remove all list structures with {func}`ak.flatten`, you can't expect the machine learning algorithm to learn about lists (whatever they mean in your application), and if you truncate them at a particular length, it won't learn about the values that have been removed.

(In principle, graph neural networks should be able to learn about variable-length data without any losses, but I'm not familiar enough with how to set up these processes to explain it. If you're an expert, we'd like to hear tips and tricks in [GitHub Discussions](https://github.com/scikit-hep/awkward-1.0/discussions)!)

```{code-cell} ipython3
import awkward as ak
import numpy as np
```

Flattening an array
-------------------

Suppose you have an array like this:

```{code-cell} ipython3
array = ak.Array([[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]])
array
```

If the list boundaries are irrelevant to your machine learning application, you can simply {func}`ak.flatten` them.

```{code-cell} ipython3
ak.flatten(array)
```

The default `axis` for {func}`ak.flatten` is `1`, which is to say, the first level of nested list (`axis=1`) gets squashed to the outer level of array nesting (`axis=0`). If you have many levels to flatten at once, you can use `axis=None`:

```{code-cell} ipython3
ak.flatten(ak.Array([[[[[[1.1, 2.2, 3.3]]], [[[4.4, 5.5]]]]]]), axis=None)
```

However, be aware that {func}`ak.flatten` with `axis=None` will also merge all fields of a record, which is usually undesirable, and the order might not be what you expect.

```{code-cell} ipython3
ak.flatten(
    ak.Array([{"x": 1.1, "y": 10}, {"x": 2.2, "y": 20}, {"x": 3.3, "y": 30}]), axis=None
)
```

Also be aware that flattening (for any `axis`) removes missing values (at that `axis`). That is, at the level where lists are concatenated, missing lists are treated the same way as empty lists.

```{code-cell} ipython3
ak.flatten(ak.Array([[1.1, 2.2, 3.3], None, [4.4, 5.5]]))
```

But only at that level.

```{code-cell} ipython3
ak.flatten(ak.Array([[1.1, 2.2, None, 3.3], [], [4.4, 5.5]]))
```

Markers at the end of each list
-------------------------------

Flattening is likely to be useful for training recurrent neural networks, which learn a sequence in order. If, for instance, the values in your nested lists represent words and the lists are sentences, the machine would learn what typical sentences look like. However, it would not learn that the ends of sentences are special.

Suppose we have a nested array of integers representing words in a corpus like this:

```{code-cell} ipython3
array = ak.Array([[5512, 1364], [657], [4853, 6421, 3461, 7745], [5245, 654, 4216]])
array
```

Flattening it would turn them into a big run-on sentence, which may be a bad thing to learn.

```{code-cell} ipython3
ak.flatten(array)
```

Suppose that we want to fix this by adding `0` as a marker meaning "end of sentence/stop." One way to do it is to make an array of `[0]` lists with the same length as `array` and {func}`ak.concatenate` them at `axis=1`.

```{code-cell} ipython3
periods = np.zeros((len(array), 1), np.int64)
periods
```

```{code-cell} ipython3
ak.concatenate([array, periods], axis=1)
```

```{code-cell} ipython3
ak.concatenate([array, periods], axis=1).to_list()
```

```{code-cell} ipython3
ak.flatten(ak.concatenate([array, periods], axis=1))
```

Padding lists to a common length
--------------------------------

A general function for turning an array of lists into lists of the same length is {func}`ak.pad_none`. With the default `clip=False`, it ensures that a set of lists have _at least_ a given target length.

```{code-cell} ipython3
array = ak.Array([[0, 1, 2], [], [3, 4], [5], [6, 7, 8, 9]])
array
```

```{code-cell} ipython3
ak.pad_none(array, 2).to_list()
```

"At least length 2" means that the list is still a variable-length type, which we can see with the "`var *`" in its type string.

```{code-cell} ipython3
ak.pad_none(array, 2).type
```

To produce lists of an exact length, set `clip=True`.

```{code-cell} ipython3
ak.pad_none(array, 2, clip=True).to_list()
```

```{code-cell} ipython3
ak.pad_none(array, 2, clip=True).type
```

Now the type string says that the nested lists all have exactly two elements each. This can be directly converted into NumPy (allowing for missing data with {func}`ak.to_numpy`; casting as `np.asarray` doesn't allow the arrays to be [NumPy masked arrays](https://numpy.org/doc/stable/reference/maskedarray.html)).

```{code-cell} ipython3
ak.to_numpy(ak.pad_none(array, 2, clip=True))
```

Perhaps your machine learning library knows how to deal with NumPy masked arrays. If it does not, you can replace all of the missing values with {func}`ak.fill_none`.

{func}`ak.pad_none` and {func}`ak.fill_none` are frequently used together.

```{code-cell} ipython3
ak.fill_none(ak.pad_none(array, 2, clip=True), 999)
```

```{code-cell} ipython3
np.asarray(ak.fill_none(ak.pad_none(array, 2, clip=True), 999))
```

Record fields into lists
------------------------

Sometimes, the data you need to put into one big array (tensor) for machine learning is scattered among several record fields. In Awkward Array, record fields are discontiguous (are stored in separate arrays) and nested lists are contiguous (same array). This will require a copy using {func}`ak.concatenate`.

```{code-cell} ipython3
array = ak.Array(
    [
        {"a": 11, "b": 12, "c": 13, "d": 14, "e": 15, "f": 16, "g": 17, "h": 18},
        {"a": 21, "b": 22, "c": 23, "d": 24, "e": 25, "f": 26, "g": 27, "h": 28},
        {"a": 31, "b": 32, "c": 33, "d": 34, "e": 35, "f": 36, "g": 37, "h": 38},
        {"a": 41, "b": 42, "c": 43, "d": 44, "e": 45, "f": 46, "g": 47, "h": 48},
        {"a": 51, "b": 52, "c": 53, "d": 54, "e": 55, "f": 56, "g": 57, "h": 58},
    ]
)
array
```

To concatenate, say, `array.a` and `array.b` as though `[a, b]` were a list, we have to put them into lists (of length 1). NumPy's `np.newaxis` slice will do that.

```{code-cell} ipython3
array.a[:, np.newaxis], array.b[:, np.newaxis]
```

```{code-cell} ipython3
ak.concatenate([array.a[:, np.newaxis], array.b[:, np.newaxis]], axis=1)
```

If there are a lot of fields, doing this manually for each one would be a chore, so we use {func}`ak.fields` and {func}`ak.unzip`.

```{code-cell} ipython3
ak.fields(array)
```

```{code-cell} ipython3
ak.unzip(array)
```

Now it's a one-liner.

```{code-cell} ipython3
ak.concatenate(ak.unzip(array[:, np.newaxis]), axis=1)
```

```{code-cell} ipython3
np.asarray(ak.concatenate(ak.unzip(array[:, np.newaxis]), axis=1))
```
