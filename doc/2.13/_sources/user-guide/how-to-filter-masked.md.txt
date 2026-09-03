---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

How to filter with arrays containing missing values
===================================================

```{code-cell} ipython3
import awkward as ak
import numpy as np
```

(how-to-filter-ragged:indexing-with-missing-values)=
## Indexing with missing values
In {ref}`how-to-filter-masked:building-an-awkward-index`, we looked building arrays of integers to perform awkward indexing using {func}`ak.argmin` and {func}`ak.argmax`. In particular, the `keepdims` argument of {func}`ak.argmin` and {func}`ak.argmax` is very useful for creating arrays that can be used to index into the original array. However, reducers such as {func}`ak.argmax` behave differently when they are asked to operate upon empty lists. 

Let's first create an array that contains empty sublists:

```{code-cell} ipython3
array = ak.Array(
    [
        [],
        [10, 3, 2, 9],
        [4, 5, 5, 12, 6],
        [],
        [8, 9, -1],
    ]
)
array
```

Awkward reducers accept a `mask_identity` argument, which changes the {attr}`ak.Array.type` and the values of the result:

```{code-cell} ipython3
ak.argmax(array, keepdims=True, axis=-1, mask_identity=False)
```

```{code-cell} ipython3
ak.argmax(array, keepdims=True, axis=-1, mask_identity=True)
```

Setting `mask_identity=True` yields the identity value for the reducer instead of `None` when reducing empty lists. From the above examples of {func}`ak.argmax`, we can see that the identity for the {func}`ak.argmax` is `-1`: What happens if we try and use the array produced with `mask_identity=False` to index into `array`?

+++

As discussed in {ref}`how-to-filter-ragged:indexing-with-argmin-and-argmax`, we first need to convert _at least_ one dimension to a ragged dimension

```{code-cell} ipython3
index = ak.from_regular(
    ak.argmax(array, keepdims=True, axis=-1, mask_identity=False)
)
```

Now, if we try and index into `array` with `index`, it will raise an exception

```{code-cell} ipython3
:tags: [raises-exception]

array[index]
```

From the error message, it is clear that for some sublist(s) the index `-1` is out of range. This makes sense; some of our sublists are empty, meaning that there is no valid integer to index into them. 

Now let's look at the result of indexing with `mask_identity=True`. 

```{code-cell} ipython3
index = ak.argmax(array, keepdims=True, axis=-1, mask_identity=True)
```

Because it contains an option type, `index` already satisfies rule (2) in {ref}`how-to-filter-masked:building-an-awkward-index`, and we do not need to convert it to a ragged array. We can see that this index succeeds:

```{code-cell} ipython3
array[index]
```

Here, the missing values in the index array correspond to missing values _in the output array_.

+++

## Indexing with missing sublists

Ragged indexing also supports using `None` in place of _empty sublists_ within an index. For example, given the following array

```{code-cell} ipython3
array = ak.Array(
    [
        [10, 3, 2, 9],
        [4, 5, 5, 12, 6],
        [],
        [8, 9, -1],
    ]
)
array
```

let's use build a ragged index to pull out some particular values. Rather than using empty lists, we can use `None` to mask out sublists that we don't care about:

```{code-cell} ipython3
array[
    [
        [0, 1],
        None,
        [],
        [2],
    ],
]
```

If we compare this with simply providing an empty sublist,

```{code-cell} ipython3
array[
    [
        [0, 1],
        [],
        [],
        [2],
    ],
]
```

we can see that the `None` value introduces an option-type into the final result. `None` values can be used at _any_ level in the index array to introduce an option-type at that depth in the result.
