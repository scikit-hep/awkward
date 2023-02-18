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

How to filter with ragged arrays
================================

```{code-cell} ipython3
import awkward as ak
import numpy as np
```

## What is awkward indexing?

One of the most powerful features of NumPy is the expressiveness of its indexing system. A NumPy array [can be sliced in many different ways](https://numpy.org/doc/stable/user/basics.indexing.html#basic-indexing), such as with a single integer, or an array of integers. Awkward Array implements most of these indexing styles, but adds an additional variant: _awkward indexing_.

+++

Consider the following ragged array:

```{code-cell} ipython3
array = ak.Array(
    [
        [
            [0.0, 1.1, 2.2],
            [3.3, 4.4, 5.5, 6.6],
            [7.7],
        ],
        [],
        [
            [8.8, 9.9, 10.10, 11.11, 12.12],
        ],
    ]
)
array
```

We can easily pull out the first two items with a simple slice

```{code-cell} ipython3
array[..., :2]
```

But what if we wanted to pull out a different number of items for each sublist, e.g. to produce the following array:
```
[[[], [3.3], [7.7]],
 [],
 [[10.10, 11.11, 12.12]]]
----------------------------------------------
type: 3 * var * var * float64
```

+++

To produce this result, we need awkward indexing.

+++

(how-to-filter-masked:building-an-awkward-index)=
## Building an awkward index

+++

Awkward indexing requires an index array that
1. has a structure matching the array being sliced **up to** (but not including) the final dimension of the index
2. has at _least_ one ragged (`var`) dimension **or** contain missing values

By structure, we mean the number of sublists in each dimension, which can be seen with {func}`ak.num`:

+++

`axis=0` has a single list of three items:

```{code-cell} ipython3
ak.num(array, axis=0)
```

`axis=1` has three lists, the first with three items, the second with zero items, the third with a single item:

```{code-cell} ipython3
ak.num(array, axis=1)
```

To put this more simply, the final dimension of the awkward index is used to pull items out of the array. Therefore, Awkward needs the preceeding dimensions to line up!

+++

Recall that we wanted to pull out the following result from `array` using awkward indexing:
```
[[[], [3.3], [7.7]],
 [],
 [[10.10, 11.11, 12.12]]]
----------------------------------------------
type: 3 * var * var * float64
```

+++

It's clear that we want to pull specific items out of the _final_ dimension of the array. Let's find out where these particular items are located in their sublists. Awkward Array provides a special function {func}`ak.local_index` to find the index of each item in the array

```{code-cell} ipython3
ak.local_index(["x", "y", "z"])
```

The word "local" refers to the way that {func}`ak.local_index` computes the index of each item relative to the sublist in which it is found. e.g. for a two-dimensional array:

```{code-cell} ipython3
ak.local_index(
    [
        ["up", "charm", "top"],
        ["down", "strange"],
        ["bottom"],
    ]
)
```

 {func}`ak.local_index` also takes an `axis` parameter, but here we only need the default `axis=-1`. It can be seen that this local index has exactly the same _structure_ as `array`.

```{code-cell} ipython3
array
```

```{code-cell} ipython3
ak.local_index(array)
```

To create our awkward index, all we need to do is create an array _like_ `ak.local_index(array)`, but with only the local indices that we want to keep, i.e.

```{code-cell} ipython3
index = ak.Array(
    [
        [[], [0], [0]],
        [],
        [[2, 3, 4]],
    ]
)
```

We can see that this array matches the leading structure of `array`, and has at least one `var` dimension

```{code-cell} ipython3
index.type.show()
```

Let's see what slicing `array` with this awkward index looks like:

```{code-cell} ipython3
array[index]
```

Clearly this index produces the result that we were aiming for!

+++

(how-to-filter-ragged:indexing-with-argmin-and-argmax)=
## Indexing with `argmin` and `argmax`

+++

Awkward indexing is especially useful when combined with the positional {func}`ak.argmin` and {func}`ak.argmax` reducers. These functions accept an `keepdims=True` argument that can be used to keep _the same number of dimensions_ as the original array. There is also a `mask_identity` argument is explained in {ref}`how-to-filter-ragged:indexing-with-missing-values`. For now, we will set it to `False`.

```{code-cell} ipython3
array = ak.Array(
    [
        [10, 3, 2, 9],
        [4, 5, 5, 12, 6],
        [8, 9, -1],
    ]
)
array
```

With `keepdims=False`, all reducers collapse a dimension of the original array:

```{code-cell} ipython3
ak.argmin(array, axis=1, keepdims=False, mask_identity=False)
```

If we try and use this index to slice `array`, it will likely not produce the result we might initially expect:

```{code-cell} ipython3
array[ak.argmin(array, axis=1, keepdims=False, mask_identity=False)]
```

Instead of pulling out the smallest items in `array` along `axis=1`, we have simply re-arranged the sublists of `array` along `axis=0`. Our index has only a single dimension, so for each value in `ak.argmin(array, axis=-1)`, Awkward pulls out the corresponding item from `array`. We want to pull values out of the _second_ dimension, so our index array needs to be two dimensional.

+++

Let's now look at what happens with `keepdims=True`. The result is a two dimensional, fully regular array, with no missing values:

```{code-cell} ipython3
ak.argmin(array, axis=-1, keepdims=True, mask_identity=False)
```

Before we can use this as an index array, we need to convert _at least_ one dimension to a ragged dimension. This follows from rule (2) described in {ref}`how-to-filter-masked:building-an-awkward-index`.

```{code-cell} ipython3
ak.from_regular(
    ak.argmin(array, axis=-1, keepdims=True, mask_identity=False)
)
```

We can now use this array to index into `array`:

```{code-cell} ipython3
array[
    ak.from_regular(
        ak.argmin(array, axis=-1, keepdims=True, mask_identity=False)
    )
]
```

it produces the expected result!

+++

## Filtering with booleans
As described in {ref}`how-to-filter-masked:building-an-awkward-index`, Awkward Array's awkward indexing is a generalisation of the advanced indexing supported by NumPy. It is therefore reasonable to ask whether Awkward supports awkward indexing with 
_boolean_ values, selecting only values for which the index is `True`. 

Let's create an array of integers:

```{code-cell} ipython3
numbers = ak.Array(
    [
        [0, 1, 2, 3],
        [4, 5, 6],
        [8, 9, 10, 11, 12],
    ]
)
```

We can use awkward indexing to keep only the even values. Let's generate a boolean mask with the same structure as `numbers`. In order for there to be a single boolean value for each item in `numbers`, the filter array must have exactly the same number of elements. Ufuncs, such as {func}`np.mod`, are powerful tools for generating boolean masks, as they directly preserve the exact structure of the original array:

```{code-cell} ipython3
is_even = (numbers % 2) == 0
is_even
```

```{code-cell} ipython3
numbers
```

Now we can use `is_even` to slice `numbers`:

```{code-cell} ipython3
numbers[is_even]
```

Note that this is different to what would happen with NumPy's boolean indexing:

```{code-cell} ipython3
numbers_np = np.array(
    [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ]
)
```

```{code-cell} ipython3
numbers_np[(numbers_np % 2) == 0]
```

NumPy, lacking a ragged array structure, has to flatten the result whereas Awkward Array preserves the number of dimensions in the result.

```{code-cell} ipython3
numbers[
    [[True, False, True, False],
     [False],
     [False, True, False]]
]
```
