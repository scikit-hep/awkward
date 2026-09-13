# How to filter with ragged arrays

```ipython3
import awkward as ak
import numpy as np
```

## What is awkward indexing?

One of the most powerful features of NumPy is the expressiveness of its indexing system. A NumPy array [can be sliced in many different ways](https://numpy.org/doc/stable/user/basics.indexing.html#basic-indexing), such as with a single integer, or an array of integers. Awkward Array implements most of these indexing styles, but adds an additional variant: *awkward indexing*.

Consider the following ragged array:

```ipython3
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

```ipython3
array[..., :2]
```

But what if we wanted to pull out a different number of items for each sublist, e.g. to produce the following array:

```default
[[[], [3.3], [7.7]],
 [],
 [[10.10, 11.11, 12.12]]]
----------------------------------------------
type: 3 * var * var * float64
```

To produce this result, we need awkward indexing.

<a id="how-to-filter-masked-building-an-awkward-index"></a>

## Building an awkward index

Awkward indexing requires an index array that

1. has a structure matching the array being sliced **up to** (but not including) the final dimension of the index
2. has at *least* one ragged (`var`) dimension **or** contain missing values

By structure, we mean the number of sublists in each dimension, which can be seen with [`ak.num()`](sphinx-llm:137e3487aaa0453fa61d9c86fc644b7b#ak.num):

`axis=0` has a single list of three items:

```ipython3
ak.num(array, axis=0)
```

`axis=1` has three lists, the first with three items, the second with zero items, the third with a single item:

```ipython3
ak.num(array, axis=1)
```

To put this more simply, the final dimension of the awkward index is used to pull items out of the array. Therefore, Awkward needs the preceeding dimensions to line up!

Recall that we wanted to pull out the following result from `array` using awkward indexing:

```default
[[[], [3.3], [7.7]],
 [],
 [[10.10, 11.11, 12.12]]]
----------------------------------------------
type: 3 * var * var * float64
```

It’s clear that we want to pull specific items out of the *final* dimension of the array. Let’s find out where these particular items are located in their sublists. Awkward Array provides a special function [`ak.local_index()`](sphinx-llm:5b4feaf8bda9476caeebad2d79cf1487#ak.local_index) to find the index of each item in the array

```ipython3
ak.local_index(["x", "y", "z"])
```

The word “local” refers to the way that [`ak.local_index()`](sphinx-llm:5b4feaf8bda9476caeebad2d79cf1487#ak.local_index) computes the index of each item relative to the sublist in which it is found. e.g. for a two-dimensional array:

```ipython3
ak.local_index(
    [
        ["up", "charm", "top"],
        ["down", "strange"],
        ["bottom"],
    ]
)
```

[`ak.local_index()`](sphinx-llm:5b4feaf8bda9476caeebad2d79cf1487#ak.local_index) also takes an `axis` parameter, but here we only need the default `axis=-1`. It can be seen that this local index has exactly the same *structure* as `array`.

```ipython3
array
```

```ipython3
ak.local_index(array)
```

To create our awkward index, all we need to do is create an array *like* `ak.local_index(array)`, but with only the local indices that we want to keep, i.e.

```ipython3
index = ak.Array(
    [
        [[], [0], [0]],
        [],
        [[2, 3, 4]],
    ]
)
```

We can see that this array matches the leading structure of `array`, and has at least one `var` dimension

```ipython3
index.type.show()
```

```myst-ansi
3 * var * var * int64
```

Let’s see what slicing `array` with this awkward index looks like:

```ipython3
array[index]
```

Clearly this index produces the result that we were aiming for!

<a id="how-to-filter-ragged-indexing-with-argmin-and-argmax"></a>

## Indexing with `argmin` and `argmax`

Awkward indexing is especially useful when combined with the positional [`ak.argmin()`](sphinx-llm:836050fe860646e49e64fda769d3ddaa#ak.argmin) and [`ak.argmax()`](sphinx-llm:5588cbb2c2804faf860257487414bbc0#ak.argmax) reducers. These functions accept an `keepdims=True` argument that can be used to keep *the same number of dimensions* as the original array. There is also a `mask_identity` argument is explained in [Indexing with missing values](sphinx-llm:4dd13e8e53de4bff849930c380345a08#how-to-filter-ragged-indexing-with-missing-values). For now, we will set it to `False`.

```ipython3
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

```ipython3
ak.argmin(array, axis=1, keepdims=False, mask_identity=False)
```

If we try and use this index to slice `array`, it will likely not produce the result we might initially expect:

```ipython3
array[ak.argmin(array, axis=1, keepdims=False, mask_identity=False)]
```

Instead of pulling out the smallest items in `array` along `axis=1`, we have simply re-arranged the sublists of `array` along `axis=0`. Our index has only a single dimension, so for each value in `ak.argmin(array, axis=-1)`, Awkward pulls out the corresponding item from `array`. We want to pull values out of the *second* dimension, so our index array needs to be two dimensional.

Let’s now look at what happens with `keepdims=True`. The result is a two dimensional, fully regular array, with no missing values:

```ipython3
ak.argmin(array, axis=-1, keepdims=True, mask_identity=False)
```

Before we can use this as an index array, we need to convert *at least* one dimension to a ragged dimension. This follows from rule (2) described in [Building an awkward index](sphinx-llm:c008389bc06241cb9bebf83d1cf6cc26).

```ipython3
ak.from_regular(
    ak.argmin(array, axis=-1, keepdims=True, mask_identity=False)
)
```

We can now use this array to index into `array`:

```ipython3
array[
    ak.from_regular(
        ak.argmin(array, axis=-1, keepdims=True, mask_identity=False)
    )
]
```

it produces the expected result!

## Filtering with booleans

As described in [Building an awkward index](sphinx-llm:c008389bc06241cb9bebf83d1cf6cc26), Awkward Array’s awkward indexing is a generalisation of the advanced indexing supported by NumPy. It is therefore reasonable to ask whether Awkward supports awkward indexing with
*boolean* values, selecting only values for which the index is `True`.

Let’s create an array of integers:

```ipython3
numbers = ak.Array(
    [
        [0, 1, 2, 3],
        [4, 5, 6],
        [8, 9, 10, 11, 12],
    ]
)
```

We can use awkward indexing to keep only the even values. Let’s generate a boolean mask with the same structure as `numbers`. In order for there to be a single boolean value for each item in `numbers`, the filter array must have exactly the same number of elements. Ufuncs, such as `np.mod()`, are powerful tools for generating boolean masks, as they directly preserve the exact structure of the original array:

```ipython3
is_even = (numbers % 2) == 0
is_even
```

```ipython3
numbers
```

Now we can use `is_even` to slice `numbers`:

```ipython3
numbers[is_even]
```

Note that this is different to what would happen with NumPy’s boolean indexing:

```ipython3
numbers_np = np.array(
    [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ]
)
```

```ipython3
numbers_np[(numbers_np % 2) == 0]
```

NumPy, lacking a ragged array structure, has to flatten the result whereas Awkward Array preserves the number of dimensions in the result.

```ipython3
numbers[
    [[True, False, True, False],
     [False],
     [False, True, False]]
]
```
