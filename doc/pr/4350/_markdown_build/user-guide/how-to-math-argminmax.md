# Min/max/sort one array by another

A common task in data analysis is to select items from one array that minimizes or maximizes another, or to sort one array by the values of another.

```ipython3
import awkward as ak
```

## Naive attempt goes wrong

For instance, in

```ipython3
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

```ipython3
score = data.x**2 + data.y**2
score
```

At first, it would seem that [`ak.argmax()`](sphinx-llm:5588cbb2c2804faf860257487414bbc0#ak.argmax) is what you need to identify the item with the highest score from each list and select it from `data`.

```ipython3
best_index = ak.argmax(score, axis=1)
best_index
```

However, if you attempt to slice the `data` with this, you’ll either get an indexing error or lists instead of records:

```ipython3
data[best_index]
```

## What happend?

Following the logic for [reducers](sphinx-llm:be92daa510d04afcba9a0da0b0119a60), the [`ak.argmin()`](sphinx-llm:836050fe860646e49e64fda769d3ddaa#ak.argmin) function returns an array with one fewer dimension than the input: the `data` is an array of lists of records, but `best_index` is an array of integers. We want an array of lists of integers.

The `keepdims=True` parameter can ensure that the output has the same number of dimensions as the input:

```ipython3
best_index = ak.argmax(score, axis=1, keepdims=True)
best_index
```

Now these integers are at the same level of depth as the records that we want to select:

```ipython3
result = data[best_index]
result
```

In the above, each length-1 list contains the record with the highest `score`. Even the empty list, for which the [`ak.argmax()`](sphinx-llm:5588cbb2c2804faf860257487414bbc0#ak.argmax) is missing (`None`), is now a length-1 list containing `None`. We can remove this length-1 list structure with a slice:

```ipython3
result[:, 0]
```

To summarize this as a handy idiom, the way to get the record with maximum `data.x**2 + data.y**2` from an array of lists of records named `data` is

```ipython3
data[ak.argmax(data.x**2 + data.y**2, axis=1, keepdims=True)][:, 0]
```

For an array of lists of lists of records, `axis=2` and the final slice would be `[:, :, 0]`, and so on.

## Sorting by another array

In addition to selecting items corresponding to the minimum or maximum of some other array, we may want to sort by another array. Just as [`ak.argmin()`](sphinx-llm:836050fe860646e49e64fda769d3ddaa#ak.argmin) and [`ak.argmax()`](sphinx-llm:5588cbb2c2804faf860257487414bbc0#ak.argmax) are the functions that would convey indexes from one array to another, [`ak.argsort()`](sphinx-llm:1ac2c8dad5854d5eb0608d90863ee4e6#ak.argsort) conveys sorted indexes from one array to another array. However, [`ak.argsort()`](sphinx-llm:1ac2c8dad5854d5eb0608d90863ee4e6#ak.argsort) always maintains the total number of dimensions, so we don’t need to worry about `keepdims`.

```ipython3
sorted_indexes = ak.argsort(score)
sorted_indexes
```

```ipython3
data[sorted_indexes]
```

This sorted data has the same type as `data`:

```ipython3
data.type.show()
```

```myst-ansi
5 * var * {
    title: string,
    x: int64,
    y: float64
}
```

It’s exactly what we want. [`ak.argsort()`](sphinx-llm:1ac2c8dad5854d5eb0608d90863ee4e6#ak.argsort) is easier to use than [`ak.argmin()`](sphinx-llm:836050fe860646e49e64fda769d3ddaa#ak.argmin) and [`ak.argmax()`](sphinx-llm:5588cbb2c2804faf860257487414bbc0#ak.argmax).

## Getting the top *n* items

The [`ak.min()`](sphinx-llm:d40aa9e4100d4897a291c711bc7a7042#ak.min), [`ak.max()`](sphinx-llm:8d04431ada5f4197aee6b08d9430b68f#ak.max), [`ak.argmin()`](sphinx-llm:836050fe860646e49e64fda769d3ddaa#ak.argmin), and [`ak.argmax()`](sphinx-llm:5588cbb2c2804faf860257487414bbc0#ak.argmax) functions select one extreme value. If you want the top *n* items (with *n ≠ 1*), you can use [`ak.sort()`](sphinx-llm:04e24ad5c9624efeab7a13c95c4f71cf#ak.sort) or [`ak.argsort()`](sphinx-llm:1ac2c8dad5854d5eb0608d90863ee4e6#ak.argsort), followed by a slice:

```ipython3
top2 = data[ak.argsort(score)][:, :2]
top2
```

Notice, though, that not all of these lists have length 2. The lists with 0 or 1 input items have 0 or 1 output items: these lists have *up to* length 2. That may be fine, but the example with [`ak.argmax()`](sphinx-llm:5588cbb2c2804faf860257487414bbc0#ak.argmax), above, resulted in `None` for an empty list. We could emulate that with [`ak.pad_none()`](sphinx-llm:3c6da69bd2db4b3da8eda918d1a9429c#ak.pad_none).

```ipython3
padded = ak.pad_none(top2, 2, axis=1)
padded
```

The data type still says “`var *`”, meaning that the lists are allowed to be variable-length, even though they happen to all have length 2. At this point, we might not care because that’s all we need in order to convert these fields into NumPy arrays (e.g. for some machine learning process):

```ipython3
ak.to_numpy(padded.x)
```

```ipython3
ak.to_numpy(padded.y)
```

Or we might want to force the data type to ensure that the lists have length 2, using [`ak.to_regular()`](sphinx-llm:2a2dc6754b2948d79f20c53eb2b3d2b9#ak.to_regular), [`ak.enforce_type()`](sphinx-llm:ba952268b279401086f48d422c6fd7b9#ak.enforce_type), or just by passing `clip=True` in the original [`ak.pad_none()`](sphinx-llm:3c6da69bd2db4b3da8eda918d1a9429c#ak.pad_none).

```ipython3
ak.to_regular(padded, axis=1)
```

(Now the list lengths are “`2 *`”, rather than “`var *`”.)
