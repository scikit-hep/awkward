# ak.run_lengths

Defined in [awkward.operations.ak_run_lengths](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_run_lengths.py) on [line 17](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_run_lengths.py#L17).

#### ak.run_lengths(array, \*, highlevel=True, behavior=None, attrs=None)

Returns the lengths of runs of identical values at the deepest level.

See also [`ak.num`](sphinx-llm:137e3487aaa0453fa61d9c86fc644b7b#ak.num), [`ak.argsort`](sphinx-llm:1ac2c8dad5854d5eb0608d90863ee4e6#ak.argsort), [`ak.unflatten`](sphinx-llm:5d3b5ff645af400f8e5c1b3e5c653d77#ak.unflatten).

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  The lengths of sequences of identical values at the deepest level
  of nesting, returning an array with the same structure but with `int64` type.

### Examples

For example,

```pycon
>>> array = ak.Array([1.1, 1.1, 1.1, 2.2, 3.3, 3.3, 4.4, 4.4, 5.5])
>>> ak.run_lengths(array)
<Array [3, 1, 2, 2, 1] type='5 * int64'>
```

There are 3 instances of 1.1, followed by 1 instance of 2.2, 2 instances of 3.3,
2 instances of 4.4, and 1 instance of 5.5.

The order and uniqueness of the input data doesn’t matter,

```pycon
>>> array = ak.Array([1.1, 1.1, 1.1, 5.5, 4.4, 4.4, 1.1, 1.1, 5.5])
>>> ak.run_lengths(array)
<Array [3, 1, 2, 2, 1] type='5 * int64'>
```

just the difference between each value and its neighbors.

The data can be nested, but runs don’t cross list boundaries.

```pycon
>>> array = ak.Array([[1.1, 1.1, 1.1, 2.2, 3.3], [3.3, 4.4], [4.4, 5.5]])
>>> ak.run_lengths(array)
<Array [[3, 1, 1], [1, 1], [1, 1]] type='3 * var * int64'>
```

This function recognizes strings as distinguishable values.

```pycon
>>> array = ak.Array([["one", "one"], ["one", "two", "two"], ["three", "two", "two"]])
>>> ak.run_lengths(array)
<Array [[2], [1, 2], [1, 2]] type='3 * var * int64'>
```

Note that this can be combined with [`ak.argsort`](sphinx-llm:1ac2c8dad5854d5eb0608d90863ee4e6#ak.argsort) and [`ak.unflatten`](sphinx-llm:5d3b5ff645af400f8e5c1b3e5c653d77#ak.unflatten) to compute
a “group by” operation:

```pycon
>>> array = ak.Array([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 1, "y": 1.1},
...                   {"x": 3, "y": 3.3}, {"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}])
>>> sorted = array[ak.argsort(array.x)]
>>> sorted.x
<Array [1, 1, 1, 2, 2, 3] type='6 * int64'>
>>> ak.run_lengths(sorted.x)
<Array [3, 2, 1] type='3 * int64'>
>>> ak.unflatten(sorted, ak.run_lengths(sorted.x)).show()
[[{x: 1, y: 1.1}, {x: 1, y: 1.1}, {x: 1, y: 1.1}],
 [{x: 2, y: 2.2}, {x: 2, y: 2.2}],
 [{x: 3, y: 3.3}]]
```

Unlike a database “group by,” this operation can be applied in bulk to many sublists
(though the run lengths need to be fully flattened to be used as `counts` for
[`ak.unflatten`](sphinx-llm:5d3b5ff645af400f8e5c1b3e5c653d77#ak.unflatten), and you need to specify `axis=-1` as the depth).

```pycon
>>> array = ak.Array([[{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 1, "y": 1.1}],
...                   [{"x": 3, "y": 3.3}, {"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}]])
>>> sorted = array[ak.argsort(array.x)]
>>> sorted.x
<Array [[1, 1, 2], [1, 2, 3]] type='2 * var * int64'>
>>> ak.run_lengths(sorted.x)
<Array [[2, 1], [1, 1, 1]] type='2 * var * int64'>
>>> counts = ak.flatten(ak.run_lengths(sorted.x), axis=None)
>>> ak.unflatten(sorted, counts, axis=-1).show()
[[[{x: 1, y: 1.1}, {x: 1, y: 1.1}], [{x: 2, y: 2.2}]],
 [[{x: 1, y: 1.1}], [{x: 2, y: 2.2}], [{x: 3, y: 3.3}]]]
```
