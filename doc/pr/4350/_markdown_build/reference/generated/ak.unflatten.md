# ak.unflatten

Defined in [awkward.operations.ak_unflatten](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_unflatten.py) on [line 25](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_unflatten.py#L25).

#### ak.unflatten(array, counts, axis=0, \*, highlevel=True, behavior=None, attrs=None)

Returns an array with an additional level of nesting.

An inner dimension can be unflattened by setting the `axis` parameter, but
operations like this constrain the `counts` more tightly.

Also note that new lists created by this function cannot cross partitions
(which is only possible at `axis=0`, anyway).

See also [`ak.num`](sphinx-llm:137e3487aaa0453fa61d9c86fc644b7b#ak.num) and [`ak.flatten`](sphinx-llm:6b8f65e7187745b09f39a6bdf1966b16#ak.flatten).

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **counts** ([*int*](https://docs.python.org/3/library/functions.html#int) *or* *array*) – Number of elements the new level should have.
    If an integer, the new level will be regularly sized; otherwise,
    it will consist of variable-length lists with the given lengths.
  * **axis** ([*int*](https://docs.python.org/3/library/functions.html#int) *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – The dimension at which this operation is applied. The
    outermost dimension is `0`, followed by `1`, etc., and negative
    values count backward from the innermost: `-1` is the innermost
    dimension, `-2` is the next level up, etc.
    If a str, it is interpreted as the name of the axis which maps
    to an int if named axes are present. Named axes are attached
    to an array using [`ak.with_named_axis`](sphinx-llm:850fa57157f54411a7f413a32c8bd96a#ak.with_named_axis) and removed with
    [`ak.without_named_axis`](sphinx-llm:30b66a6c5ad241b0b374a553292778a2#ak.without_named_axis); also see the
    [Named axes user guide](../../user-guide/how-to-array-properties-named-axis.html).
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An array with an additional level of nesting. This is roughly the
  inverse of [`ak.flatten`](sphinx-llm:6b8f65e7187745b09f39a6bdf1966b16#ak.flatten), where `counts` were obtained by [`ak.num`](sphinx-llm:137e3487aaa0453fa61d9c86fc644b7b#ak.num) (both
  with `axis=1`).

### Examples

For example,

```pycon
>>> original = ak.Array([[0, 1, 2], [], [3, 4], [5], [6, 7, 8, 9]])
>>> counts = ak.num(original)
>>> array = ak.flatten(original)
>>> counts
<Array [3, 0, 2, 1, 4] type='5 * int64'>
>>> array
<Array [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] type='10 * int64'>
>>> ak.unflatten(array, counts)
<Array [[0, 1, 2], [], [3, ...], [5], [6, 7, 8, 9]] type='5 * var * int64'>
```

For example, we can subdivide an already divided list:

```pycon
>>> original = ak.Array([[1, 2, 3, 4], [], [5, 6, 7], [8, 9]])
>>> ak.unflatten(original, [2, 2, 1, 2, 1, 1], axis=1).show()
[[[1, 2], [3, 4]],
 [],
 [[5], [6, 7]],
 [[8], [9]]]
```

But the counts have to add up to the lengths of those lists. We can’t mix
values from the first `[1, 2, 3, 4]` with values from the next `[5, 6, 7]`.

```pycon
>>> ak.unflatten(original, [2, 1, 2, 2, 1, 1], axis=1).show()
ValueError: while calling
    ak.unflatten(
        array = <Array [[1, 2, 3, 4], [], ..., [8, 9]] type='4 * var * int64'>
        counts = [2, 1, 2, 2, 1, 1]
        axis = 1
        highlevel = True
        behavior = None
    )
Error details: structure imposed by 'counts' does not fit in the array or partition at axis=1
```
