# ak.argsort

Defined in [awkward.operations.ak_argsort](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_argsort.py) on [line 21](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_argsort.py#L21).

#### ak.argsort(array, axis=-1, \*, ascending=True, stable=True, highlevel=True, behavior=None, attrs=None)

Sorts an array along an axis, returning integer indexes.

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **axis** ([*int*](https://docs.python.org/3/library/functions.html#int) *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – The dimension at which this operation is applied. The
    outermost dimension is `0`, followed by `1`, etc., and negative
    values count backward from the innermost: `-1` is the innermost
    dimension, `-2` is the next level up, etc.
    If a str, it is interpreted as the name of the axis which maps
    to an int if named axes are present. Named axes are attached
    to an array using [`ak.with_named_axis`](sphinx-llm:850fa57157f54411a7f413a32c8bd96a#ak.with_named_axis) and removed with
    [`ak.without_named_axis`](sphinx-llm:30b66a6c5ad241b0b374a553292778a2#ak.without_named_axis); also see the
    [Named axes user guide](../../user-guide/how-to-array-properties-named-axis.html).
  * **ascending** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, the first value in each sorted group
    will be smallest, the last value largest; if False, the order
    is from largest to smallest.
  * **stable** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, use a stable sorting algorithm; if False,
    use a sorting algorithm that is not guaranteed to be stable.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An array of integer indexes that would sort the array if applied
  as an integer-array slice.

### Examples

For example,

```pycon
>>> ak.argsort(ak.Array([[7.7, 5.5, 7.7], [], [2.2], [8.8, 2.2]]))
<Array [[1, 0, 2], [], [0], [1, 0]] type='4 * var * int64'>
```

The result of this function can be used to index other arrays with the
same shape:

```pycon
>>> data = ak.Array([[7, 5, 7], [], [2], [8, 2]])
>>> index = ak.argsort(data)
>>> index
<Array [[1, 0, 2], [], [0], [1, 0]] type='4 * var * int64'>
>>> data[index]
<Array [[5, 7, 7], [], [2], [2, 8]] type='4 * var * int64'>
```
