# ak.drop_none

Defined in [awkward.operations.ak_drop_none](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_drop_none.py) on [line 21](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_drop_none.py#L21).

#### ak.drop_none(array, axis=None, highlevel=True, behavior=None, attrs=None)

Removes missing values (None) from a given array.

* **Parameters:**
  * **array** – Data in which to remove Nones.
  * **axis** (*None* *or* [*int*](https://docs.python.org/3/library/functions.html#int) *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – If None, the operation drops Nones at all levels
    of nesting, returning an array of the same dimension, but without Nones.
    If an int, the depth at which to drop Nones.
    The outermost dimension is `0`, followed by `1`, etc., and negative
    values count backward from the innermost: `-1` is the innermost
    dimension, `-2` is the next level up, etc.
    If a str, it is interpreted as the name of the axis which maps to
    an int if named axes are present.
    Named axes are attached to an array using [`ak.with_named_axis`](sphinx-llm:850fa57157f54411a7f413a32c8bd96a#ak.with_named_axis) and
    removed with [`ak.without_named_axis`](sphinx-llm:30b66a6c5ad241b0b374a553292778a2#ak.without_named_axis); also see the
    [Named axes user guide](../../user-guide/how-to-array-properties-named-axis.html).
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An array with missing values (None) removed.

### Examples

For example, in the following `array`,

```pycon
>>> array = ak.Array([[[0]], [[None]], [[1], None], [[2, None]]])
```

The None value will be removed, resulting in

```pycon
>>> ak.drop_none(array)
<Array [[[0]], [[]], [[1]], [[2]]] type='4 * var * var * int64'>
```

The default axis is None, however an axis can be specified:

```pycon
>>> ak.drop_none(array, axis=1)
<Array [[[0]], [[None]], [[1]], [[2, None]]] type='4 * var * var * ?int64'>
```
