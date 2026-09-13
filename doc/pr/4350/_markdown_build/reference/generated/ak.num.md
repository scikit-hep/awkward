# ak.num

Defined in [awkward.operations.ak_num](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_num.py) on [line 23](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_num.py#L23).

#### ak.num(array, axis=1, \*, highlevel: [bool](https://docs.python.org/3/library/functions.html#bool) = True, behavior: awkward._typing.Mapping | [None](https://docs.python.org/3/library/constants.html#None) = None, attrs: awkward._typing.Mapping | [None](https://docs.python.org/3/library/constants.html#None) = None)

Returns the number of elements at a given axis depth.

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
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An array of integers specifying the number of elements at a
  particular level.

### Examples

For instance, given the following doubly nested `array`,

```pycon
>>> array = ak.Array([[[1.1, 2.2, 3.3],
...                    [],
...                    [4.4, 5.5],
...                    [6.6]
...                   ],
...                   [],
...                   [[7.7],
...                    [8.8, 9.9]]
...                   ])
```

The number of elements in `axis=1` is

```pycon
>>> ak.num(array, axis=1)
<Array [4, 0, 2] type='3 * int64'>
```

and the number of elements at the next level down, `axis=2`, is

```pycon
>>> ak.num(array, axis=2)
<Array [[3, 0, 2, 1], [], [1, 2]] type='3 * var * int64'>
```

The `axis=0` case is special: it returns a scalar, the length of the array.

```pycon
>>> ak.num(array, axis=0)
3
```

This function is useful for ensuring that slices do not raise errors. For
instance, suppose that we want to select the first element from each
of the outermost nested lists of `array`. One of these lists is empty, so
selecting the first element (`0`) would raise an error. However, if our
first selection is `ak.num(array) > 0`, we are left with only those lists
that *do* have a first element:

```pycon
>>> array[ak.num(array) > 0, 0]
<Array [[1.1, 2.2, 3.3], [7.7]] type='2 * var * float64'>
```

To keep a placeholder (None) in each place we do not want to select,
consider using [`ak.mask`](sphinx-llm:9a507fdd6567423185d9261d0dc62bad#ak.mask) instead of a [`ak.Array.__getitem__`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array.__getitem__).

```pycon
>>> array.mask[ak.num(array) > 0][:, 0]
<Array [[1.1, 2.2, 3.3], None, [7.7]] type='3 * option[var * float64]'>
```
