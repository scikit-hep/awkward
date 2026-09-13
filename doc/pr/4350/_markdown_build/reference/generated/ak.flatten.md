# ak.flatten

Defined in [awkward.operations.ak_flatten](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_flatten.py) on [line 22](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_flatten.py#L22).

#### ak.flatten(array, axis=1, \*, highlevel=True, behavior=None, attrs=None)

Returns an array with one or all levels of nesting removed.

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **axis** (*None* *or* [*int*](https://docs.python.org/3/library/functions.html#int) *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – If None, the operation flattens all levels of
    nesting, returning a 1-dimensional array. If an int, it flattens
    at a specified depth. The outermost dimension is `0`, followed
    by `1`, etc., and negative values count backward from the
    innermost: `-1` is the innermost dimension, `-2` is the next
    level up, etc. If a str, it is interpreted as the
    name of the axis which maps to an int if named axes are present.
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
  An array with one level of nesting removed by erasing the
  boundaries between consecutive lists. Since this operates on a level of
  nesting, `axis=0` is a special case that only removes values at the
  top level that are equal to None.

### Examples

Consider the following.

```pycon
>>> array = ak.Array([[[1.1, 2.2, 3.3],
...                    [],
...                    [4.4, 5.5],
...                    [6.6]],
...                   [],
...                   [[7.7],
...                    [8.8, 9.9]
...                   ]])
```

At `axis=1`, the outer lists (length 4, length 0, length 2) become a single
list (of length 6).

```pycon
>>> ak.flatten(array, axis=1).show()
[[1.1, 2.2, 3.3],
 [],
 [4.4, 5.5],
 [6.6],
 [7.7],
 [8.8, 9.9]]
```

At `axis=2`, the inner lists (lengths 3, 0, 2, 1, 1, and 2) become three
lists (of lengths 6, 0, and 3).

```pycon
>>> ak.flatten(array, axis=2).show()
[[1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
 [],
 [7.7, 8.8, 9.9]]
```

There’s also an option to completely flatten the array with `axis=None`.
This is useful for passing the data to a function that doesn’t care about
nested structure, such as a plotting routine.

```pycon
>>> ak.flatten(array, axis=None).show()
[1.1,
 2.2,
 3.3,
 4.4,
 5.5,
 6.6,
 7.7,
 8.8,
 9.9]
```

Missing values are eliminated by flattening: there is no distinction
between an empty list and a value of None at the level of flattening.

```pycon
>>> array = ak.Array([[1.1, 2.2, 3.3], None, [4.4], [], [5.5]])
>>> ak.flatten(array, axis=1)
<Array [1.1, 2.2, 3.3, 4.4, 5.5] type='5 * float64'>
```

As a consequence, flattening at `axis=0` does only one thing: it removes
None values from the top level.

```pycon
>>> ak.flatten(array, axis=0)
<Array [[1.1, 2.2, 3.3], [4.4], [], [5.5]] type='4 * var * float64'>
```

As a technical detail, the flattening operation can be trivial in a common
case, [`ak.contents.ListOffsetArray`](sphinx-llm:00659f65177e4a1db8f94cec2f25984d#ak.contents.ListOffsetArray) in which the first `offset` is `0`.
In that case, the flattened data is simply the array node’s `content`.

```pycon
>>> array = ak.Array([[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]])
>>> array.layout
<ListOffsetArray len='5'>
    <offsets><Index dtype='int64' len='6'>
        [ 0  3  3  5  6 10]
    </Index></offsets>
    <content><NumpyArray dtype='float64' len='10'>
        [0.  1.1 2.2 3.3 4.4 5.5 6.6 7.7 8.8 9.9]
    </NumpyArray></content>
</ListOffsetArray>
```

```pycon
>>> ak.flatten(array).layout
<NumpyArray dtype='float64' len='10'>
    [0.  1.1 2.2 3.3 4.4 5.5 6.6 7.7 8.8 9.9]
</NumpyArray>
```

```pycon
>>> array.layout.content
<NumpyArray dtype='float64' len='10'>
    [0.  1.1 2.2 3.3 4.4 5.5 6.6 7.7 8.8 9.9]
</NumpyArray>
```

However, it is important to keep in mind that this is a special case:
[`ak.flatten`](sphinx-llm:0d1ee503747a4cda9195dcd2914c7ab1) and `content` are not interchangeable!

```pycon
>>> array = ak.Array(
...     ak.contents.ListArray(
...         ak.index.Index64(np.array([ 9, 100, 5, 8, 1])),
...         ak.index.Index64(np.array([12, 100, 7, 9, 5])),
...         ak.contents.NumpyArray(
...             np.array([999, 6.6, 7.7, 8.8, 9.9, 3.3, 4.4, 999, 5.5, 0., 1.1, 2.2, 999])
...         ),
...     )
... )
>>> array.show()
[[0, 1.1, 2.2],
 [],
 [3.3, 4.4],
 [5.5],
 [6.6, 7.7, 8.8, 9.9]]
```

```pycon
>>> ak.flatten(array).show()
[0,
 1.1,
 2.2,
 3.3,
 4.4,
 5.5,
 6.6,
 7.7,
 8.8,
 9.9]
```

```pycon
>>> ak.Array(array.layout.content).show()
[999,
 6.6,
 7.7,
 8.8,
 9.9,
 3.3,
 4.4,
 999,
 5.5,
 0,
 1.1,
 2.2,
 999]
```
