# ak.local_index

Defined in [awkward.operations.ak_local_index](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_local_index.py) on [line 21](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_local_index.py#L21).

#### ak.local_index(array, axis=-1, \*, highlevel=True, behavior=None, attrs=None)

Returns the within-list index of each element at a given axis depth.

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
  An array of integers giving the local (within-list) position of each
  element at the given `axis`.

### Examples

For example,

```pycon
>>> array = ak.Array([
...     [[0.0, 1.1, 2.2], []],
...     [[3.3, 4.4]],
...     [],
...     [[5.5], [], [6.6, 7.7, 8.8, 9.9]]])
>>> ak.local_index(array, axis=0)
<Array [0, 1, 2, 3] type='4 * int64'>
>>> ak.local_index(array, axis=1)
<Array [[0, 1], [0], [], [0, 1, 2]] type='4 * var * int64'>
>>> ak.local_index(array, axis=2)
<Array [[[0, 1, 2], []], ..., [[0], ..., [...]]] type='4 * var * var * int64'>
```

Note that you can make a Pandas-style MultiIndex by calling this function on
every axis.

```pycon
>>> multiindex = ak.zip([ak.local_index(array, i) for i in range(array.ndim)])
>>> multiindex.show()
[[[(0, 0, 0), (0, 0, 1), (0, 0, 2)], []],
 [[(1, 0, 0), (1, 0, 1)]],
 [],
 [[(3, 0, 0)], [], [(3, 2, 0), (3, 2, 1), (3, 2, 2), (3, 2, 3)]]]
>>> ak.flatten(ak.flatten(multiindex)).show()
[(0, 0, 0),
 (0, 0, 1),
 (0, 0, 2),
 (1, 0, 0),
 (1, 0, 1),
 (3, 0, 0),
 (3, 2, 0),
 (3, 2, 1),
 (3, 2, 2),
 (3, 2, 3)]
```

But if you’re interested in Pandas, you may want to use [`ak.to_dataframe`](sphinx-llm:470705b5237144ff83c26cdb7cdcee4d#ak.to_dataframe) directly.

```pycon
>>> ak.to_dataframe(array)
                            values
entry subentry subsubentry
0     0        0               0.0
               1               1.1
               2               2.2
1     0        0               3.3
               1               4.4
3     0        0               5.5
      2        0               6.6
               1               7.7
               2               8.8
               3               9.9
```
