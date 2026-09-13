# ak.ptp

Defined in [awkward.operations.ak_ptp](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_ptp.py) on [line 27](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_ptp.py#L27).

#### ak.ptp(array, axis=None, \*, keepdims=False, mask_identity=True, highlevel=True, behavior=None, attrs=None)

Returns the range of values over one or all levels of nesting.

Many types are supported, including all Awkward Arrays and Records. The
range of an empty list is None, unless `mask_identity=False`, in which case
it is 0. This operation is the same as NumPy’s
[ptp](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ptp.html)
if all lists at a given dimension have the same length and no None values,
but it generalizes to cases where they do not.

See [`ak.sum`](sphinx-llm:08d8d4556ebd40868b7762f4e0ae9d1a#ak.sum) for a more complete description of nested list and missing
value (None) handling in reducers.

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **axis** (*None* *or* [*int*](https://docs.python.org/3/library/functions.html#int) *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – If None, combine all values from the array into
    a single scalar result; if an int, group by that axis: `0` is the
    outermost, `1` is the first level of nested lists, etc., and
    negative `axis` counts from the innermost: `-1` is the innermost,
    `-2` is the next level up, etc; if a str, it is interpreted as the
    name of the axis which maps to an int if named axes are present.
    Named axes are attached to an array using [`ak.with_named_axis`](sphinx-llm:850fa57157f54411a7f413a32c8bd96a#ak.with_named_axis) and
    removed with [`ak.without_named_axis`](sphinx-llm:30b66a6c5ad241b0b374a553292778a2#ak.without_named_axis); also see the
    [Named axes user guide](../../user-guide/how-to-array-properties-named-axis.html).
  * **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If False, this reducer decreases the number of
    dimensions by 1; if True, the reduced values are wrapped in a new
    length-1 dimension so that the result of this operation may be
    broadcasted with the original array.
  * **mask_identity** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, reducing over empty lists results in
    None (an option type); otherwise, reducing over empty lists
    results in the operation’s identity of 0.
* **Returns:**
  The range of values in each group of elements from `array`.

### Examples

For example, with

```pycon
>>> array = ak.Array([[0, 1, 2, 3],
...                   [          ],
...                   [4, 5      ]])
```

The range of the innermost lists is

```pycon
>>> ak.ptp(array, axis=-1)
<Array [3, None, 1] type='3 * ?int64'>
```

because there are three lists, the first has a range of `3`, the second is
`None` because the list is empty, and the third has a range of `1`. Similarly,

```pycon
>>> ak.ptp(array, axis=-1, mask_identity=False)
<Array [3, 0, 1] type='3 * float64'>
```

The second value is `0` because the list is empty.
