# ak.mean

Defined in [awkward.operations.ak_mean](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_mean.py) on [line 29](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_mean.py#L29).

#### ak.mean(x, weight=None, axis=None, \*, keepdims=False, mask_identity=False, highlevel=True, behavior=None, attrs=None)

Computes the mean over one or all levels of nesting.

Many types are supported, including all Awkward Arrays and Records. The
grouping is performed the same way as for reducers, though this operation is
not a reducer and has no identity. It is the same as NumPy’s
[mean](https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html)
if all lists at a given dimension have the same length and no None values,
but it generalizes to cases where they do not.

Passing all arguments to the reducers, the mean is calculated as:

```default
ak.sum(x*weight) / ak.sum(weight)
```

See [`ak.sum`](sphinx-llm:08d8d4556ebd40868b7762f4e0ae9d1a#ak.sum) for a complete description of handling nested lists and
missing values (None) in reducers.

See also [`ak.nanmean`](sphinx-llm:d58141c9dfb5425b9cfedecf70a020e5#ak.nanmean).

* **Parameters:**
  * **x** – The data on which to compute the mean (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **weight** – Data that can be broadcasted to `x` to give each value a
    weight. Weighting values equally is the same as no weights;
    weighting some values higher increases the significance of those
    values. Weights can be zero or negative.
  * **axis** (*None* *or* [*int*](https://docs.python.org/3/library/functions.html#int) *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – If None, combine all values from the array into
    a single scalar result; if an int, group by that axis: `0` is the
    outermost, `1` is the first level of nested lists, etc., and
    negative `axis` counts from the innermost: `-1` is the innermost,
    `-2` is the next level up, etc; if a str, it is interpreted as the
    name of the axis which maps to an int if named axes are present.
    Named axes are attached to an array using [`ak.with_named_axis`](sphinx-llm:850fa57157f54411a7f413a32c8bd96a#ak.with_named_axis) and
    removed with [`ak.without_named_axis`](sphinx-llm:30b66a6c5ad241b0b374a553292778a2#ak.without_named_axis); also see the
    [Named axes user guide](../../user-guide/how-to-array-properties-named-axis.html).
  * **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If False, this function decreases the number of
    dimensions by 1; if True, the output values are wrapped in a new
    length-1 dimension so that the result of this operation may be
    broadcasted with the original array.
  * **mask_identity** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, the application of this function on
    empty lists results in None (an option type); otherwise, the
    calculation is followed through with the reducers’ identities,
    usually resulting in floating-point `nan`.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  The mean in each group of elements from `x`.

### Examples

For example, with an `array` like

```pycon
>>> array = ak.Array([[0, 1, 2, 3],
                      [          ],
                      [4, 5      ]])
```

The mean of the innermost lists is

```pycon
>>> ak.mean(array, axis=-1)
<Array [1.5, nan, 4.5] type='3 * float64'>
```

because there are three lists, the first has mean `1.5`, the second is
empty, and the third has mean `4.5`.

The mean of the outermost lists is

```pycon
>>> ak.mean(array, axis=0)
<Array [2, 3, 2, 3] type='4 * float64'>
```

because the longest list has length 4, the mean of `0` and `4` is `2.0`,
the mean of `1` and `5` is `3.0`, the mean of `2` (by itself) is `2.0`,
and the mean of `3` (by itself) is `3.0`. This follows the same grouping
behavior as reducers.
