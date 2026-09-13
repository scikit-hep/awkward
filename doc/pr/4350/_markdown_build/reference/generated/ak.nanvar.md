# ak.nanvar

Defined in [awkward.operations.ak_var](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_var.py) on [line 128](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_var.py#L128).

#### ak.nanvar(x, weight=None, ddof=0, axis=None, \*, keepdims=False, mask_identity=True, highlevel=True, behavior=None, attrs=None)

Computes the variance, treating NaN values as missing.

Equivalent to:

```default
ak.var(ak.nan_to_none(array))
```

with all other arguments unchanged.

See also [`ak.var`](sphinx-llm:a102203a874241f493f04760ce9ca52a#ak.var).

* **Parameters:**
  * **x** – The data on which to compute the variance (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **weight** – Data that can be broadcasted to `x` to give each value a
    weight. Weighting values equally is the same as no weights;
    weighting some values higher increases the significance of those
    values. Weights can be zero or negative.
  * **ddof** ([*int*](https://docs.python.org/3/library/functions.html#int)) – “delta degrees of freedom”: the divisor used in the
    calculation is `sum(weights) - ddof`. Use this for “reduced
    variance.”
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
  Like [`ak.var`](sphinx-llm:a102203a874241f493f04760ce9ca52a#ak.var), but treating NaN (“not a number”) values as missing.
