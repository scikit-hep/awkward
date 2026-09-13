# ak.linear_fit

Defined in [awkward.operations.ak_linear_fit](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_linear_fit.py) on [line 20](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_linear_fit.py#L20).

#### ak.linear_fit(x, y, weight=None, axis=None, \*, keepdims=False, mask_identity=False, highlevel=True, behavior=None, attrs=None)

Computes the linear fit of y against x over one or all levels of nesting.

Many types are supported, including all Awkward Arrays and Records, which
must be broadcastable to each other. The grouping is performed the same way
as for reducers, though this operation is not a reducer and has no identity.

This function has no NumPy equivalent.

Passing all arguments to the reducers, the linear fit is calculated as:

```default
sumw            = ak.sum(weight)
sumwx           = ak.sum(weight * x)
sumwy           = ak.sum(weight * y)
sumwxx          = ak.sum(weight * x**2)
sumwxy          = ak.sum(weight * x * y)
delta           = (sumw*sumwxx) - (sumwx*sumwx)

intercept       = ((sumwxx*sumwy) - (sumwx*sumwxy)) / delta
slope           = ((sumw*sumwxy) - (sumwx*sumwy))   / delta
intercept_error = np.sqrt(sumwxx / delta)
slope_error     = np.sqrt(sumw   / delta)
```

The results, `intercept`, `slope`, `intercept_error`, and `slope_error`,
are given as an [`ak.Record`](sphinx-llm:e158df99f80043309c758f6e933352b2#ak.Record) with four fields. The values of these fields
might be arrays or even nested arrays; they match the structure of `x` and
`y`.

See [`ak.sum`](sphinx-llm:08d8d4556ebd40868b7762f4e0ae9d1a#ak.sum) for a complete description of handling nested lists and
missing values (None) in reducers, and [`ak.mean`](sphinx-llm:9dfbdca7e4104165863b3b3f4391b21e#ak.mean) for an example with another
non-reducer.

* **Parameters:**
  * **x** – One coordinate to use in the linear fit (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **y** – The other coordinate to use in the linear fit (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **weight** – Data that can be broadcasted to `x` and `y` to give each point
    a weight. Weighting points equally is the same as no weights;
    weighting some points higher increases the significance of those
    points. Weights can be zero or negative.
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
  The linear fit of `y` with respect to `x`.
