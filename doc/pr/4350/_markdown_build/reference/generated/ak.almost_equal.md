# ak.almost_equal

Defined in [awkward.operations.ak_almost_equal](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_almost_equal.py) on [line 21](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_almost_equal.py#L21).

#### ak.almost_equal(left, right, \*, rtol: [float](https://docs.python.org/3/library/functions.html#float) = 1e-05, atol: [float](https://docs.python.org/3/library/functions.html#float) = 1e-08, dtype_exact: [bool](https://docs.python.org/3/library/functions.html#bool) = True, check_parameters: [bool](https://docs.python.org/3/library/functions.html#bool) = True, check_regular: [bool](https://docs.python.org/3/library/functions.html#bool) = True, check_named_axis: [bool](https://docs.python.org/3/library/functions.html#bool) = True)

Returns True if two arrays are equal within the given tolerances and options.

The relative difference (`rtol * abs(b)`) and the absolute difference
`atol` are added together to compare against the absolute difference
between `left` and `right`.

TypeTracer arrays are not supported, as there is very little information
to be compared.

* **Parameters:**
  * **left** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **right** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **rtol** – the relative tolerance parameter (see below).
  * **atol** – the absolute tolerance parameter (see below).
  * **dtype_exact** – whether the dtypes must be exactly the same, or just the
    same family.
  * **check_parameters** – whether to compare parameters.
  * **check_regular** – whether to consider ragged and regular dimensions as
    unequal.
  * **check_named_axis** – bool (default=True) whether to consider named axes as unequal.
* **Returns:**
  True if the two array-like arguments are equal within the given options
  and tolerances, False otherwise.
