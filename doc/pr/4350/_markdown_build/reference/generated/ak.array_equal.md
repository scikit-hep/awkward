# ak.array_equal

Defined in [awkward.operations.ak_array_equal](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_array_equal.py) on [line 12](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_array_equal.py#L12).

#### ak.array_equal(a1, a2, equal_nan: [bool](https://docs.python.org/3/library/functions.html#bool) = False, dtype_exact: [bool](https://docs.python.org/3/library/functions.html#bool) = True, same_content_types: [bool](https://docs.python.org/3/library/functions.html#bool) = True, check_parameters: [bool](https://docs.python.org/3/library/functions.html#bool) = True, check_regular: [bool](https://docs.python.org/3/library/functions.html#bool) = True, check_named_axis: [bool](https://docs.python.org/3/library/functions.html#bool) = True)

Returns True if two arrays have the same shape and elements.

TypeTracer arrays are not supported, as there is very little information
to be compared.

* **Parameters:**
  * **a1** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **a2** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **equal_nan** – bool (default=False)
    Whether to count NaN values as equal to each other.
  * **dtype_exact** – bool (default=True) whether the dtypes must be exactly the same, or just the
    same family.
  * **same_content_types** – bool (default=True)
    Whether to require all content classes to match
  * **check_parameters** – bool (default=True) whether to compare parameters.
  * **check_regular** – bool (default=True) whether to consider ragged and regular dimensions as
    unequal.
  * **check_named_axis** – bool (default=True) whether to consider named axes as unequal.
* **Returns:**
  True if two arrays have the same shape and elements, False otherwise.
