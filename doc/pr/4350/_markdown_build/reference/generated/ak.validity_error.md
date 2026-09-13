# ak.validity_error

Defined in [awkward.operations.ak_validity_error](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_validity_error.py) on [line 11](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_validity_error.py#L11).

#### ak.validity_error(array, \*, exception=False)

Returns an error message if the array has a structural error, or empty if valid.

Checks for errors in the structure of the array, such as indexes that run
beyond the length of a node’s `content`, etc. Either an error is raised or
a string describing the error is returned.

See also [`ak.is_valid`](sphinx-llm:cb72d589ba0e433d9cedf3f12cc0da1c#ak.is_valid).

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **exception** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, validity errors raise exceptions.
* **Returns:**
  An empty string if `array` is valid, or a string describing the structural
  error otherwise.
