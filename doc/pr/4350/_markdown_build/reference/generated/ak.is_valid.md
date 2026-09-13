# ak.is_valid

Defined in [awkward.operations.ak_is_valid](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_is_valid.py) on [line 11](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_is_valid.py#L11).

#### ak.is_valid(array, \*, exception=False)

Returns True if the array has no structural errors and False otherwise.

Checks for errors in the structure of the array, such as indexes that run
beyond the length of a node’s `content`, etc. Either an error is raised or
the function returns a boolean.

See also [`ak.validity_error`](sphinx-llm:ccbf3feebdb343f8ae3e23ce953f394a#ak.validity_error).

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **exception** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, validity errors raise exceptions.
* **Returns:**
  True if `array` has no structural errors, False otherwise.
