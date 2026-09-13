# ak.is_tuple

Defined in [awkward.operations.ak_is_tuple](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_is_tuple.py) on [line 11](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_is_tuple.py#L11).

#### ak.is_tuple(array)

Returns True if a record, or the outermost record of an array, is a tuple.

If `array` is a record, this returns True if the record is a tuple. If
`array` is an array, this returns True if the outermost record is a tuple.

* **Parameters:**
  **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
* **Returns:**
  True if `array` (or its outermost record) is a tuple, False otherwise.
