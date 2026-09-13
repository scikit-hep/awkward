# ak.str.index_in

Defined in [awkward.operations.str.akstr_index_in](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/str/akstr_index_in.py) on [line 15](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/str/akstr_index_in.py#L15).

#### ak.str.index_in(array, value_set, \*, skip_nones=False, highlevel=True, behavior=None, attrs=None)

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **value_set** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes), set of
    values to search for in `array`.
  * **skip_nones** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, None values in `array` are not matched
    against `value_set`; otherwise, None is considered a legal value.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.

Returns the index of the first pattern in `value_set` that each string in
`array` matches. If the string is not found within `value_set`, then the
index is set to None.

Note: this function does not raise an error if the `array` does not
contain any string or bytestring data.

Requires the pyarrow library and calls
[pyarrow.compute.index_in](https://arrow.apache.org/docs/python/generated/pyarrow.compute.index_in.html).
