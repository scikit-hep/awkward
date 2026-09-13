# ak.str.join

Defined in [awkward.operations.str.akstr_join](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/str/akstr_join.py) on [line 20](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/str/akstr_join.py#L20).

#### ak.str.join(array, separator, \*, highlevel=True, behavior=None, attrs=None)

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **separator** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* [*bytes*](https://docs.python.org/3/library/stdtypes.html#bytes) *, or* *array* *of* *them to broadcast*) – separator to
    insert between strings. If array-like, `separator` is broadcast
    against `array`.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.

Concatenate the strings in `array`. The `separator` is inserted between
each string. If array-like, `separator` is broadcast against `array` which
permits a unique separator for each list of strings in `array`.

Note: this function does not raise an error if the `array` does not
contain any string or bytestring data.

Requires the pyarrow library and calls
[pyarrow.compute.binary_join](https://arrow.apache.org/docs/python/generated/pyarrow.compute.binary_join.html).

See also: [`ak.str.join_element_wise`](sphinx-llm:5121f9395cf048e38b8cc313836940d0#ak.str.join_element_wise).
