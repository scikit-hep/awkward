# ak.str.split_pattern

Defined in [awkward.operations.str.akstr_split_pattern](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/str/akstr_split_pattern.py) on [line 12](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/str/akstr_split_pattern.py#L12).

#### ak.str.split_pattern(array, pattern, \*, max_splits=None, reverse=False, highlevel=True, behavior=None, attrs=None)

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **pattern** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *or* [*bytes*](https://docs.python.org/3/library/stdtypes.html#bytes)) – Pattern of characters/bytes to split on.
  * **max_splits** (*None* *or* [*int*](https://docs.python.org/3/library/functions.html#int)) – Maximum number of splits for each input
    value. If None, unlimited.
  * **reverse** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, start splitting from the end of each input
    value; otherwise, start splitting from the beginning of each
    value. This flag only has an effect if `max_splits` is not None.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.

Splits any string or bytestring-valued data into a list of substrings
according to the given separator.

Note: this function does not raise an error if the `array` does not
contain any string or bytestring data.

Requires the pyarrow library and calls
[pyarrow.compute.split_pattern](https://arrow.apache.org/docs/python/generated/pyarrow.compute.split_pattern.html).

See also: [`ak.str.split_whitespace`](sphinx-llm:b378d3cae48d4a6b90f5696453fd9037#ak.str.split_whitespace), [`ak.str.split_pattern_regex`](sphinx-llm:23d6edb8aa98411092d4dcbb08da5bc3#ak.str.split_pattern_regex).
