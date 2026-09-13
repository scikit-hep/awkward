# ak.str.starts_with

Defined in [awkward.operations.str.akstr_starts_with](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/str/akstr_starts_with.py) on [line 12](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/str/akstr_starts_with.py#L12).

#### ak.str.starts_with(array, pattern, \*, ignore_case=False, highlevel=True, behavior=None, attrs=None)

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **pattern** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *or* [*bytes*](https://docs.python.org/3/library/stdtypes.html#bytes)) – Substring pattern to test against the start
    of each string in `array`.
  * **ignore_case** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, perform a case-insensitive match;
    otherwise, the match is case-sensitive.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.

Returns True for every string in `array` if it starts with the given literal
suffix `pattern`. Depending upon the value of `ignore_case`, the matching
function will be case-insensitive.

Note: this function does not raise an error if the `array` does not
contain any string or bytestring data.

Requires the pyarrow library and calls
[pyarrow.compute.starts_with](https://arrow.apache.org/docs/python/generated/pyarrow.compute.starts_with.html).
