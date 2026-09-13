# ak.str.replace_substring_regex

Defined in [awkward.operations.str.akstr_replace_substring_regex](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/str/akstr_replace_substring_regex.py) on [line 12](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/str/akstr_replace_substring_regex.py#L12).

#### ak.str.replace_substring_regex(array, pattern, replacement, \*, max_replacements=None, highlevel=True, behavior=None, attrs=None)

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **pattern** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Regular expression pattern to look for inside input values.
  * **replacement** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *or* [*bytes*](https://docs.python.org/3/library/stdtypes.html#bytes)) – What to replace the pattern with.
  * **max_replacements** (*None* *or* [*int*](https://docs.python.org/3/library/functions.html#int)) – If not None and not -1, limits the
    maximum number of replacements per string/bytestring, counting from
    the left.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.

Replaces non-overlapping subsequences of any string or bytestring-valued
data that match a regular expression `pattern` with `replacement`.

The `pattern` and `replacement` are scalars; they cannot be different
for each string/bytestring in the sample.

Note: this function does not raise an error if the `array` does not
contain any string or bytestring data.

Requires the pyarrow library and calls
[pyarrow.compute.replace_substring_regex](https://arrow.apache.org/docs/python/generated/pyarrow.compute.replace_substring_regex.html)
or
[pyarrow.compute.replace_substring_regex](https://arrow.apache.org/docs/python/generated/pyarrow.compute.replace_substring_regex.html)
on strings and bytestrings, respectively.

See also: [`ak.str.replace_substring_regex`](sphinx-llm:ad4dd4916b554ffebf66a18f5e94f868).
