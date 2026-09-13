# ak.str.ltrim

Defined in [awkward.operations.str.akstr_ltrim](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/str/akstr_ltrim.py) on [line 12](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/str/akstr_ltrim.py#L12).

#### ak.str.ltrim(array, characters, \*, highlevel=True, behavior=None, attrs=None)

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **characters** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *or* [*bytes*](https://docs.python.org/3/library/stdtypes.html#bytes)) – Individual characters to be trimmed
    from the string.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.

Removes any leading characters of `characters` from any string or
bytestring-valued data.

If the data are strings, `characters` are interpreted as unordered,
individual codepoints.

If the data are bytestrings, `characters` are interpreted as unordered,
individual bytes.

Note: this function does not raise an error if the `array` does not
contain any string or bytestring data.

Requires the pyarrow library and calls
[pyarrow.compute.utf8_ltrim](https://arrow.apache.org/docs/python/generated/pyarrow.compute.utf8_ltrim.html)
or
[pyarrow.compute.ascii_ltrim](https://arrow.apache.org/docs/python/generated/pyarrow.compute.ascii_ltrim.html)
on strings and bytestrings, respectively.

See also: [`ak.str.ltrim_whitespace`](sphinx-llm:c00ce81bc30c4d4eaa53605a2996d5da#ak.str.ltrim_whitespace).
