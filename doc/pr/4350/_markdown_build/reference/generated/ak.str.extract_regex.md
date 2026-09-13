# ak.str.extract_regex

Defined in [awkward.operations.str.akstr_extract_regex](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/str/akstr_extract_regex.py) on [line 12](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/str/akstr_extract_regex.py#L12).

#### ak.str.extract_regex(array, pattern, \*, highlevel=True, behavior=None, attrs=None)

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **pattern** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *or* [*bytes*](https://docs.python.org/3/library/stdtypes.html#bytes)) – Regular expression with named capture fields.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.

Returns None for every string in `array` if it does not match `pattern`;
otherwise, a record whose fields are named capture groups and whose
contents are the substrings they’ve captured.

Uses [Google RE2](https://github.com/google/re2/wiki/Syntax), and `pattern` must
contain named groups. The syntax for a named group is `(?P<...>...)` in which
the first `...` is a name and the last `...` is a regular expression.

For example,

```pycon
>>> array = ak.Array([["one1", "two2", "three3"], [], ["four4", "five5"]])
>>> result = ak.str.extract_regex(array, "(?P<vowel>[aeiou])(?P<number>[0-9]+)")
>>> result.show(type=True)
type: 3 * var * ?{
    vowel: ?string,
    number: ?string
}
[[{vowel: 'e', number: '1'}, {vowel: 'o', number: '2'}, {vowel: 'e', number: '3'}],
 [],
 [None, {vowel: 'e', number: '5'}]]
```

(The string `"four4"` does not match because the vowel is not immediately before
the number.)

Regular expressions with unnamed groups or features not implemented by RE2 raise an error.

Note: this function does not raise an error if the `array` does not
contain any string or bytestring data.

Requires the pyarrow library and calls
[pyarrow.compute.extract_regex](https://arrow.apache.org/docs/python/generated/pyarrow.compute.extract_regex.html).
