# ak.to_json

Defined in [awkward.operations.ak_to_json](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_to_json.py) on [line 25](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_to_json.py#L25).

#### ak.to_json(array, file=None, \*, line_delimited=False, num_indent_spaces=None, num_readability_spaces=0, nan_string=None, posinf_string=None, neginf_string=None, complex_record_fields=None, convert_bytes=None, convert_other=None)

Converts an Awkward Array into JSON text, as a string or to a file.

This function converts the array into Python objects with [`ak.to_list`](sphinx-llm:c5d0843c37c34cb88bbec16becc6a6f0#ak.to_list), performs
some conversions to make the data JSON serializable (`nan_string`,
`posinf_string`, `neginf_string`, `complex_record_fields`, `convert_bytes`,
`convert_other`), then uses `json.dumps` to return a string or `json.dump`
to write to a file (depending on the value of `file`).

If `line_delimited` is True or a line-delimiter string like `"\r\n"`/`os.linesep`,
the output is line-delimited JSON, variously referred to as “ldjson”, “ndjson”, and
“jsonl”. (Use an appropriate file extension!)

To pretty-print the JSON, set `num_indent_spaces=4, num_readability_spaces=1` (for
example).

Awkward Array types have the following JSON translations.

* [`ak.types.OptionType`](sphinx-llm:f8e0eb36e1334eef887fb5ab81458600#ak.types.OptionType): missing values are converted into None.
* [`ak.types.ListType`](sphinx-llm:649f7f37622a4fbc999e70526300cf12#ak.types.ListType): converted into JSON lists.
* [`ak.types.RegularType`](sphinx-llm:a729849acf504584b9a7d4ae728c6263#ak.types.RegularType): also converted into JSON lists. JSON (and
  Python) forms lose information about the regularity of list lengths.
* [`ak.types.ListType`](sphinx-llm:649f7f37622a4fbc999e70526300cf12#ak.types.ListType) or [`ak.types.RegularType`](sphinx-llm:a729849acf504584b9a7d4ae728c6263#ak.types.RegularType) with parameter `"__array__"`
  equal to `"string"`: converted into JSON strings.
* [`ak.types.RecordType`](sphinx-llm:516b58d6a2f7412aadbaf0859053019a#ak.types.RecordType) without field names: converted into JSON
  objects with numbers as strings for keys.
* [`ak.types.RecordType`](sphinx-llm:516b58d6a2f7412aadbaf0859053019a#ak.types.RecordType) with field names: converted into JSON objects.
* [`ak.types.UnionType`](sphinx-llm:5e1485b00d7b47f39f0082bb755fa92f#ak.types.UnionType): JSON data are naturally heterogeneous.

If the array contains any NaN (not a number), infinite values, or
imaginary/complex types, `nan_string`, `posinf_string`, and/or `neginf_string`
\_must_ be supplied.

If the array contains any raw bytestrings (`"__array__"` equal to `"bytestring"`),
`convert_bytes` \_must_ be supplied. To interpret as strings, use `bytes.decode`.
To Base64-encode, use `lambda x: base64.b64encode(x).decode()`.

Other non-serializable types are only possible through custom behaviors that
override `__getitem__` (which might return arbitrary Python objects). Use
`convert_other` to detect these types and convert them.

See also [`ak.from_json`](sphinx-llm:8dc434abe56445a0ba5c48a99f18b65e#ak.from_json).

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **file** (*None* *,* *path-like* *, or* *file-like object*) – If None, this function returns
    JSON-encoded bytes. Otherwise, this function has no return value.
    If a string/pathlib.Path, this function opens a file with that name, writes JSON
    data, and closes the file. If that path has a URI protocol (like
    “[https://](https://)” or “s3://”), this function attempts to open the file with
    the fsspec library. If a file-like object with a `write` method,
    this function writes to the object, but does not close it.
  * **line_delimited** ([*bool*](https://docs.python.org/3/library/functions.html#bool) *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – If False, a single JSON document is written,
    representing the entire array or record. If True, each element of the
    array (or just the one record) is written on a separate line of text,
    separated by `"\n"`. If a string, such as `"\r\n"`, it is taken as a
    custom line delimiter. (Use `os.linesep` for a platform-dependent
    line delimiter.)
  * **num_indent_spaces** (*None* *or* *nonnegative int*) – Number of spaces to indent nested
    elements, for pretty-printed JSON. If None, the JSON output is written
    on one line of text. Ignored if `line_delimited` is True or a string.
  * **num_readability_spaces** (*nonnegative int*) – Number of spaces to include after
    commas (`,`) and colons (`:`), for pretty-printed JSON.
  * **nan_string** (*None* *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – If not None, floating-point NaN values will be
    replaced with this string instead of a JSON number.
  * **posinf_string** (*None* *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – If not None, floating-point positive infinity
    values will be replaced with this string instead of a JSON number.
  * **neginf_string** (*None* *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – If not None, floating-point negative infinity
    values will be replaced with this string instead of a JSON number.
  * **complex_record_fields** (*None* *or*  *(*[*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* [*str*](https://docs.python.org/3/library/stdtypes.html#str) *)*) – If not None, defines a pair of
    field names to interpret records as complex numbers, such as
    `("real", "imag")`.
  * **convert_bytes** (*None* *or* *function*) – If not None, this function is applied to
    all Python 3 bytes objects to produce something JSON serializable,
    such as a string using UTF-8 or Base64 encoding, lists of integers, etc.
  * **convert_other** (*None* *or* *function*) – Passed to `json.dump` or `json.dumps`
    as `default` to convert any other objects that [`ak.to_list`](sphinx-llm:c5d0843c37c34cb88bbec16becc6a6f0#ak.to_list) would return
    but are not JSON serializable.
* **Returns:**
  The JSON text as bytes when `file` is None; otherwise None, and the JSON
  is written to `file`.
