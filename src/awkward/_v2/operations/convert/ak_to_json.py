# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import json
from urllib.parse import urlparse
from numbers import Number

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def to_json(
    array,
    file=None,
    line_delimited=False,
    num_indent_spaces=None,
    num_readability_spaces=0,
    nan_string=None,
    infinity_string=None,
    minus_infinity_string=None,
    complex_record_fields=None,
    convert_bytes=None,
    convert_other=None,
):
    """
    Args:
        array: Data to convert to JSON.
        file (None, str, or file-like object): If None, this function returns a
            JSON-encoded string. Otherwise, this function has no return value.
            If a string, this function opens a file with that name, writes JSON
            data, and closes the file. If that string has a URI protocol (like
            "https://" or "s3://"), this function attempts to open the file with
            the fsspec library. If a file-like object with a `write` method,
            this function writes to the object, but does not close it.
        line_delimited (bool or str): If False, a single JSON document is written,
            representing the entire array or record. If True, each element of the
            array (or just the one record) is written on a separate line of text,
            separated by `"\n"`. If a string, such as `"\r\n"`, it is taken as a
            custom line delimiter. (Use `os.linesep` for a platform-dependent
            line delimiter.)
        num_indent_spaces (None or nonnegative int): Number of spaces to indent nested
            elements, for pretty-printed JSON. If None, the JSON output is written
            on one line of text. Ignored if `line_delimited` is True or a string.
        num_readability_spaces (nonnegative int): Number of spaces to include after
            commas (`,`) and colons (`:`), for pretty-printed JSON.
        nan_string (None or str): If not None, floating-point NaN values will be
            replaced with this string instead of a JSON number.
        infinity_string (None or str): If not None, floating-point positive infinity
            values will be replaced with this string instead of a JSON number.
        minus_infinity_string (None or str): If not None, floating-point negative
            infinity values will be replaced with this string instead of a JSON
            number.
        complex_record_fields (None or (str, str)): If not None, defines a pair of
            field names to interpret records as complex numbers.
        convert_bytes (None or function): If not None, this function is applied to
            all Python 3 bytes objects to produce something JSON serializable,
            such as a string using UTF-8 or Base64 encoding, lists of integers, etc.
        convert_other (None or function): Passed to `json.dump` or `json.dumps`
            as `default` to convert any other objects that #ak.to_list would return
            but are not JSON serializable.

    Converts `array` (many types supported, including all Awkward Arrays and
    Records) into a JSON string.

    This function converts the array into Python objects with #ak.to_list, performs
    some conversions to make the data JSON serializable (`nan_string`, `infinity_string`,
    `minus_infinity_string`, `complex_record_fields`, `convert_bytes`, `convert_other`),
    then uses `json.dumps` to return a string or `json.dump` to write to a file
    (depending on the value of `file`).

    If `line_delimited` is True or a line-delimiter string like `"\r\n"`/`os.linesep`,
    the output is line-delimited JSON, variously referred to as "ldjson", "ndjson", and
    "jsonl". (Use an appropriate file extension!)

    To pretty-print the JSON, set `num_indent_spaces=4, num_readability_spaces=1` (for
    example).

    Awkward Array types have the following JSON translations.

       * #ak.types.PrimitiveType: converted into JSON booleans and numbers.
       * #ak.types.OptionType: missing values are converted into None.
       * #ak.types.ListType: converted into JSON lists.
       * #ak.types.RegularType: also converted into JSON lists. JSON (and
         Python) forms lose information about the regularity of list lengths.
       * #ak.types.ListType or #ak.types.RegularType with parameter `"__array__"`
         equal to `"string"`: converted into JSON strings.
       * #ak.types.RecordArray without field names: converted into JSON
         objects with numbers as strings for keys.
       * #ak.types.RecordArray with field names: converted into JSON objects.
       * #ak.types.UnionArray: JSON data are naturally heterogeneous.

    If the array contains any NaN (not a number), infinite values, or
    imaginary/complex types, `nan_string` or `infinity_string` _must_ be supplied.

    If the array contains any raw bytestrings (`"__array__"` equal to `"bytestring"`),
    `convert_bytes` _must_ be supplied. To interpret as strings, use `bytes.decode`.
    To Base64-encode, use `lambda x: base64.b64encode(x).decode()`.

    Other non-serializable types are only possible through custom behaviors that
    override `__getitem__` (which might return arbitrary Python objects). Use
    `convert_other` to detect these types and convert them.

    See also #ak.from_json.
    """
    with ak._v2._util.OperationErrorContext(
        "ak._v2.to_json",
        dict(
            array=array,
            file=file,
            line_delimited=line_delimited,
            num_indent_spaces=num_indent_spaces,
            num_readability_spaces=num_readability_spaces,
            nan_string=nan_string,
            infinity_string=infinity_string,
            minus_infinity_string=minus_infinity_string,
            complex_record_fields=complex_record_fields,
            convert_bytes=convert_bytes,
            convert_other=convert_other,
        ),
    ):
        return _impl(
            array,
            file,
            line_delimited,
            num_indent_spaces,
            num_readability_spaces,
            nan_string,
            infinity_string,
            minus_infinity_string,
            complex_record_fields,
            convert_bytes,
            convert_other,
        )


def _impl(
    array,
    file,
    line_delimited,
    num_indent_spaces,
    num_readability_spaces,
    nan_string,
    infinity_string,
    minus_infinity_string,
    complex_record_fields,
    convert_bytes,
    convert_other,
):
    if array is None or isinstance(array, (bool, str, bytes, Number)):
        return json.dumps(array)

    elif isinstance(array, bytes):
        return json.dumps(array.decode("utf-8", "surrogateescape"))

    elif isinstance(array, np.ndarray):
        out = ak._v2.contents.NumpyArray(array)

    elif isinstance(array, ak._v2.highlevel.Array):
        out = array.layout

    elif isinstance(array, ak._v2.highlevel.Record):
        out = array.layout.array[array.layout.at : array.layout.at + 1]

    elif isinstance(array, ak._v2.highlevel.ArrayBuilder):
        out = array.snapshot().layout

    elif isinstance(array, ak._v2.record.Record):
        out = array.array[array.at : array.at + 1]

    elif isinstance(array, ak.layout.ArrayBuilder):
        formstr, length, buffers = array.to_buffers()
        form = ak._v2.forms.from_json(formstr)

        out = ak._v2.operations.convert.from_buffers(
            form, length, buffers, highlevel=False
        )

    elif isinstance(array, ak._v2.contents.Content):
        out = array

    else:
        raise ak._v2._util.error(TypeError(f"unrecognized array type: {repr(array)}"))

    jsondata = out.to_json(
        nan_string=nan_string,
        infinity_string=infinity_string,
        minus_infinity_string=minus_infinity_string,
        complex_record_fields=complex_record_fields,
        convert_bytes=convert_bytes,
        behavior=ak._v2._util.behavior_of(array),
    )

    if line_delimited and not ak._v2._util.isstr(line_delimited):
        line_delimited = "\n"

    separators = (
        "," + " " * num_readability_spaces,
        ":" + " " * num_readability_spaces,
    )

    if file is not None:
        if ak._v2._util.isstr(file):
            parsed_url = urlparse(file)
            if parsed_url.scheme == "" or parsed_url.netloc == "":

                def opener():
                    return open(file, "w", encoding="utf8")

            else:
                import fsspec

                def opener():
                    return fsspec.open(file, "w", encoding="utf8")

        else:

            def opener():
                return _NoContextManager(file)

    try:
        if line_delimited:
            if file is None:
                out = []
                for datum in jsondata:
                    out.append(
                        json.dumps(
                            datum,
                            skipkeys=True,
                            ensure_ascii=True,
                            check_circular=False,
                            allow_nan=False,
                            indent=None,
                            separators=separators,
                            default=convert_other,
                            sort_keys=False,
                        )
                    )
                return line_delimited.join(out)

            else:
                with opener() as file:
                    json.dump(
                        datum,
                        file,
                        skipkeys=True,
                        ensure_ascii=True,
                        check_circular=False,
                        allow_nan=False,
                        indent=None,
                        separators=separators,
                        default=convert_other,
                        sort_keys=False,
                    )
                    file.write(line_delimited)

        else:
            if isinstance(array, (ak._v2.highlevel.Record, ak._v2.record.Record)):
                jsondata = jsondata[0]

            if file is None:
                return json.dumps(
                    jsondata,
                    skipkeys=True,
                    ensure_ascii=True,
                    check_circular=False,
                    allow_nan=False,
                    indent=num_indent_spaces,
                    separators=separators,
                    default=convert_other,
                    sort_keys=False,
                )
            else:
                with opener() as file:
                    return json.dump(
                        jsondata,
                        file,
                        skipkeys=True,
                        ensure_ascii=True,
                        check_circular=False,
                        allow_nan=False,
                        indent=num_indent_spaces,
                        separators=separators,
                        default=convert_other,
                        sort_keys=False,
                    )

    except Exception as err:
        raise ak._v2._util.error(err)


class _NoContextManager:
    def __init__(self, file):
        self.file = file

    def __enter__(self):
        return self.file

    def __exit__(self, exception_type, exception_value, exception_traceback):
        pass
