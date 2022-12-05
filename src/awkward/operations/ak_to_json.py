# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import json
import pathlib
from numbers import Number
from urllib.parse import urlparse

from awkward_cpp.lib import _ext

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()
numpy = ak._nplikes.Numpy.instance()


def to_json(
    array,
    file=None,
    *,
    line_delimited=False,
    num_indent_spaces=None,
    num_readability_spaces=0,
    nan_string=None,
    posinf_string=None,
    neginf_string=None,
    complex_record_fields=None,
    convert_bytes=None,
    convert_other=None,
):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        file (None, str/pathlib.Path, or file-like object): If None, this function returns
            JSON-encoded bytes. Otherwise, this function has no return value.
            If a string/pathlib.Path, this function opens a file with that name, writes JSON
            data, and closes the file. If that path has a URI protocol (like
            "https://" or "s3://"), this function attempts to open the file with
            the fsspec library. If a file-like object with a `write` method,
            this function writes to the object, but does not close it.
        line_delimited (bool or str): If False, a single JSON document is written,
            representing the entire array or record. If True, each element of the
            array (or just the one record) is written on a separate line of text,
            separated by `"\\n"`. If a string, such as `"\\r\\n"`, it is taken as a
            custom line delimiter. (Use `os.linesep` for a platform-dependent
            line delimiter.)
        num_indent_spaces (None or nonnegative int): Number of spaces to indent nested
            elements, for pretty-printed JSON. If None, the JSON output is written
            on one line of text. Ignored if `line_delimited` is True or a string.
        num_readability_spaces (nonnegative int): Number of spaces to include after
            commas (`,`) and colons (`:`), for pretty-printed JSON.
        nan_string (None or str): If not None, floating-point NaN values will be
            replaced with this string instead of a JSON number.
        posinf_string (None or str): If not None, floating-point positive infinity
            values will be replaced with this string instead of a JSON number.
        neginf_string (None or str): If not None, floating-point negative infinity
            values will be replaced with this string instead of a JSON number.
        complex_record_fields (None or (str, str)): If not None, defines a pair of
            field names to interpret records as complex numbers, such as
            `("real", "imag")`.
        convert_bytes (None or function): If not None, this function is applied to
            all Python 3 bytes objects to produce something JSON serializable,
            such as a string using UTF-8 or Base64 encoding, lists of integers, etc.
        convert_other (None or function): Passed to `json.dump` or `json.dumps`
            as `default` to convert any other objects that #ak.to_list would return
            but are not JSON serializable.

    Converts `array` (many types supported, including all Awkward Arrays and
    Records) into JSON text. Returns bytes (encoded JSON) if `file` is None;
    otherwise, this function returns nothing and writes to a file.

    This function converts the array into Python objects with #ak.to_list, performs
    some conversions to make the data JSON serializable (`nan_string`,
    `posinf_string`, `neginf_string`, `complex_record_fields`, `convert_bytes`,
    `convert_other`), then uses `json.dumps` to return a string or `json.dump`
    to write to a file (depending on the value of `file`).

    If `line_delimited` is True or a line-delimiter string like `"\\r\\n"`/`os.linesep`,
    the output is line-delimited JSON, variously referred to as "ldjson", "ndjson", and
    "jsonl". (Use an appropriate file extension!)

    To pretty-print the JSON, set `num_indent_spaces=4, num_readability_spaces=1` (for
    example).

    Awkward Array types have the following JSON translations.

    * #ak.types.OptionType: missing values are converted into None.
    * #ak.types.ListType: converted into JSON lists.
    * #ak.types.RegularType: also converted into JSON lists. JSON (and
      Python) forms lose information about the regularity of list lengths.
    * #ak.types.ListType or #ak.types.RegularType with parameter `"__array__"`
      equal to `"string"`: converted into JSON strings.
    * #ak.types.RecordType without field names: converted into JSON
      objects with numbers as strings for keys.
    * #ak.types.RecordType with field names: converted into JSON objects.
    * #ak.types.UnionType: JSON data are naturally heterogeneous.

    If the array contains any NaN (not a number), infinite values, or
    imaginary/complex types, `nan_string`, `posinf_string`, and/or `neginf_string`
    _must_ be supplied.

    If the array contains any raw bytestrings (`"__array__"` equal to `"bytestring"`),
    `convert_bytes` _must_ be supplied. To interpret as strings, use `bytes.decode`.
    To Base64-encode, use `lambda x: base64.b64encode(x).decode()`.

    Other non-serializable types are only possible through custom behaviors that
    override `__getitem__` (which might return arbitrary Python objects). Use
    `convert_other` to detect these types and convert them.

    See also #ak.from_json.
    """
    with ak._errors.OperationErrorContext(
        "ak.to_json",
        dict(
            array=array,
            file=file,
            line_delimited=line_delimited,
            num_indent_spaces=num_indent_spaces,
            num_readability_spaces=num_readability_spaces,
            nan_string=nan_string,
            posinf_string=posinf_string,
            neginf_string=neginf_string,
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
            posinf_string,
            neginf_string,
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
    posinf_string,
    neginf_string,
    complex_record_fields,
    convert_bytes,
    convert_other,
):
    if array is None or isinstance(array, (bool, str, bytes, Number)):
        out = ak.operations.from_iter([array], highlevel=False)

    elif isinstance(array, ak.highlevel.Array):
        out = array.layout

    elif isinstance(array, ak.highlevel.Record):
        out = array.layout.array[array.layout.at : array.layout.at + 1]

    elif isinstance(array, ak.highlevel.ArrayBuilder):
        out = array.snapshot().layout

    elif isinstance(array, ak.record.Record):
        out = array.array[array.at : array.at + 1]

    elif isinstance(array, _ext.ArrayBuilder):
        formstr, length, buffers = array.to_buffers()
        form = ak.forms.from_json(formstr)

        out = ak.operations.from_buffers(form, length, buffers, highlevel=False)

    elif isinstance(array, ak.contents.Content):
        out = array

    elif hasattr(array, "shape") and hasattr(array, "dtype"):
        out = ak.contents.NumpyArray(array)

    else:
        raise ak._errors.wrap_error(
            TypeError(f"unrecognized array type: {repr(array)}")
        )

    jsondata = out.to_json(
        nan_string=nan_string,
        posinf_string=posinf_string,
        neginf_string=neginf_string,
        complex_record_fields=complex_record_fields,
        convert_bytes=convert_bytes,
        behavior=ak._util.behavior_of(array),
    )

    if line_delimited and not isinstance(line_delimited, str):
        line_delimited = "\n"

    separators = (
        "," + " " * num_readability_spaces,
        ":" + " " * num_readability_spaces,
    )

    if file is not None:
        if isinstance(file, (str, pathlib.Path)):
            parsed_url = urlparse(file)
            if parsed_url.scheme == "" or parsed_url.netloc == "":

                def opener():
                    return open(file, "w", encoding="utf8")

            else:
                import fsspec

                def opener():
                    return fsspec.open(file, "w", encoding="utf8").open()

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
                    out.append(line_delimited)
                return "".join(out)

            else:
                with opener() as openfile:
                    for datum in jsondata:
                        json.dump(
                            datum,
                            openfile,
                            skipkeys=True,
                            ensure_ascii=True,
                            check_circular=False,
                            allow_nan=False,
                            indent=None,
                            separators=separators,
                            default=convert_other,
                            sort_keys=False,
                        )
                        openfile.write(line_delimited)

        else:
            if isinstance(array, (ak.highlevel.Record, ak.record.Record)):
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
                with opener() as openfile:
                    return json.dump(
                        jsondata,
                        openfile,
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
        raise ak._errors.wrap_error(err) from err


class _NoContextManager:
    def __init__(self, file):
        self.file = file

    def __enter__(self):
        return self.file

    def __exit__(self, exception_type, exception_value, exception_traceback):
        pass
