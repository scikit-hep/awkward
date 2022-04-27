# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak
import json
from numbers import Number

np = ak.nplike.NumpyMetadata.instance()


def to_json_file(
    array,
    destination=None,
    pretty=False,
    maxdecimals=None,
    nan_string=None,
    infinity_string=None,
    minus_infinity_string=None,
    complex_record_fields=None,
    convert_bytes=None,
):
    """
    Args:
        array: Data to convert to JSON.
        destination (str): a file name to write to (overwrite) that
            file (returning None).
        pretty (bool): If True, indent the output for human readability; if
            False, output compact JSON without spaces.
        maxdecimals (None or int): If an int, limit the number of
            floating-point decimals to this number; if None, write all digits.
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
            such as a string using UTF-8 or Base-64 encoding, lists of integers, etc.

    Converts `array` (many types supported, including all Awkward Arrays and
    Records) into a JSON file.

    Awkward Array types have the following JSON translations.

       * #ak.types.PrimitiveType: converted into JSON booleans and numbers.
       * #ak.types.OptionType: missing values are converted into None.
       * #ak.types.ListType: converted into JSON lists.
       * #ak.types.RegularType: also converted into JSON lists. JSON (and
         Python) forms lose information about the regularity of list lengths.
       * #ak.types.ListType with parameter `"__array__"` equal to
         `"__bytestring__"` or `"__string__"`: converted into JSON strings.
       * #ak.types.RecordArray without field names: converted into JSON
         objects with numbers as strings for keys.
       * #ak.types.RecordArray with field names: converted into JSON objects.
       * #ak.types.UnionArray: JSON data are naturally heterogeneous.

    See also #ak.from_json.
    """
    with ak._v2._util.OperationErrorContext(
        "ak._v2.to_json_file",
        dict(
            array=array,
            destination=destination,
            pretty=pretty,
            maxdecimals=maxdecimals,
            nan_string=nan_string,
            infinity_string=infinity_string,
            minus_infinity_string=minus_infinity_string,
            complex_record_fields=complex_record_fields,
            convert_bytes=convert_bytes,
        ),
    ):
        return _impl(
            array,
            destination,
            pretty,
            maxdecimals,
            nan_string,
            infinity_string,
            minus_infinity_string,
            complex_record_fields,
            convert_bytes,
        )


def _impl(
    array,
    destination,
    pretty,
    maxdecimals,
    nan_string,
    infinity_string,
    minus_infinity_string,
    complex_record_fields,
    convert_bytes,
):
    if array is None or isinstance(array, (bool, str, bytes, Number)):
        return json.dump(array)

    elif isinstance(array, bytes):
        return json.dump(array.decode("utf-8", "surrogateescape"))

    elif isinstance(array, np.ndarray):
        out = ak._v2.contents.NumpyArray(array)

    elif isinstance(array, ak._v2.highlevel.Array):
        out = array.layout

    elif isinstance(array, ak._v2.highlevel.Record):
        out = array.layout

    elif isinstance(array, ak._v2.highlevel.ArrayBuilder):
        out = array.snapshot().layout

    elif isinstance(array, ak._v2.record.Record):
        out = array

    elif isinstance(array, ak.layout.ArrayBuilder):
        formstr, length, buffers = array.to_buffers()
        form = ak._v2.forms.from_json(formstr)

        out = ak._v2.operations.convert.from_buffers(
            form, length, buffers, highlevel=False
        )
        # FIXME: the code is a copy from snapshot,
        # because this call returns v1:
        # out = array.snapshot()

    elif isinstance(array, ak._v2.contents.Content):
        out = array

    else:
        raise ak._v2._util.error(TypeError(f"unrecognized array type: {repr(array)}"))

    with open(destination, "w", encoding="utf-8") as file:
        jsondata = out.to_json(
            nan_string=nan_string,
            infinity_string=infinity_string,
            minus_infinity_string=minus_infinity_string,
            complex_record_fields=complex_record_fields,
            convert_bytes=convert_bytes,
            behavior=ak._v2._util.behavior_of(array),
        )
        try:
            json.dump(jsondata, file, separators=(",", ":"))
        except Exception as err:
            raise ak._v2._util.error(err)
