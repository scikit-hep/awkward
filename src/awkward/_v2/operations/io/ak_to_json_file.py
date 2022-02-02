# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import awkward as ak
import json
from numbers import Number

np = ak.nplike.NumpyMetadata.instance()


class ComplexEncoder(json.JSONEncoder):
    def __init__(
        self,
        nan_string=None,
        infinity_string=None,
        minus_infinity_string=None,
        complex_real_string=None,
        complex_imag_string=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.nan_string = "NaN" if nan_string is None else ('"' + nan_string + '"')
        self.infinity_string = (
            "Infinity" if infinity_string is None else ('"' + infinity_string + '"')
        )
        self.minus_infinity_string = (
            "-Infinity"
            if minus_infinity_string is None
            else ('"' + minus_infinity_string + '"')
        )
        self.complex_real_string = complex_real_string
        self.complex_imag_string = complex_imag_string

    def iterencode(self, o, _one_shot=False):
        """Encode the given object and yield each string
        representation as available.

        For example::

            for chunk in JSONEncoder().iterencode(bigobject):
                mysocket.write(chunk)

        """
        if self.check_circular:
            markers = {}
        else:
            markers = None
        if self.ensure_ascii:
            _encoder = json.encoder.encode_basestring_ascii
        else:
            _encoder = json.encoder.encode_basestring

        def floatstr(
            o,
            allow_nan=self.allow_nan,
            _repr=float.__repr__,
            _inf=json.encoder.INFINITY,
            _neginf=-json.encoder.INFINITY,
        ):
            # Check for specials.  Note that this type of test is processor
            # and/or platform-specific, so do tests which don't depend on the
            # internals.

            if o != o:
                text = self.nan_string
            elif o == _inf:
                text = self.infinity_string
            elif o == _neginf:
                text = self.minus_infinity_string
            else:
                return _repr(o)

            if not allow_nan:
                raise ValueError(
                    "Out of range float values are not JSON compliant: " + repr(o)
                )

            return text

        _iterencode = json.encoder._make_iterencode(
            markers,
            self.default,
            _encoder,
            self.indent,
            floatstr,
            self.key_separator,
            self.item_separator,
            self.sort_keys,
            self.skipkeys,
            _one_shot,
        )
        return _iterencode(o, 0)

    def default(self, z):
        if isinstance(z, complex):
            if self.complex_real_string is None or self.complex_imag_string is None:
                raise ValueError(
                    "Complex numbers can't be converted to JSON without"
                    " setting 'complex_record_fields' "
                )
            return {self.complex_real_string: z.real, self.complex_imag_string: z.imag}
        else:
            super().default(z)


def to_json_file(
    array,
    destination=None,
    pretty=False,
    maxdecimals=None,
    nan_string=None,
    infinity_string=None,
    minus_infinity_string=None,
    complex_record_fields=None,
    buffersize=65536,
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
        buffersize (int): Size (in bytes) of the buffer used by the JSON
            parser.

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

    See also #ak.from_json and #ak.Array.tojson.
    """
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
        raise TypeError(f"unrecognized array type: {repr(array)}")

    if complex_record_fields is None:
        complex_real_string = None
        complex_imag_string = None
    elif (
        isinstance(complex_record_fields, tuple)
        and len(complex_record_fields) == 2
        and isinstance(complex_record_fields[0], str)
        and isinstance(complex_record_fields[1], str)
    ):
        complex_real_string, complex_imag_string = complex_record_fields

    with open(destination, "w", encoding="utf-8") as f:
        for chunk in json.dumps(
            out.tojson(
                nan_string,
                infinity_string,
                minus_infinity_string,
                complex_real_string,
                complex_imag_string,
            ),
            separators=(",", ":"),
        ):
            f.write(chunk)
