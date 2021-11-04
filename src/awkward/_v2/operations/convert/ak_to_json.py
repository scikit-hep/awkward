# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def to_json(
    array,
    destination=None,
    pretty=False,
    maxdecimals=None,
    nan_string=None,
    infinity_string=None,
    minus_infinity_string=None,
    complex_record_fields=None,
    buffersize=65536,
):  # note: move ability to write to file into to_json_file
    raise NotImplementedError


#     """
#     Args:
#         array: Data to convert to JSON.
#         destination (None or str): If None, this function returns a JSON str;
#             if a str, it uses that as a file name and writes (overwrites) that
#             file (returning None).
#         pretty (bool): If True, indent the output for human readability; if
#             False, output compact JSON without spaces.
#         maxdecimals (None or int): If an int, limit the number of
#             floating-point decimals to this number; if None, write all digits.
#         nan_string (None or str): If not None, floating-point NaN values will be
#             replaced with this string instead of a JSON number.
#         infinity_string (None or str): If not None, floating-point positive infinity
#             values will be replaced with this string instead of a JSON number.
#         minus_infinity_string (None or str): If not None, floating-point negative
#             infinity values will be replaced with this string instead of a JSON
#             number.
#         complex_record_fields (None or (str, str)): If not None, defines a pair of
#             field names to interpret records as complex numbers.
#         buffersize (int): Size (in bytes) of the buffer used by the JSON
#             parser.

#     Converts `array` (many types supported, including all Awkward Arrays and
#     Records) into a JSON string or file.

#     Awkward Array types have the following JSON translations.

#        * #ak.types.PrimitiveType: converted into JSON booleans and numbers.
#        * #ak.types.OptionType: missing values are converted into None.
#        * #ak.types.ListType: converted into JSON lists.
#        * #ak.types.RegularType: also converted into JSON lists. JSON (and
#          Python) forms lose information about the regularity of list lengths.
#        * #ak.types.ListType with parameter `"__array__"` equal to
#          `"__bytestring__"` or `"__string__"`: converted into JSON strings.
#        * #ak.types.RecordArray without field names: converted into JSON
#          objects with numbers as strings for keys.
#        * #ak.types.RecordArray with field names: converted into JSON objects.
#        * #ak.types.UnionArray: JSON data are naturally heterogeneous.

#     See also #ak.from_json and #ak.Array.tojson.
#     """
#     if array is None or isinstance(array, (bool, str, bytes, numbers.Number)):
#         return json.dumps(array)

#     elif isinstance(array, bytes):
#         return json.dumps(array.decode("utf-8", "surrogateescape"))

#     elif ak._v2._util.py27 and isinstance(array, ak._v2._util.unicode):
#         return json.dumps(array)

#     elif isinstance(array, np.ndarray):
#         out = ak._v2.contents.NumpyArray(array)

#     elif isinstance(array, ak._v2.highlevel.Array):
#         out = array.layout

#     elif isinstance(array, ak._v2.highlevel.Record):
#         out = array.layout

#     elif isinstance(array, ak._v2.highlevel.ArrayBuilder):
#         out = array.snapshot().layout

#     elif isinstance(array, ak._v2.record.Record):
#         out = array

#     elif isinstance(array, ak.layout.ArrayBuilder):
#         out = array.snapshot()

#     elif isinstance(array, (ak._v2.contents.Content, ak.partition.PartitionedArray)):   # NO PARTITIONED ARRAY
#         out = array

#     elif isinstance(array, ak._v2.contents.Content):
#         out = ak._v2.tmp_for_testing.v2_to_v1(array)

#     else:
#         raise TypeError(
#             "unrecognized array type: {0}".format(repr(array))
#
#         )

#     if complex_record_fields is None:
#         complex_real_string = None
#         complex_imag_string = None
#     elif (
#         isinstance(complex_record_fields, tuple)
#         and len(complex_record_fields) == 2
#         and isinstance(complex_record_fields[0], str)
#         and isinstance(complex_record_fields[1], str)
#     ):
#         complex_real_string, complex_imag_string = complex_record_fields

#     if destination is None:
#         return out.tojson(
#             pretty=pretty,
#             maxdecimals=maxdecimals,
#             nan_string=nan_string,
#             infinity_string=infinity_string,
#             minus_infinity_string=minus_infinity_string,
#             complex_real_string=complex_real_string,
#             complex_imag_string=complex_imag_string,
#         )
#     else:
#         return out.tojson(
#             destination,
#             pretty=pretty,
#             maxdecimals=maxdecimals,
#             buffersize=buffersize,
#             nan_string=nan_string,
#             infinity_string=infinity_string,
#             minus_infinity_string=minus_infinity_string,
#             complex_real_string=complex_real_string,
#             complex_imag_string=complex_imag_string,
#         )
