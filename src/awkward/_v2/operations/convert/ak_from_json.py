# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


# _maybe_json_str = re.compile(r"^\s*(\[|\{|\"|[0-9]|true|false|null)")
# _maybe_json_bytes = re.compile(br"^\s*(\[|\{|\"|[0-9]|true|false|null)")


def from_json(  # note: move ability to read from file into from_json_file
    source,
    nan_string=None,
    infinity_string=None,
    minus_infinity_string=None,
    complex_record_fields=None,
    highlevel=True,
    behavior=None,
    initial=1024,
    resize=1.5,
    buffersize=65536,
):
    raise NotImplementedError


#     """
#     Args:
#         source (str): JSON-formatted string or filename to convert into an array.
#         nan_string (None or str): If not None, strings with this value will be
#             interpreted as floating-point NaN values.
#         infinity_string (None or str): If not None, strings with this value will
#             be interpreted as floating-point positive infinity values.
#         minus_infinity_string (None or str): If not None, strings with this value
#             will be interpreted as floating-point negative infinity values.
#         complex_record_fields (None or (str, str)): If not None, defines a pair of
#             field names to interpret records as complex numbers.
#         highlevel (bool): If True, return an #ak.Array; otherwise, return
#             a low-level #ak.layout.Content subclass.
#         behavior (None or dict): Custom #ak.behavior for the output array, if
#             high-level.
#         initial (int): Initial size (in bytes) of buffers used by
#             #ak.layout.ArrayBuilder (see #ak.layout.ArrayBuilderOptions).
#         resize (float): Resize multiplier for buffers used by
#             #ak.layout.ArrayBuilder (see #ak.layout.ArrayBuilderOptions);
#             should be strictly greater than 1.
#         buffersize (int): Size (in bytes) of the buffer used by the JSON
#             parser.

#     Converts a JSON string into an Awkward Array.

#     Internally, this function uses #ak.layout.ArrayBuilder (see the high-level
#     #ak.ArrayBuilder documentation for a more complete description), so it
#     has the same flexibility and the same constraints. Any heterogeneous
#     and deeply nested JSON can be converted, but the output will never have
#     regular-typed array lengths.

#     See also #ak.from_json_schema and #ak.to_json.
#     """

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

#     is_path, source = ak._v2._util.regularize_path(source)

#     if ak._v2._util.is_file_path(source):
#         layout = ak._ext.fromjsonfile(
#             source,
#             nan_string=nan_string,
#             infinity_string=infinity_string,
#             minus_infinity_string=minus_infinity_string,
#             initial=initial,
#             resize=resize,
#             buffersize=buffersize,
#         )
#     elif not is_path and (
#         (isinstance(source, bytes) and _maybe_json_bytes.match(source))
#         or _maybe_json_str.match(source)
#     ):
#         layout = ak._ext.fromjson(
#             source,
#             nan_string=nan_string,
#             infinity_string=infinity_string,
#             minus_infinity_string=minus_infinity_string,
#             initial=initial,
#             resize=resize,
#             buffersize=buffersize,
#         )
#     else:
#         if ak._v2._util.py27:
#             exc = IOError
#         else:
#             exc = FileNotFoundError
#         raise exc("file not found or not a regular file: {0}".format(source))

#     def getfunction(recordnode):
#         if isinstance(recordnode, ak._v2.contents.RecordArray):
#             keys = recordnode.keys()
#             if complex_record_fields[0] in keys and complex_record_fields[1] in keys:
#                 nplike = ak.nplike.of(recordnode)
#                 real = recordnode[complex_record_fields[0]]
#                 imag = recordnode[complex_record_fields[1]]
#                 if (
#                     isinstance(real, ak._v2.contents.NumpyArray)
#                     and len(real.shape) == 1
#                     and isinstance(imag, ak._v2.contents.NumpyArray)
#                     and len(imag.shape) == 1
#                 ):
#                     return lambda: nplike.asarray(real) + nplike.asarray(imag) * 1j
#                 else:
#                     raise ValueError(
#                         "Complex number fields must be numbers"
#
#                     )
#                 return lambda: ak._v2.contents.NumpyArray(real + imag * 1j)
#             else:
#                 return None
#         else:
#             return None

#     if complex_imag_string is not None:
#         layout = ak._v2._util.recursively_apply(layout, getfunction, pass_depth=False)

#     return ak._v2._util.maybe_wrap(layout, behavior, highlevel)
