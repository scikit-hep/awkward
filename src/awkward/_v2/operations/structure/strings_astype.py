# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def strings_astype(array, to, highlevel=True, behavior=None):
    raise NotImplementedError


#     """
#     Args:
#         array: Array whose strings should be converted to a new numeric type.
#         to (dtype or dtype specifier): Type to convert the strings into.
#         highlevel (bool): If True, return an #ak.Array; otherwise, return
#             a low-level #ak.layout.Content subclass.
#         behavior (None or dict): Custom #ak.behavior for the output array, if
#             high-level.

#     Converts all strings in the array to a new type, leaving the structure
#     untouched.

#     For example,

#         >>> array = ak.Array(["1", "2", "    3    ", "00004", "-5"])
#         >>> ak.strings_astype(array, np.int32)
#         <Array [1, 2, 3, 4, -5] type='5 * int32'>

#     and

#         >>> array = ak.Array(["1.1", "2.2", "    3.3    ", "00004.4", "-5.5"])
#         >>> ak.strings_astype(array, np.float64)
#         <Array [1.1, 2.2, 3.3, 4.4, -5.5] type='5 * float64'>

#     and finally,

#         >>> array = ak.Array([["1.1", "2.2", "    3.3    "], [], ["00004.4", "-5.5"]])
#         >>> ak.strings_astype(array, np.float64)
#         <Array [[1.1, 2.2, 3.3], [], [4.4, -5.5]] type='3 * var * float64'>

#     See also #ak.numbers_astype.
#     """
#     to_dtype = np.dtype(to)

#     def getfunction(layout):
#         if isinstance(layout, ak._v2._util.listtypes) and (
#             layout.parameter("__array__") == "string"
#             or layout.parameter("__array__") == "bytestring"
#         ):
#             layout = without_parameters(layout, highlevel=False)
#             max_length = ak.max(num(layout))
#             regulararray = layout.rpad_and_clip(max_length, 1)
#             maskedarray = ak._v2.operations.convert.to_numpy(
#                 regulararray, allow_missing=True
#             )
#             npstrings = maskedarray.data
#             if maskedarray.mask is not False:
#                 npstrings[maskedarray.mask] = 0
#             npnumbers = (
#                 npstrings.reshape(-1).view("<S" + str(max_length)).astype(to_dtype)
#             )
#             return lambda: ak._v2.contents.NumpyArray(npnumbers)
#         else:
#             return None

#     layout = ak._v2.operations.convert.to_layout(
#         array, allow_record=False, allow_other=False
#     )
#     out = ak._v2._util.recursively_apply(
#         layout,
#         getfunction,
#         pass_depth=False,
#         pass_user=False,
#     )
#     return ak._v2._util.maybe_wrap_like(out, array, behavior, highlevel)
