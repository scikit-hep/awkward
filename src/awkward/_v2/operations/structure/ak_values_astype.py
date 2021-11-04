# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def values_astype(array, to, highlevel=True, behavior=None):
    raise NotImplementedError


#     """
#     Args:
#         array: Array whose numbers should be converted to a new numeric type.
#         to (dtype or dtype specifier): Type to convert the numbers into.
#         highlevel (bool): If True, return an #ak.Array; otherwise, return
#             a low-level #ak.layout.Content subclass.
#         behavior (None or dict): Custom #ak.behavior for the output array, if
#             high-level.

#     Converts all numbers in the array to a new type, leaving the structure
#     untouched.

#     For example,

#         >>> array = ak.Array([1.1, 2.2, 3.3, 4.4, 5.5])
#         >>> ak.values_astype(array, np.int32)
#         <Array [1, 2, 3, 4, 5] type='5 * int32'>

#     and

#         >>> array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
#         >>> ak.values_astype(array, np.int32)
#         <Array [[1, 2, 3], [], [4, 5]] type='3 * var * int32'>

#     Note, when converting values to a `np.datetime64` type that is unitless, a
#     default '[us]' unit is assumed - until further specified as numpy dtypes.

#     For example,

#         >>> array = ak.Array([1567416600000])
#         >>> ak.values_astype(array, "datetime64[ms]")
#         <Array [2019-09-02T09:30:00.000] type='1 * datetime64'>

#     or

#         >>> array = ak.Array([1567416600000])
#         >>> ak.values_astype(array, np.dtype("M8[ms]"))
#         <Array [2019-09-02T09:30:00.000] type='1 * datetime64'>

#     See also #ak.strings_astype.
#     """
#     to_dtype = np.dtype(to)
#     to_str = _dtype_to_string.get(to_dtype)
#     if to_str is None:
#         if to_dtype.name.startswith("datetime64"):
#             to_str = to_dtype.name
#         else:
#             raise ValueError(
#                 "cannot use {0} to cast the numeric type of an array".format(to_dtype)
#
#             )

#     layout = ak._v2.operations.convert.to_layout(
#         array, allow_record=False, allow_other=False
#     )
#     out = layout.numbers_to_type(to_str)

#     return ak._v2._util.maybe_wrap_like(out, array, behavior, highlevel)


# _dtype_to_string = {
#     np.dtype(np.bool_): "bool",
#     np.dtype(np.int8): "int8",
#     np.dtype(np.int16): "int16",
#     np.dtype(np.int32): "int32",
#     np.dtype(np.int64): "int64",
#     np.dtype(np.uint8): "uint8",
#     np.dtype(np.uint16): "uint16",
#     np.dtype(np.uint32): "uint32",
#     np.dtype(np.uint64): "uint64",
#     np.dtype(np.float32): "float32",
#     np.dtype(np.float64): "float64",
#     np.dtype(np.complex64): "complex64",
#     np.dtype(np.complex128): "complex128",
#     np.dtype(np.datetime64): "datetime64",
#     np.dtype(np.timedelta64): "timedelta64",
# }

# if hasattr(np, "float16"):
#     _dtype_to_string[np.dtype(np.float16)] = "float16"
# if hasattr(np, "float128"):
#     _dtype_to_string[np.dtype(np.float128)] = "float128"
# if hasattr(np, "complex256"):
#     _dtype_to_string[np.dtype(np.complex256)] = "complex256"
