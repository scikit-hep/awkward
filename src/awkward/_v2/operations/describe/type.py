# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def type(array):
    raise NotImplementedError


#     """
#     The high-level type of an `array` (many types supported, including all
#     Awkward Arrays and Records) as #ak.types.Type objects.

#     The high-level type ignores #layout differences like
#     #ak.layout.ListArray64 versus #ak.layout.ListOffsetArray64, but
#     not differences like "regular-sized lists" (i.e.
#     #ak.layout.RegularArray) versus "variable-sized lists" (i.e.
#     #ak.layout.ListArray64 and similar).

#     Types are rendered as [Datashape](https://datashape.readthedocs.io/)
#     strings, which makes the same distinctions.

#     For example,

#         ak.Array([[{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}],
#                   [],
#                   [{"x": 3.3, "y": [3, 3, 3]}]])

#     has type

#         3 * var * {"x": float64, "y": var * int64}

#     but

#         ak.Array(np.arange(2*3*5).reshape(2, 3, 5))

#     has type

#         2 * 3 * 5 * int64

#     Some cases, like heterogeneous data, require [extensions beyond the
#     Datashape specification](https://github.com/blaze/datashape/issues/237).
#     For example,

#         ak.Array([1, "two", [3, 3, 3]])

#     has type

#         3 * union[int64, string, var * int64]

#     but "union" is not a Datashape type-constructor. (Its syntax is
#     similar to existing type-constructors, so it's a plausible addition
#     to the language.)
#     """
#     if array is None:
#         return ak._v2.types.UnknownType()

#     elif isinstance(array, (bool, np.bool_)):
#         return ak._v2.types.PrimitiveType("bool")

#     elif isinstance(array, numbers.Integral):
#         return ak._v2.types.PrimitiveType("int64")

#     elif isinstance(array, numbers.Real):
#         return ak._v2.types.PrimitiveType("float64")

#     elif isinstance(
#         array,
#         (
#             np.int8,
#             np.int16,
#             np.int32,
#             np.int64,
#             np.uint8,
#             np.uint16,
#             np.uint32,
#             np.uint64,
#             np.float32,
#             np.float64,
#             np.complex64,
#             np.complex128,
#             np.datetime64,
#             np.timedelta64,
#         ),
#     ):
#         return ak._v2.types.PrimitiveType(type.dtype2primitive[array.dtype.type])

#     elif isinstance(array, ak._v2.highlevel.Array):
#         return ak._v2._util.highlevel_type(array.layout, array.behavior, True)

#     elif isinstance(array, ak._v2.highlevel.Record):
#         return ak._v2._util.highlevel_type(array.layout, array.behavior, False)

#     elif isinstance(array, ak._v2.highlevel.ArrayBuilder):
#         return ak._v2._util.highlevel_type(array.snapshot().layout, array.behavior, True)

#     elif isinstance(array, ak._v2.record.Record):
#         return array.type(ak._v2._util.typestrs(None))

#     elif isinstance(array, np.ndarray):
#         if len(array.shape) == 0:
#             return type(array.reshape((1,))[0])
#         else:
#             try:
#                 out = type.dtype2primitive[array.dtype.type]
#             except KeyError:
#                 raise TypeError(
#                     "numpy array type is unrecognized by awkward: %r" % array.dtype.type
#                 )
#             out = ak._v2.types.PrimitiveType(out)
#             for x in array.shape[-1:0:-1]:
#                 out = ak._v2.types.RegularType(out, x)
#             return ak._v2.types.ArrayType(out, array.shape[0])

#     elif isinstance(array, ak.layout.ArrayBuilder):
#         return array.type(ak._v2._util.typestrs(None))

#     elif isinstance(array, (ak._v2.contents.Content, ak.partition.PartitionedArray)):   # NO PARTITIONED ARRAY
#         return array.type(ak._v2._util.typestrs(None))

#     else:
#         raise TypeError(
#             "unrecognized array type: {0}".format(repr(array))
#
#         )


# type.dtype2primitive = {
#     np.bool_: "bool",
#     np.int8: "int8",
#     np.int16: "int16",
#     np.int32: "int32",
#     np.int64: "int64",
#     np.uint8: "uint8",
#     np.uint16: "uint16",
#     np.uint32: "uint32",
#     np.uint64: "uint64",
#     np.float32: "float32",
#     np.float64: "float64",
#     np.complex64: "complex64",
#     np.complex128: "complex128",
#     np.datetime64: "datetime64",
#     np.timedelta64: "timedelta64",
# }
