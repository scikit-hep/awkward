# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def to_cupy(array):
    raise NotImplementedError


#     """
#     Converts `array` (many types supported) into a CuPy array, if possible.

#     If the data are numerical and regular (nested lists have equal lengths
#     in each dimension, as described by the #type), they can be losslessly
#     converted to a CuPy array and this function returns without an error.

#     Otherwise, the function raises an error.

#     If `array` is a scalar, it is converted into a CuPy scalar.

#     See also #ak.from_cupy and #ak.to_numpy.
#     """
#     cupy = ak.nplike.Cupy.instance()
#     np = ak.nplike.NumpyMetadata.instance()

#     if isinstance(array, (bool, numbers.Number)):
#         return cupy.array([array])[0]

#     elif isinstance(array, cupy.ndarray):
#         return array

#     elif isinstance(array, np.ndarray):
#         return cupy.asarray(array)

#     elif isinstance(array, ak._v2.highlevel.Array):
#         return to_cupy(array.layout)

#     elif isinstance(array, ak._v2.highlevel.Record):
#         raise ValueError(
#             "CuPy does not support record structures"
#
#         )

#     elif isinstance(array, ak._v2.highlevel.ArrayBuilder):
#         return to_cupy(array.snapshot().layout)

#     elif isinstance(array, ak.layout.ArrayBuilder):
#         return to_cupy(array.snapshot())

#     elif (
#         ak._v2.operations.describe.parameters(array).get("__array__") == "bytestring"
#         or ak._v2.operations.describe.parameters(array).get("__array__") == "string"
#     ):
#         raise ValueError(
#             "CuPy does not support arrays of strings"
#
#         )

#     elif isinstance(array, ak.partition.PartitionedArray):   # NO PARTITIONED ARRAY
#         return cupy.concatenate([to_cupy(x) for x in array.partitions])

#     elif isinstance(array, ak._v2._util.virtualtypes):
#         return to_cupy(array.array)

#     elif isinstance(array, ak._v2._util.unknowntypes):
#         return cupy.array([])

#     elif isinstance(array, ak._v2._util.indexedtypes):
#         return to_cupy(array.project())

#     elif isinstance(array, ak._v2._util.uniontypes):
#         contents = [to_cupy(array.project(i)) for i in range(array.numcontents)]
#         out = cupy.concatenate(contents)

#         tags = cupy.asarray(array.tags)
#         for tag, content in enumerate(contents):
#             mask = tags == tag
#             out[mask] = content
#         return out

#     elif isinstance(array, ak._v2.contents.UnmaskedArray):
#         return to_cupy(array.content)

#     elif isinstance(array, ak._v2._util.optiontypes):
#         content = to_cupy(array.project())

#         shape = list(content.shape)
#         shape[0] = len(array)
#         mask0 = cupy.asarray(array.bytemask()).view(np.bool_)
#         if mask0.any():
#             raise ValueError(
#                 "CuPy does not support masked arrays"
#
#             )
#         else:
#             return content

#     elif isinstance(array, ak._v2.contents.RegularArray):
#         out = to_cupy(array.content)
#         head, tail = out.shape[0], out.shape[1:]
#         shape = (head // array.size, array.size) + tail
#         return out[: shape[0] * array.size].reshape(shape)

#     elif isinstance(array, ak._v2._util.listtypes):
#         return to_cupy(array.toRegularArray())

#     elif isinstance(array, ak._v2._util.recordtypes):
#         raise ValueError(
#             "CuPy does not support record structures"
#
#         )

#     elif isinstance(array, ak._v2.contents.NumpyArray):
#         return array.to_cupy()

#     elif isinstance(array, ak._v2.contents.Content):
#         raise AssertionError(
#             "unrecognized Content type: {0}".format(type(array))
#
#         )

#     elif isinstance(array, Iterable):
#         return cupy.asarray(array)

#     else:
#         raise ValueError(
#             "cannot convert {0} into cp.ndarray".format(array)
#
#         )
