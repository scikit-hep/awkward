# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def validity_error(array, exception=False):
    pass


#     """
#     Args:
#         array (#ak.Array, #ak.Record, #ak.layout.Content, #ak.layout.Record, #ak.ArrayBuilder, #ak.layout.ArrayBuilder):
#             Array or record to check.
#         exception (bool): If True, validity errors raise exceptions.

#     Returns None if there are no errors and a str containing the error message
#     if there are.

#     Checks for errors in the structure of the array, such as indexes that run
#     beyond the length of a node's `content`, etc. Either an error is raised or
#     a string describing the error is returned.

#     See also #ak.is_valid.
#     """
#     if isinstance(array, (ak._v2.highlevel.Array, ak._v2.highlevel.Record)):
#         return validity_error(array.layout, exception=exception)

#     elif isinstance(array, ak._v2.highlevel.ArrayBuilder):
#         return validity_error(array.snapshot().layout, exception=exception)

#     elif isinstance(
#         array,
#         (
#             ak._v2.contents.Content,
#             ak._v2.record.Record,
#             ak.partition.PartitionedArray,   # NO PARTITIONED ARRAY
#         ),
#     ):
#         out = array.validityerror()
#         if out is not None and exception:
#             raise ValueError(out)
#         else:
#             return out

#     elif isinstance(array, ak.layout.ArrayBuilder):
#         return validity_error(array.snapshot(), exception=exception)

#     else:
#         raise TypeError(
#             "not an awkward array: {0}".format(repr(array))
#
#         )
