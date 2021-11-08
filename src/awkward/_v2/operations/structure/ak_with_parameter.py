# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def with_parameter(array, parameter, value, highlevel=True, behavior=None):
    raise NotImplementedError


#     """
#     Args:
#         array: Data convertible into an Awkward Array.
#         parameter (str): Name of the parameter to set on that array.
#         value (JSON): Value of the parameter to set on that array.
#         highlevel (bool): If True, return an #ak.Array; otherwise, return
#             a low-level #ak.layout.Content subclass.
#         behavior (None or dict): Custom #ak.behavior for the output array, if
#             high-level.

#     This function returns a new array with a parameter set on the outermost
#     node of its #ak.Array.layout.

#     Note that a "new array" is a lightweight shallow copy, not a duplication
#     of large data buffers.

#     You can also remove a single parameter with this function, since setting
#     a parameter to None is equivalent to removing it.
#     """
#     layout = ak._v2.operations.convert.to_layout(
#         array, allow_record=True, allow_other=False
#     )

#     if isinstance(layout, ak.partition.PartitionedArray):   # NO PARTITIONED ARRAY
#         out = layout.replace_partitions(
#             x.withparameter(parameter, value) for x in layout.partitions
#         )
#     else:
#         out = layout.withparameter(parameter, value)

#     return ak._v2._util.maybe_wrap_like(out, array, behavior, highlevel)
