# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def without_parameters(array, highlevel=True, behavior=None):
    raise NotImplementedError


#     """
#     Args:
#         array: Data convertible into an Awkward Array.
#         highlevel (bool): If True, return an #ak.Array; otherwise, return
#             a low-level #ak.layout.Content subclass.
#         behavior (None or dict): Custom #ak.behavior for the output array, if
#             high-level.

#     This function returns a new array without any parameters in its
#     #ak.Array.layout, on nodes of any level of depth.

#     Note that a "new array" is a lightweight shallow copy, not a duplication
#     of large data buffers.
#     """
#     layout = ak._v2.operations.convert.to_layout(
#         array, allow_record=True, allow_other=False
#     )

#     out = ak._v2._util.recursively_apply(
#         layout, lambda layout: None, pass_depth=False, keep_parameters=False
#     )

#     return ak._v2._util.maybe_wrap_like(out, array, behavior, highlevel)
