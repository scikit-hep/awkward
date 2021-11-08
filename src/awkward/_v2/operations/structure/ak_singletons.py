# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def singletons(array, highlevel=True, behavior=None):
    raise NotImplementedError


#     """
#     Args:
#         array: Data to wrap in lists of length 1 if present and length 0
#             if missing (None).
#         highlevel (bool): If True, return an #ak.Array; otherwise, return
#             a low-level #ak.layout.Content subclass.
#         behavior (None or dict): Custom #ak.behavior for the output array, if
#             high-level.

#     Returns a singleton list (length 1) wrapping each non-missing value and
#     an empty list (length 0) in place of each missing value.

#     For example,

#         >>> array = ak.Array([1.1, 2.2, None, 3.3, None, None, 4.4, 5.5])
#         >>> print(ak.singletons(array))
#         [[1.1], [2.2], [], [3.3], [], [], [4.4], [5.5]]

#     See #ak.firsts to invert this function.
#     """

#     def getfunction(layout):
#         nplike = ak.nplike.of(layout)

#         if isinstance(layout, ak._v2._util.optiontypes):
#             nulls = nplike.asarray(layout.bytemask()).view(np.bool_)
#             offsets = nplike.ones(len(layout) + 1, dtype=np.int64)
#             offsets[0] = 0
#             offsets[1:][nulls] = 0
#             nplike.cumsum(offsets, out=offsets)
#             return lambda: ak._v2.contents.ListOffsetArray64(
#                 ak._v2.index.Index64(offsets), layout.project()
#             )
#         else:
#             return None

#     layout = ak._v2.operations.convert.to_layout(array)
#     out = ak._v2._util.recursively_apply(layout, getfunction, pass_depth=False)

#     return ak._v2._util.maybe_wrap_like(out, array, behavior, highlevel)
