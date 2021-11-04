# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def with_name(array, name, highlevel=True, behavior=None):
    raise NotImplementedError


#     """
#     Args:
#         base: Data containing records or tuples.
#         name (str): Name to give to the records or tuples; this assigns
#             the `"__record__"` parameter.
#         highlevel (bool): If True, return an #ak.Array; otherwise, return
#             a low-level #ak.layout.Content subclass.
#         behavior (None or dict): Custom #ak.behavior for the output array, if
#             high-level.

#     Returns an #ak.Array or #ak.Record (or low-level equivalent, if
#     `highlevel=False`) with a new name. This function does not change the
#     array in-place.

#     The records or tuples may be nested within multiple levels of nested lists.
#     If records are nested within records, only the outermost are affected.

#     Setting the `"__record__"` parameter makes it possible to add behaviors
#     to the data; see #ak.Array and #ak.behavior for a more complete
#     description.
#     """

#     def getfunction(layout):
#         if isinstance(layout, ak._v2.contents.RecordArray):
#             parameters = dict(layout.parameters)
#             parameters["__record__"] = name
#             return lambda: ak._v2.contents.RecordArray(
#                 layout.contents,
#                 layout.recordlookup,
#                 len(layout),
#                 layout.identities,
#                 parameters,
#             )
#         else:
#             return None

#     out = ak._v2._util.recursively_apply(
#         ak._v2.operations.convert.to_layout(array), getfunction, pass_depth=False
#     )

#     def getfunction2(layout):
#         if isinstance(layout, ak._v2._util.uniontypes):
#             return lambda: layout.simplify(merge=True, mergebool=False)
#         else:
#             return None

#     out2 = ak._v2._util.recursively_apply(out, getfunction2, pass_depth=False)

#     return ak._v2._util.maybe_wrap_like(out2, array, behavior, highlevel)
