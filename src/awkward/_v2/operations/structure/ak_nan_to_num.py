# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


# @ak._v2._connect.numpy.implements("nan_to_num")
def nan_to_num(
    array, copy=True, nan=0.0, posinf=None, neginf=None, highlevel=True, behavior=None
):
    raise NotImplementedError


#     """
#     Args:
#         array: Array whose `NaN` values should be converted to a number.
#         copy (bool): Ignored (Awkward Arrays are immutable).
#         nan (int, float, broadcastable array): Value to be used to fill `NaN` values.
#         posinf (None, int, float, broadcastable array): Value to be used to fill positive infinity
#             values. If None, positive infinities are replaced with a very large number.
#         neginf (None, int, float, broadcastable array): Value to be used to fill negative infinity
#             values. If None, negative infinities are replaced with a very small number.
#         highlevel (bool): If True, return an #ak.Array; otherwise, return
#             a low-level #ak.layout.Content subclass.
#         behavior (None or dict): Custom #ak.behavior for the output array, if
#             high-level.

#     Implements [np.nan_to_num](https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html)
#     for Awkward Arrays.
#     """
#     behavior = ak._util.behaviorof(array, behavior=behavior)

#     broadcasting_ids = {}
#     broadcasting = []

#     layout = ak.operations.convert.to_layout(array)
#     broadcasting.append(layout)

#     nan_layout = ak.operations.convert.to_layout(nan, allow_other=True)
#     if isinstance(nan_layout, ak.layout.Content):
#         broadcasting_ids[id(nan)] = len(broadcasting)
#         broadcasting.append(nan_layout)

#     posinf_layout = ak.operations.convert.to_layout(posinf, allow_other=True)
#     if isinstance(posinf_layout, ak.layout.Content):
#         broadcasting_ids[id(posinf)] = len(broadcasting)
#         broadcasting.append(posinf_layout)

#     neginf_layout = ak.operations.convert.to_layout(neginf, allow_other=True)
#     if isinstance(neginf_layout, ak.layout.Content):
#         broadcasting_ids[id(neginf)] = len(broadcasting)
#         broadcasting.append(neginf_layout)

#     nplike = ak.nplike.of(layout)

#     if len(broadcasting) == 1:

#         def getfunction(layout):
#             if isinstance(layout, ak.layout.NumpyArray):
#                 return lambda: ak.layout.NumpyArray(
#                     nplike.nan_to_num(
#                         nplike.asarray(layout),
#                         nan=nan,
#                         posinf=posinf,
#                         neginf=neginf,
#                     )
#                 )
#             else:
#                 return None

#         out = ak._util.recursively_apply(
#             layout, getfunction, pass_depth=False, pass_user=False
#         )

#     else:

#         def getfunction(inputs):
#             if all(isinstance(x, ak.layout.NumpyArray) for x in inputs):
#                 tmp_layout = nplike.asarray(inputs[0])
#                 if id(nan) in broadcasting_ids:
#                     tmp_nan = nplike.asarray(inputs[broadcasting_ids[id(nan)]])
#                 else:
#                     tmp_nan = nan
#                 if id(posinf) in broadcasting_ids:
#                     tmp_posinf = nplike.asarray(inputs[broadcasting_ids[id(posinf)]])
#                 else:
#                     tmp_posinf = posinf
#                 if id(neginf) in broadcasting_ids:
#                     tmp_neginf = nplike.asarray(inputs[broadcasting_ids[id(neginf)]])
#                 else:
#                     tmp_neginf = neginf
#                 return lambda: (ak.layout.NumpyArray(
#                     nplike.nan_to_num(
#                         tmp_layout,
#                         nan=tmp_nan,
#                         posinf=tmp_posinf,
#                         neginf=tmp_neginf,
#                     )
#                 ),)
#             else:
#                 return None

#         out = ak._util.broadcast_and_apply(
#             broadcasting,
#             getfunction,
#             behavior,
#             pass_depth=False,
#         )
#         assert isinstance(out, tuple) and len(out) == 1
#         out = out[0]

#     return ak._util.maybe_wrap(out, behavior, highlevel)
