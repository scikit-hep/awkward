# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


# @ak._connect._numpy.implements("isclose")
def isclose(
    a, b, rtol=1e-05, atol=1e-08, equal_nan=False, highlevel=True, behavior=None
):
    pass


#     """
#     Args:
#         a: First array to compare.
#         b: Second array to compare.
#         rtol (float): The relative tolerance parameter.
#         atol (float): The absolute tolerance parameter.
#         equal_nan (bool): Whether to compare `NaN` as equal. If True, `NaN` in `a`
#             will be considered equal to `NaN` in `b`.
#         highlevel (bool): If True, return an #ak.Array; otherwise, return
#             a low-level #ak.layout.Content subclass.
#         behavior (None or dict): Custom #ak.behavior for the output array, if
#             high-level.

#     Implements [np.isclose](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html)
#     for Awkward Arrays.
#     """
#     one = ak.operations.convert.to_layout(a)
#     two = ak.operations.convert.to_layout(b)
#     nplike = ak.nplike.of(one, two)

#     def getfunction(inputs):
#         if isinstance(inputs[0], ak._v2.contents.NumpyArray) and isinstance(
#             inputs[1], ak._v2.contents.NumpyArray
#         ):
#             return lambda: (
#                 ak._v2.contents.NumpyArray(
#                     nplike.isclose(
#                         nplike.asarray(inputs[0]),
#                         nplike.asarray(inputs[1]),
#                         rtol=rtol,
#                         atol=atol,
#                         equal_nan=equal_nan,
#                     )
#                 ),
#             )
#         else:
#             return None

#     behavior = ak._v2._util.behaviorof(one, two, behavior=behavior)
#     out = ak._v2._util.broadcast_and_apply(
#         [one, two], getfunction, behavior, pass_depth=False
#     )
#     assert isinstance(out, tuple) and len(out) == 1
#     result = out[0]

#     return ak._v2._util.maybe_wrap(result, behavior, highlevel)
