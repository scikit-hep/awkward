# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


# @ak._v2._connect.numpy.implements("ravel")
def ravel(array, highlevel=True, behavior=None):
    raise NotImplementedError


#     """
#     Args:
#         array: Data containing nested lists to flatten
#         highlevel (bool): If True, return an #ak.Array; otherwise, return
#             a low-level #ak.layout.Content subclass.
#         behavior (None or dict): Custom #ak.behavior for the output array, if
#             high-level.

#     Returns an array with all level of nesting removed by erasing the
#     boundaries between consecutive lists.

#     This is the equivalent of NumPy's `np.ravel` for Awkward Arrays.

#     Consider the following doubly nested `array`.

#         ak.Array([[
#                    [1.1, 2.2, 3.3],
#                    [],
#                    [4.4, 5.5],
#                    [6.6]],
#                   [],
#                   [
#                    [7.7],
#                    [8.8, 9.9]
#                   ]])

#     Ravelling the array produces a flat array

#         >>> print(ak.ravel(array))
#         [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]

#     Missing values are eliminated by flattening: there is no distinction
#     between an empty list and a value of None at the level of flattening.
#     """
#     layout = ak._v2.operations.convert.to_layout(
#         array, allow_record=False, allow_other=False
#     )
#     nplike = ak.nplike.of(layout)

#     out = ak._v2._util.completely_flatten(layout)
#     assert isinstance(out, tuple) and all(isinstance(x, np.ndarray) for x in out)

#     if any(isinstance(x, nplike.ma.MaskedArray) for x in out):
#         out = ak._v2.contents.NumpyArray(nplike.ma.concatenate(out))
#     else:
#         out = ak._v2.contents.NumpyArray(nplike.concatenate(out))

#     return ak._v2._util.maybe_wrap_like(out, array, behavior, highlevel)
