# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def from_jax(array, regulararray=False, highlevel=True, behavior=None):
    raise NotImplementedError


#     """
#     Args:
#         array (jax.numpy.array): The `jax.numpy.array` to convert into an Awkward Array.
#         regulararray (bool): If True and the array is multidimensional,
#             the dimensions are represented by nested #ak.layout.RegularArray
#             nodes; if False and the array is multidimensional, the dimensions
#             are represented by a multivalued #ak.layout.NumpyArray.shape.
#             If the array is one-dimensional, this has no effect.
#         highlevel (bool): If True, return an #ak.Array; otherwise, return
#             a low-level #ak.layout.Content subclass.
#         behavior (None or dict): Custom #ak.behavior for the output array, if
#             high-level.

#     Converts a JAX array into an Awkward Array.

#     The resulting layout may involve the following #ak.layout.Content types
#     (only):

#        * #ak.layout.NumpyArray
#        * #ak.layout.RegularArray if `regulararray=True`.

#     See also #ak.from_cupy and #ak.from_numpy.
#     """

#     def recurse(array):
#         if regulararray and len(array.shape) > 1:
#             return ak._v2.contents.RegularArray(
#                 recurse(array.reshape((-1,) + array.shape[2:])),
#                 array.shape[1],
#                 array.shape[0],
#             )

#         if len(array.shape) == 0:
#             data = ak._v2.contents.NumpyArray.from_jax(array.reshape(1))
#         else:
#             data = ak._v2.contents.NumpyArray.from_jax(array)

#         return data

#     layout = recurse(array)

#     return ak._v2._util.maybe_wrap(layout, behavior, highlevel)
