# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


# @ak._connect._numpy.implements("prod")
def prod(array, axis=None, keepdims=False, mask_identity=False):
    pass


#     """
#     Args:
#         array: Data to multiply.
#         axis (None or int): If None, combine all values from the array into
#             a single scalar result; if an int, group by that axis: `0` is the
#             outermost, `1` is the first level of nested lists, etc., and
#             negative `axis` counts from the innermost: `-1` is the innermost,
#             `-2` is the next level up, etc.
#         keepdims (bool): If False, this reducer decreases the number of
#             dimensions by 1; if True, the reduced values are wrapped in a new
#             length-1 dimension so that the result of this operation may be
#             broadcasted with the original array.
#         mask_identity (bool): If True, reducing over empty lists results in
#             None (an option type); otherwise, reducing over empty lists
#             results in the operation's identity.

#     Multiplies elements of `array` (many types supported, including all
#     Awkward Arrays and Records). The identity of multiplication is `1` and it
#     is usually not masked. This operation is the same as NumPy's
#     [prod](https://docs.scipy.org/doc/numpy/reference/generated/numpy.prod.html)
#     if all lists at a given dimension have the same length and no None values,
#     but it generalizes to cases where they do not.

#     See #ak.sum for a more complete description of nested list and missing
#     value (None) handling in reducers.
#     """
#     layout = ak.operations.convert.to_layout(
#         array, allow_record=False, allow_other=False
#     )
#     if axis is None:

#         def reduce(xs):
#             if len(xs) == 1:
#                 return xs[0]
#             else:
#                 return xs[0] * reduce(xs[1:])

#         return reduce(
#             [ak.nplike.of(x).prod(x) for x in ak._util.completely_flatten(layout)]
#         )
#     else:
#         behavior = ak._util.behaviorof(array)
#         return ak._util.wrap(
#             layout.prod(axis=axis, mask=mask_identity, keepdims=keepdims), behavior
#         )
