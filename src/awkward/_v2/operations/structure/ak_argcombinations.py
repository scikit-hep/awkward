# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def argcombinations(
    array,
    n,
    replacement=False,
    axis=1,
    fields=None,
    parameters=None,
    with_name=None,
    highlevel=True,
    behavior=None,
):
    raise NotImplementedError


#     """
#     Args:
#         array: Array from which to choose `n` items without replacement.
#         n (int): The number of items to choose from each list: `2` chooses
#             unique pairs, `3` chooses unique triples, etc.
#         replacement (bool): If True, combinations that include the same
#             item more than once are allowed; otherwise each item in a
#             combinations is strictly unique.
#         axis (int): The dimension at which this operation is applied. The
#             outermost dimension is `0`, followed by `1`, etc., and negative
#             values count backward from the innermost: `-1` is the innermost
#             dimension, `-2` is the next level up, etc.
#         fields (None or list of str): If None, the pairs/triples/etc. are
#             tuples with unnamed fields; otherwise, these `fields` name the
#             fields. The number of `fields` must be equal to `n`.
#         parameters (None or dict): Parameters for the new
#             #ak.layout.RecordArray node that is created by this operation.
#         with_name (None or str): Assigns a `"__record__"` name to the new
#             #ak.layout.RecordArray node that is created by this operation
#             (overriding `parameters`, if necessary).
#         highlevel (bool): If True, return an #ak.Array; otherwise, return
#             a low-level #ak.layout.Content subclass.

#     Computes a Cartesian product (i.e. cross product) of `array` with itself
#     that is restricted to combinations sampled without replacement,
#     like #ak.combinations, but returning integer indexes for
#     #ak.Array.__getitem__.

#     The motivation and uses of this function are similar to those of
#     #ak.argcartesian. See #ak.combinations and #ak.argcartesian for a more
#     complete description.
#     """
#     if parameters is None:
#         parameters = {}
#     else:
#         parameters = dict(parameters)
#     if with_name is not None:
#         parameters["__record__"] = with_name

#     if axis < 0:
#         raise ValueError(
#             "the 'axis' for argcombinations must be non-negative"
#
#         )
#     else:
#         layout = ak._v2.operations.convert.to_layout(
#             array, allow_record=False, allow_other=False
#         ).localindex(axis)
#         out = layout.combinations(
#             n, replacement=replacement, keys=fields, parameters=parameters, axis=axis
#         )
#         return ak._v2._util.maybe_wrap_like(out, array, behavior, highlevel)
