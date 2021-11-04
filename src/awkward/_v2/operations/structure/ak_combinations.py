# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def combinations(
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
#         n (int): The number of items to choose in each list: `2` chooses
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
#         behavior (None or dict): Custom #ak.behavior for the output array, if
#             high-level.

#     Computes a Cartesian product (i.e. cross product) of `array` with itself
#     that is restricted to combinations sampled without replacement. If the
#     normal Cartesian product is thought of as an `n` dimensional tensor, these
#     represent the "upper triangle" of sets without repetition. If
#     `replacement=True`, the diagonal of this "upper triangle" is included.

#     As a simple example with `axis=0`, consider the following `array`

#         ak.Array(["a", "b", "c", "d", "e"])

#     The combinations choose `2` are:

#         >>> ak.to_list(ak.combinations(array, 2, axis=0))
#         [('a', 'b'), ('a', 'c'), ('a', 'd'), ('a', 'e'),
#                      ('b', 'c'), ('b', 'd'), ('b', 'e'),
#                                  ('c', 'd'), ('c', 'e'),
#                                              ('d', 'e')]

#     Including the diagonal allows pairs like `('a', 'a')`.

#         >>> ak.to_list(ak.combinations(array, 2, axis=0, replacement=True))
#         [('a', 'a'), ('a', 'b'), ('a', 'c'), ('a', 'd'), ('a', 'e'),
#                      ('b', 'b'), ('b', 'c'), ('b', 'd'), ('b', 'e'),
#                                  ('c', 'c'), ('c', 'd'), ('c', 'e'),
#                                              ('d', 'd'), ('d', 'e'),
#                                                          ('e', 'e')]

#     The combinations choose `3` can't be easily arranged as a triangle
#     in two dimensions.

#         >>> ak.to_list(ak.combinations(array, 3, axis=0))
#         [('a', 'b', 'c'), ('a', 'b', 'd'), ('a', 'b', 'e'), ('a', 'c', 'd'), ('a', 'c', 'e'),
#          ('a', 'd', 'e'), ('b', 'c', 'd'), ('b', 'c', 'e'), ('b', 'd', 'e'), ('c', 'd', 'e')]

#     Including the (three-dimensional) diagonal allows triples like
#     `('a', 'a', 'a')`, but also `('a', 'a', 'b')`, `('a', 'b', 'b')`, etc.,
#     but not `('a', 'b', 'a')`. All combinations are in the same order as
#     the original array.

#         >>> ak.to_list(ak.combinations(array, 3, axis=0, replacement=True))
#         [('a', 'a', 'a'), ('a', 'a', 'b'), ('a', 'a', 'c'), ('a', 'a', 'd'), ('a', 'a', 'e'),
#          ('a', 'b', 'b'), ('a', 'b', 'c'), ('a', 'b', 'd'), ('a', 'b', 'e'), ('a', 'c', 'c'),
#          ('a', 'c', 'd'), ('a', 'c', 'e'), ('a', 'd', 'd'), ('a', 'd', 'e'), ('a', 'e', 'e'),
#          ('b', 'b', 'b'), ('b', 'b', 'c'), ('b', 'b', 'd'), ('b', 'b', 'e'), ('b', 'c', 'c'),
#          ('b', 'c', 'd'), ('b', 'c', 'e'), ('b', 'd', 'd'), ('b', 'd', 'e'), ('b', 'e', 'e'),
#          ('c', 'c', 'c'), ('c', 'c', 'd'), ('c', 'c', 'e'), ('c', 'd', 'd'), ('c', 'd', 'e'),
#          ('c', 'e', 'e'), ('d', 'd', 'd'), ('d', 'd', 'e'), ('d', 'e', 'e'), ('e', 'e', 'e')]

#     The primary purpose of this function, however, is to compute a different
#     set of combinations for each element of an array: in other words, `axis=1`.
#     The following `array` has a different number of items in each element.

#         ak.Array([[1, 2, 3, 4], [], [5], [6, 7, 8]])

#     There are 6 ways to choose pairs from 4 elements, 0 ways to choose pairs
#     from 0 elements, 0 ways to choose pairs from 1 element, and 3 ways to
#     choose pairs from 3 elements.

#         >>> ak.to_list(ak.combinations(array, 2))
#         [
#          [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
#          [],
#          [],
#          [(6, 7), (6, 8), (7, 8)]
#         ]

#     Note, however, that the combinatorics isn't determined by equality of
#     the data themselves, but by their placement in the array. For example,
#     even if all elements of an array are equal, the output has the same
#     structure.

#         >>> same = ak.Array([[7, 7, 7, 7], [], [7], [7, 7, 7]])
#         >>> ak.to_list(ak.combinations(same, 2))
#         [
#          [(7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7)],
#          [],
#          [],
#          [(7, 7), (7, 7), (7, 7)]
#         ]

#     To get records instead of tuples, pass a set of field names to `fields`.

#         >>> ak.to_list(ak.combinations(array, 2, fields=["x", "y"]))
#         [
#          [{'x': 1, 'y': 2}, {'x': 1, 'y': 3}, {'x': 1, 'y': 4},
#                             {'x': 2, 'y': 3}, {'x': 2, 'y': 4},
#                                               {'x': 3, 'y': 4}],
#          [],
#          [],
#          [{'x': 6, 'y': 7}, {'x': 6, 'y': 8},
#                             {'x': 7, 'y': 8}]]

#     This operation can be constructed from #ak.argcartesian and other
#     primitives:

#         >>> left, right = ak.unzip(ak.argcartesian([array, array]))
#         >>> keep = left < right
#         >>> result = ak.zip([array[left][keep], array[right][keep]])
#         >>> ak.to_list(result)
#         [
#          [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
#          [],
#          [],
#          [(6, 7), (6, 8), (7, 8)]]

#     but it is frequently needed for data analysis, and the logic of which
#     indexes to `keep` (above) gets increasingly complicated for large `n`.

#     To get list index positions in the tuples/records, rather than data from
#     the original `array`, use #ak.argcombinations instead of #ak.combinations.
#     The #ak.argcombinations form can be particularly useful as nested indexing
#     in #ak.Array.__getitem__.
#     """
#     if parameters is None:
#         parameters = {}
#     else:
#         parameters = dict(parameters)
#     if with_name is not None:
#         parameters["__record__"] = with_name

#     layout = ak._v2.operations.convert.to_layout(
#         array, allow_record=False, allow_other=False
#     )
#     out = layout.combinations(
#         n, replacement=replacement, keys=fields, parameters=parameters, axis=axis
#     )
#     return ak._v2._util.maybe_wrap_like(out, array, behavior, highlevel)
