# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


# @ak._connect._numpy.implements("where")
def where(condition, *args, **kwargs):
    pass


#     """
#     Args:
#         condition (np.ndarray or rectilinear #ak.Array of booleans): In the
#             three-argument form of this function (`condition`, `x`, `y`),
#             True values in `condition` select values from `x` and False
#             values in `condition` select values from `y`.
#         x: Data with the same length as `condition`.
#         y: Data with the same length as `condition`.
#         mergebool (bool, default is True): If True, boolean and nummeric data
#             can be combined into the same buffer, losing information about
#             False vs `0` and True vs `1`; otherwise, they are kept in separate
#             buffers with distinct types (using an #ak.layout.UnionArray8_64).
#         highlevel (bool, default is True): If True, return an #ak.Array;
#             otherwise, return a low-level #ak.layout.Content subclass.

#     This function has a one-argument form, `condition` without `x` or `y`, and
#     a three-argument form, `condition`, `x`, and `y`. In the one-argument form,
#     it is completely equivalent to NumPy's
#     [nonzero](https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html)
#     function.

#     In the three-argument form, it acts as a vectorized ternary operator:
#     `condition`, `x`, and `y` must all have the same length and

#         output[i] = x[i] if condition[i] else y[i]

#     for all `i`. The structure of `x` and `y` do not need to be the same; if
#     they are incompatible types, the output will have #ak.type.UnionType.
#     """
#     mergebool, highlevel = ak._util.extra(
#         (), kwargs, [("mergebool", True), ("highlevel", True)]
#     )

#     akcondition = ak.operations.convert.to_layout(
#         condition, allow_record=False, allow_other=False
#     )

#     if len(args) == 0:
#         nplike = ak.nplike.of(akcondition)
#         if isinstance(akcondition, ak.partition.PartitionedArray):
#             akcondition = akcondition.replace_partitions(
#                 [
#                     ak.layout.NumpyArray(ak.operations.convert.to_numpy(x))
#                     for x in akcondition.partitions
#                 ]
#             )
#         else:
#             akcondition = ak.layout.NumpyArray(
#                 ak.operations.convert.to_numpy(akcondition)
#             )
#         out = nplike.nonzero(ak.operations.convert.to_numpy(akcondition))
#         if highlevel:
#             return tuple(
#                 ak._util.wrap(ak.layout.NumpyArray(x), ak._util.behaviorof(condition))
#                 for x in out
#             )
#         else:
#             return tuple(ak.layout.NumpyArray(x) for x in out)

#     elif len(args) == 1:
#         raise ValueError(
#             "either both or neither of x and y should be given"
#             + ak._util.exception_suffix(__file__)
#         )

#     elif len(args) == 2:
#         left, right = [
#             ak.operations.convert.to_layout(x, allow_record=False, allow_other=True)
#             for x in args
#         ]
#         good_arrays = [akcondition]
#         if isinstance(left, ak.layout.Content):
#             good_arrays.append(left)
#         if isinstance(right, ak.layout.Content):
#             good_arrays.append(right)
#         nplike = ak.nplike.of(*good_arrays)

#         def getfunction(inputs):
#             akcondition, left, right = inputs
#             if isinstance(akcondition, ak.layout.NumpyArray):
#                 npcondition = nplike.asarray(akcondition)
#                 tags = ak.layout.Index8((npcondition == 0).view(np.int8))
#                 index = ak.layout.Index64(nplike.arange(len(tags), dtype=np.int64))
#                 if not isinstance(left, ak.layout.Content):
#                     left = ak.layout.NumpyArray(nplike.repeat(left, len(tags)))
#                 if not isinstance(right, ak.layout.Content):
#                     right = ak.layout.NumpyArray(nplike.repeat(right, len(tags)))
#                 tmp = ak.layout.UnionArray8_64(tags, index, [left, right])
#                 return lambda: (tmp.simplify(mergebool=mergebool),)
#             else:
#                 return None

#         behavior = ak._util.behaviorof(akcondition, left, right)
#         out = ak._util.broadcast_and_apply(
#             [akcondition, left, right],
#             getfunction,
#             behavior,
#             pass_depth=False,
#             numpy_to_regular=True,
#         )

#         return ak._util.maybe_wrap(out[0], behavior, highlevel)

#     else:
#         raise TypeError(
#             "where() takes from 1 to 3 positional arguments but {0} were "
#             "given".format(len(args) + 1) + ak._util.exception_suffix(__file__)
#         )
