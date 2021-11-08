# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


# @ak._v2._connect.numpy.implements("broadcast_arrays")
def broadcast_arrays(*arrays, **kwargs):
    raise NotImplementedError


#     """
#     Args:
#         arrays: Arrays to broadcast into the same structure.
#         left_broadcast (bool): If True, follow rules for implicit
#             left-broadcasting, as described below.
#         right_broadcast (bool): If True, follow rules for implicit
#             right-broadcasting, as described below.
#         highlevel (bool, default is True): If True, return an #ak.Array;
#             otherwise, return a low-level #ak.layout.Content subclass.

#     Like NumPy's
#     [broadcast_arrays](https://docs.scipy.org/doc/numpy/reference/generated/numpy.broadcast_arrays.html)
#     function, this function returns the input `arrays` with enough elements
#     duplicated that they can be combined element-by-element.

#     For NumPy arrays, this means that scalars are replaced with arrays with
#     the same scalar value repeated at every element of the array, and regular
#     dimensions are created to increase low-dimensional data into
#     high-dimensional data.

#     For example,

#         >>> ak.broadcast_arrays(5,
#         ...                     [1, 2, 3, 4, 5])
#         [<Array [5, 5, 5, 5, 5] type='5 * int64'>,
#          <Array [1, 2, 3, 4, 5] type='5 * int64'>]

#     and

#         >>> ak.broadcast_arrays(np.array([1, 2, 3]),
#         ...                     np.array([[0.1, 0.2, 0.3], [10, 20, 30]]))
#         [<Array [[  1,   2,   3], [ 1,  2,  3]] type='2 * 3 * int64'>,
#          <Array [[0.1, 0.2, 0.3], [10, 20, 30]] type='2 * 3 * float64'>]

#     Note that in the second example, when the `3 * int64` array is expanded
#     to match the `2 * 3 * float64` array, it is the deepest dimension that
#     is aligned. If we try to match a `2 * int64` with the `2 * 3 * float64`,

#         >>> ak.broadcast_arrays(np.array([1, 2]),
#         ...                     np.array([[0.1, 0.2, 0.3], [10, 20, 30]]))
#         ValueError: cannot broadcast RegularArray of size 2 with RegularArray of size 3

#     NumPy has the same behavior: arrays with different numbers of dimensions
#     are aligned to the right before expansion. One can control this by
#     explicitly adding a new axis (reshape to add a dimension of length 1)
#     where the expansion is supposed to take place because a dimension of
#     length 1 can be expanded like a scalar.

#         >>> ak.broadcast_arrays(np.array([1, 2])[:, np.newaxis],
#         ...                     np.array([[0.1, 0.2, 0.3], [10, 20, 30]]))
#         [<Array [[  1,   1,   1], [ 2,  2,  2]] type='2 * 3 * int64'>,
#          <Array [[0.1, 0.2, 0.3], [10, 20, 30]] type='2 * 3 * float64'>]

#     Again, NumPy does the same thing (`np.newaxis` is equal to None, so this
#     trick is often shown with None in the slice-tuple). Where the broadcasting
#     happens can be controlled, but numbers of dimensions that don't match are
#     implicitly aligned to the right (fitting innermost structure, not
#     outermost).

#     While that might be an arbitrary decision for rectilinear arrays, it is
#     much more natural for implicit broadcasting to align left for tree-like
#     structures. That is, the root of each data structure should agree and
#     leaves may be duplicated to match. For example,

#         >>> ak.broadcast_arrays([            100,   200,        300],
#         ...                     [[1.1, 2.2, 3.3],    [], [4.4, 5.5]])
#         [<Array [[100, 100, 100], [], [300, 300]] type='3 * var * int64'>,
#          <Array [[1.1, 2.2, 3.3], [], [4.4, 5.5]] type='3 * var * float64'>]

#     One typically wants single-item-per-element data to be duplicated to
#     match multiple-items-per-element data. Operations on the broadcasted
#     arrays like

#         one_dimensional + nested_lists

#     would then have the same effect as the procedural code

#         for x, outer in zip(one_dimensional, nested_lists):
#             output = []
#             for inner in outer:
#                 output.append(x + inner)
#             yield output

#     where `x` has the same value for each `inner` in the inner loop.

#     Awkward Array's broadcasting manages to have it both ways by applying the
#     following rules:

#        * If all dimensions are regular (i.e. #ak.types.RegularType), like NumPy,
#          implicit broadcasting aligns to the right, like NumPy.
#        * If any dimension is variable (i.e. #ak.types.ListType), which can
#          never be true of NumPy, implicit broadcasting aligns to the left.
#        * Explicit broadcasting with a length-1 regular dimension always
#          broadcasts, like NumPy.

#     Thus, it is important to be aware of the distinction between a dimension
#     that is declared to be regular in the type specification and a dimension
#     that is allowed to be variable (even if it happens to have the same length
#     for all elements). This distinction is can be accessed through the
#     #ak.Array.type, but it is lost when converting an array into JSON or
#     Python objects.
#     """
#     (highlevel, left_broadcast, right_broadcast) = ak._v2._util.extra(
#         (),
#         kwargs,
#         [("highlevel", True), ("left_broadcast", True), ("right_broadcast", True)],
#     )

#     inputs = []
#     for x in arrays:
#         y = ak._v2.operations.convert.to_layout(x, allow_record=True, allow_other=True)
#         if isinstance(y, ak.partition.PartitionedArray):   # NO PARTITIONED ARRAY
#             y = y.toContent()
#         if not isinstance(y, (ak._v2.contents.Content, ak._v2.contents.Record)):
#             y = ak._v2.contents.NumpyArray(ak.nplike.of(*arrays).array([y]))
#         inputs.append(y)

#     def getfunction(inputs):
#         if all(isinstance(x, ak._v2.contents.NumpyArray) for x in inputs):
#             return lambda: tuple(inputs)
#         else:
#             return None

#     behavior = ak._v2._util.behaviorof(*arrays)
#     out = ak._v2._util.broadcast_and_apply(
#         inputs,
#         getfunction,
#         behavior,
#         left_broadcast=left_broadcast,
#         right_broadcast=right_broadcast,
#         pass_depth=False,
#         numpy_to_regular=True,
#     )
#     assert isinstance(out, tuple)
#     if highlevel:
#         return [ak._v2._util.wrap(x, behavior) for x in out]
#     else:
#         return list(out)
