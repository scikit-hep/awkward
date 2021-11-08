# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def zip(
    arrays,
    depth_limit=None,
    parameters=None,
    with_name=None,
    highlevel=True,
    behavior=None,
    right_broadcast=False,
):
    raise NotImplementedError


#     """
#     Args:
#         arrays (dict or iterable of arrays): Arrays to combine into a
#             record-containing structure (if a dict) or a tuple-containing
#             structure (if any other kind of iterable).
#         depth_limit (None or int): If None, attempt to fully broadcast the
#             `array` to all levels. If an int, limit the number of dimensions
#             that get broadcasted. The minimum value is `1`, for no
#             broadcasting.
#         parameters (None or dict): Parameters for the new
#             #ak.layout.RecordArray node that is created by this operation.
#         with_name (None or str): Assigns a `"__record__"` name to the new
#             #ak.layout.RecordArray node that is created by this operation
#             (overriding `parameters`, if necessary).
#         highlevel (bool): If True, return an #ak.Array; otherwise, return
#             a low-level #ak.layout.Content subclass.
#         behavior (None or dict): Custom #ak.behavior for the output array, if
#             high-level.
#         right_broadcast (bool): If True, follow rules for implicit
#             right-broadcasting, as described in #ak.broadcast_arrays.

#     Combines `arrays` into a single structure as the fields of a collection
#     of records or the slots of a collection of tuples. If the `arrays` have
#     nested structure, they are broadcasted with one another to form the
#     records or tuples as deeply as possible, though this can be limited by
#     `depth_limit`.

#     This operation may be thought of as the opposite of projection in
#     #ak.Array.__getitem__, which extracts fields one at a time, or
#     #ak.unzip, which extracts them all in one call.

#     Consider the following arrays, `one` and `two`.

#         >>> one = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6]])
#         >>> two = ak.Array([["a", "b", "c"], [], ["d", "e"], ["f"]])

#     Zipping them together using a dict creates a collection of records with
#     the same nesting structure as `one` and `two`.

#         >>> ak.to_list(ak.zip({"x": one, "y": two}))
#         [
#          [{'x': 1.1, 'y': 'a'}, {'x': 2.2, 'y': 'b'}, {'x': 3.3, 'y': 'c'}],
#          [],
#          [{'x': 4.4, 'y': 'd'}, {'x': 5.5, 'y': 'e'}],
#          [{'x': 6.6, 'y': 'f'}]
#         ]

#     Doing so with a list creates tuples, whose fields are not named.

#         >>> ak.to_list(ak.zip([one, two]))
#         [
#          [(1.1, 'a'), (2.2, 'b'), (3.3, 'c')],
#          [],
#          [(4.4, 'd'), (5.5, 'e')],
#          [(6.6, 'f')]
#         ]

#     Adding a third array with the same length as `one` and `two` but less
#     internal structure is okay: it gets broadcasted to match the others.
#     (See #ak.broadcast_arrays for broadcasting rules.)

#         >>> three = ak.Array([100, 200, 300, 400])
#         >>> ak.to_list(ak.zip([one, two, three]))
#         [
#          [[(1.1, 97, 100)], [(2.2, 98, 100)], [(3.3, 99, 100)]],
#          [],
#          [[(4.4, 100, 300)], [(5.5, 101, 300)]],
#          [[(6.6, 102, 400)]]
#         ]

#     However, if arrays have the same depth but different lengths of nested
#     lists, attempting to zip them together is a broadcasting error.

#         >>> one = ak.Array([[[1, 2, 3], [], [4, 5], [6]], [], [[7, 8]]])
#         >>> two = ak.Array([[[1.1, 2.2], [3.3], [4.4], [5.5]], [], [[6.6]]])
#         >>> ak.zip([one, two])
#         ValueError: in ListArray64, cannot broadcast nested list

#     For this, one can set the `depth_limit` to prevent the operation from
#     attempting to broadcast what can't be broadcasted.

#         >>> ak.to_list(ak.zip([one, two], depth_limit=1))
#         [([[1, 2, 3], [], [4, 5], [6]], [[1.1, 2.2], [3.3], [4.4], [5.5]]),
#          ([], []),
#          ([[7, 8]], [[6.6]])]

#     As an extreme, `depth_limit=1` is a handy way to make a record structure
#     at the outermost level, regardless of whether the fields have matching
#     structure or not.
#     """
#     if depth_limit is not None and depth_limit <= 0:
#         raise ValueError(
#             "depth_limit must be None or at least 1"
#
#         )

#     if isinstance(arrays, dict):
#         behavior = ak._v2._util.behaviorof(*arrays.values(), behavior=behavior)
#         recordlookup = []
#         layouts = []
#         num_scalars = 0
#         for n, x in arrays.items():
#             recordlookup.append(n)
#             try:
#                 layout = ak._v2.operations.convert.to_layout(
#                     x, allow_record=False, allow_other=False
#                 )
#             except TypeError:
#                 num_scalars += 1
#                 layout = ak._v2.operations.convert.to_layout(
#                     [x], allow_record=False, allow_other=False
#                 )
#             layouts.append(layout)

#     else:
#         behavior = ak._v2._util.behaviorof(*arrays, behavior=behavior)
#         recordlookup = None
#         layouts = []
#         num_scalars = 0
#         for x in arrays:
#             try:
#                 layout = ak._v2.operations.convert.to_layout(
#                     x, allow_record=False, allow_other=False
#                 )
#             except TypeError:
#                 num_scalars += 1
#                 layout = ak._v2.operations.convert.to_layout(
#                     [x], allow_record=False, allow_other=False
#                 )
#             layouts.append(layout)

#     to_record = num_scalars == len(arrays)

#     if with_name is not None:
#         if parameters is None:
#             parameters = {}
#         else:
#             parameters = dict(parameters)
#         parameters["__record__"] = with_name

#     def getfunction(inputs, depth):
#         if depth_limit == depth or (
#             depth_limit is None
#             and all(
#                 x.purelist_depth == 1
#                 or (
#                     x.purelist_depth == 2
#                     and x.purelist_parameter("__array__")
#                     in ("string", "bytestring", "categorical")
#                 )
#                 for x in inputs
#             )
#         ):
#             return lambda: (
#                 ak.layout.RecordArray(inputs, recordlookup, parameters=parameters),
#             )
#         else:
#             return None

#     out = ak._v2._util.broadcast_and_apply(
#         layouts,
#         getfunction,
#         behavior,
#         right_broadcast=right_broadcast,
#         pass_depth=True,
#     )
#     assert isinstance(out, tuple) and len(out) == 1
#     out = out[0]

#     if to_record:
#         out = out[0]
#         assert isinstance(out, ak.layout.Record)

#     return ak._v2._util.maybe_wrap(out, behavior, highlevel)
