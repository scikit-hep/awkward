# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def to_list(array):
    raise NotImplementedError


#     """
#     Converts `array` (many types supported, including all Awkward Arrays and
#     Records) into Python objects.

#     Awkward Array types have the following Pythonic translations.

#        * #ak.types.PrimitiveType: converted into bool, int, float.
#        * #ak.types.OptionType: missing values are converted into None.
#        * #ak.types.ListType: converted into list.
#        * #ak.types.RegularType: also converted into list. Python (and JSON)
#          forms lose information about the regularity of list lengths.
#        * #ak.types.ListType with parameter `"__array__"` equal to
#          `"__bytestring__"`: converted into bytes.
#        * #ak.types.ListType with parameter `"__array__"` equal to
#          `"__string__"`: converted into str.
#        * #ak.types.RecordArray without field names: converted into tuple.
#        * #ak.types.RecordArray with field names: converted into dict.
#        * #ak.types.UnionArray: Python data are naturally heterogeneous.

#     See also #ak.from_iter and #ak.Array.tolist.
#     """
#     if isinstance(array, np.bool_):
#         return bool(array)

#     elif isinstance(array, np.number):
#         if isinstance(array, (np.datetime64, np.timedelta64)):
#             return array
#         elif isinstance(array, np.integer):
#             return int(array)
#         elif isinstance(array, np.floating):
#             return float(array)
#         elif isinstance(array, np.all_complex):
#             return complex(array)
#         else:
#             raise NotImplementedError(type(array))

#     elif array is None or isinstance(array, (bool, str, bytes, numbers.Number)):
#         return array

#     elif ak._v2._util.py27 and isinstance(array, ak._v2._util.unicode):
#         return array

#     elif isinstance(array, np.ndarray):
#         return array.tolist()

#     elif isinstance(array, ak._v2.behaviors.string.ByteBehavior):
#         return array.__bytes__()

#     elif isinstance(array, ak._v2.behaviors.string.CharBehavior):
#         return array.__str__()

#     elif ak._v2.operations.describe.parameters(array).get("__array__") == "byte":
#         return ak._v2.behaviors.string.CharBehavior(array).__bytes__()

#     elif ak._v2.operations.describe.parameters(array).get("__array__") == "char":
#         return ak._v2.behaviors.string.CharBehavior(array).__str__()

#     elif isinstance(array, np.datetime64) or isinstance(array, np.timedelta64):
#         return array

#     elif isinstance(array, ak._v2.highlevel.Array):
#         return [to_list(x) for x in array]

#     elif isinstance(array, ak._v2.highlevel.Record):
#         return to_list(array.layout)

#     elif isinstance(array, ak._v2.highlevel.ArrayBuilder):
#         return to_list(array.snapshot())

#     elif isinstance(array, ak._v2.record.Record) and array.istuple:
#         return tuple(to_list(x) for x in array.fields())

#     elif isinstance(array, ak._v2.record.Record):
#         return {n: to_list(x) for n, x in array.fielditems()}

#     elif isinstance(array, ak.layout.ArrayBuilder):
#         return [to_list(x) for x in array.snapshot()]

#     elif isinstance(array, ak._v2.contents.NumpyArray):
#         if array.format.upper().startswith("M"):
#             return (
#                 [
#                     x
#                     for x in ak.nplike.of(array)
#                     .asarray(array.view_int64)
#                     .view(array.format)
#                 ]
#                 # FIXME: .tolist() returns
#                 # [[1567416600000000000], [1568367000000000000], [1569096000000000000]]
#                 # instead of [numpy.datetime64('2019-09-02T09:30:00'), numpy.datetime64('2019-09-13T09:30:00'), numpy.datetime64('2019-09-21T20:00:00')]
#                 # see test_from_pandas() test
#             )
#         else:
#             return ak.nplike.of(array).asarray(array).tolist()

#     elif isinstance(array, (ak._v2.contents.Content, ak.partition.PartitionedArray)):   # NO PARTITIONED ARRAY
#         return [to_list(x) for x in array]

#     elif isinstance(array, ak._v2.contents.Content):
#         import awkward._v2.tmp_for_testing

#         return to_list(awkward._v2.tmp_for_testing.v2_to_v1(array))

#     elif isinstance(array, ak._v2.record.Record):
#         import awkward._v2.tmp_for_testing

#         return to_list(
#             awkward._v2.tmp_for_testing.v2_to_v1(array.array[array.at : array.at + 1])[
#                 0
#             ]
#         )

#     elif isinstance(array, dict):
#         return dict((n, to_list(x)) for n, x in array.items())

#     elif isinstance(array, Iterable):
#         return [to_list(x) for x in array]

#     else:
#         raise TypeError(
#             "unrecognized array type: {0}".format(type(array))
#
#         )
