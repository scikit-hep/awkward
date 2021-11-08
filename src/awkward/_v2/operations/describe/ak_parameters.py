# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def parameters(array):
    raise NotImplementedError


#     """
#     Extracts parameters from the outermost array node of `array` (many types
#     supported, including all Awkward Arrays and Records).

#     Parameters are a dict from str to JSON-like objects, usually strings.
#     Every #ak.layout.Content node has a different set of parameters. Some
#     key names are special, such as `"__record__"` and `"__array__"` that name
#     particular records and arrays as capable of supporting special behaviors.

#     See #ak.Array and #ak.behavior for a more complete description of
#     behaviors.
#     """
#     if isinstance(array, (ak._v2.highlevel.Array, ak._v2.highlevel.Record)):
#         return array.layout.parameters

#     elif isinstance(
#         array,
#         (
#             ak._v2.contents.Content,
#             ak._v2.record.Record,
#             ak.partition.PartitionedArray,   # NO PARTITIONED ARRAY
#         ),
#     ):
#         return array.parameters

#     elif isinstance(array, ak._v2.highlevel.ArrayBuilder):
#         return array.snapshot().layout.parameters

#     elif isinstance(array, ak.layout.ArrayBuilder):
#         return array.snapshot().parameters

#     else:
#         return {}
