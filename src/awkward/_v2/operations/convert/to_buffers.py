# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def to_buffers(
    array,
    container=None,
    partition_start=0,
    form_key="node{id}",
    key_format="part{partition}-{form_key}-{attribute}",
    virtual="materialize",
):
    pass


#     """
#     Args:
#         array: Data to decompose into named buffers.
#         container (None or MutableMapping): The str \u2192 NumPy arrays (or
#             Python buffers) that represent the decomposed Awkward Array. This
#             `container` is only assumed to have a `__setitem__` method that
#             accepts strings as keys.
#         partition_start (non-negative int): If `array` is not partitioned, this is
#             the partition number that will be used as part of the container
#             key. If `array` is partitioned, this is the first partition number.
#         form_key (str, callable): Python format string containing
#             `"{id}"` or a function that takes non-negative integer as a string
#             and the current `layout` as keyword arguments and returns a string,
#             for use as a `form_key` on each Form node and in `key_format` (below).
#         key_format (str or callable): Python format string containing
#             `"{partition}"`, `"{form_key}"`, and/or `"{attribute}"` or a function
#             that takes these as keyword arguments and returns a string to use
#             as keys for buffers in the `container`. The `partition` is a
#             partition number (non-negative integer, passed as a string), the
#             `form_key` is the result of applying `form_key` (above), and the
#             `attribute` is a hard-coded string representing the buffer's function
#             (e.g. `"data"`, `"offsets"`, `"index"`).
#         virtual (str): If `"materialize"`, any virtual arrays will be materialized
#             and the materialized data will be included in the container. If `"pass"`,
#             a virtual array's Form is passed through as a #ak.forms.VirtualForm,
#             assuming that it contains `form_keys` that can be found in the
#             container (e.g. by a previous pass through this function). No other
#             values are allowed for this function argument.

#     Decomposes an Awkward Array into a Form and a collection of memory buffers,
#     so that data can be losslessly written to file formats and storage devices
#     that only map names to binary blobs (such as a filesystem directory).

#     This function returns a 3-tuple:

#         (form, length, container)

#     where the `form` is a #ak.forms.Form (which can be converted to JSON
#     with `tojson`), the `length` is either an integer (`len(array)`) or a list
#     of the lengths of each partition in `array`, and the `container` is either
#     the MutableMapping you passed in or a new dict containing the buffers (as
#     NumPy arrays).

#     These are also the first three arguments of #ak.from_buffers, so a full
#     round-trip is

#         >>> reconstituted = ak.from_buffers(*ak.to_buffers(original))

#     The `container` argument lets you specify your own MutableMapping, which
#     might be an interface to some storage format or device (e.g. h5py). It's
#     okay if the `container` drops NumPy's `dtype` and `shape` information,
#     leaving raw bytes, since `dtype` and `shape` can be reconstituted from
#     the #ak.forms.NumpyForm.

#     The `partition_start` argument lets you fill the `container` gradually or
#     in parallel. If the `array` is not partitioned, the `partition_start`
#     argument sets its partition number (for the container keys, through
#     `key_format`). If the `array` is partitioned, the first partition is numbered
#     `partition_start` and as many are filled as ar in `array`. See #ak.partitions
#     to get the number of partitions in `array`.

#     Here is a simple example:

#         >>> original = ak.Array([[1, 2, 3], [], [4, 5]])
#         >>> form, length, container = ak.to_buffers(original)
#         >>> form
#         {
#             "class": "ListOffsetArray64",
#             "offsets": "i64",
#             "content": {
#                 "class": "NumpyArray",
#                 "itemsize": 8,
#                 "format": "l",
#                 "primitive": "int64",
#                 "form_key": "node1"
#             },
#             "form_key": "node0"
#         }
#         >>> length
#         3
#         >>> container
#         {'part0-node0-offsets': array([0, 3, 3, 5], dtype=int64),
#          'part0-node1-data': array([1, 2, 3, 4, 5])}

#     which may be read back with

#         >>> ak.from_buffers(form, length, container)
#         <Array [[1, 2, 3], [], [4, 5]] type='3 * var * int64'>

#     Here is an example that builds up a partitioned array:

#         >>> container = {}
#         >>> lengths = []
#         >>> form, length, _ = ak.to_buffers(ak.Array([[1, 2, 3], [], [4, 5]]), container, 0)
#         >>> lengths.append(length)
#         >>> form, length, _ = ak.to_buffers(ak.Array([[6, 7, 8, 9]]), container, 1)
#         >>> lengths.append(length)
#         >>> form, length, _ = ak.to_buffers(ak.Array([[], [], []]), container, 2)
#         >>> lengths.append(length)
#         >>> form, length, _ = ak.to_buffers(ak.Array([[10]]), container, 3)
#         >>> lengths.append(length)
#         >>> form
#         {
#             "class": "ListOffsetArray64",
#             "offsets": "i64",
#             "content": {
#                 "class": "NumpyArray",
#                 "itemsize": 8,
#                 "format": "l",
#                 "primitive": "int64",
#                 "form_key": "node1"
#             },
#             "form_key": "node0"
#         }
#         >>> lengths
#         [3, 1, 3, 1]
#         >>> container
#         {'part0-node0-offsets': array([0, 3, 3, 5], dtype=int64),
#          'part0-node1-data': array([1, 2, 3, 4, 5]),
#          'part1-node0-offsets': array([0, 4], dtype=int64),
#          'part1-node1-data': array([6, 7, 8, 9]),
#          'part2-node0-offsets': array([0, 0, 0, 0], dtype=int64),
#          'part2-node1-data': array([], dtype=float64),
#          'part3-node0-offsets': array([0, 1], dtype=int64),
#          'part3-node1-data': array([10])}

#     The object returned by #ak.from_buffers is now a partitioned array:

#         >>> reconstituted = ak.from_buffers(form, lengths, container)
#         >>> reconstituted
#         <Array [[1, 2, 3], [], [4, ... [], [], [10]] type='8 * var * int64'>
#         >>> ak.partitions(reconstituted)
#         [3, 1, 3, 1]

#     If you intend to use this function for saving data, you may want to pack it
#     first with #ak.packed.

#     See also #ak.from_buffers and #ak.packed.
#     """
#     if container is None:
#         container = {}

#     def index_form(index):
#         if isinstance(index, ak._v2.index.Index64):
#             return "i64"
#         elif isinstance(index, ak._v2.index.Index32):
#             return "i32"
#         elif isinstance(index, ak._v2.index.IndexU32):
#             return "u32"
#         elif isinstance(index, ak._v2.index.Index8):
#             return "i8"
#         elif isinstance(index, ak._v2.index.IndexU8):
#             return "u8"
#         else:
#             raise AssertionError(
#                 "unrecognized index: "
#                 + repr(index)
#
#             )

#     if isinstance(form_key, str):

#         def generate_form_key(form_key):
#             def fk(**v):
#                 return form_key.format(**v)

#             return fk

#         form_key = generate_form_key(form_key)

#     if isinstance(key_format, str):

#         def generate_key_format(key_format):
#             def kf(**v):
#                 return key_format.format(**v)

#             return kf

#         key_format = generate_key_format(key_format)

#     num_form_keys = [0]

#     def little_endian(array):
#         return array.astype(array.dtype.newbyteorder("<"), copy=False)

#     def fill(layout, part):
#         has_identities = layout.identities is not None
#         parameters = layout.parameters
#         key_index = num_form_keys[0]
#         num_form_keys[0] += 1

#         if has_identities:
#             raise NotImplementedError(
#                 "ak.to_buffers for an array with Identities"
#
#             )

#         if isinstance(layout, ak._v2.contents.EmptyArray):
#             fk = form_key(id=str(key_index))
#             key = key_format(form_key=fk, attribute="data", partition=str(part))
#             container[key] = little_endian(numpy.asarray(layout))
#             return ak.forms.EmptyForm(has_identities, parameters, fk)

#         elif isinstance(
#             layout,
#             (
#                 ak._v2.contents.IndexedArray32,
#                 ak._v2.contents.IndexedArrayU32,
#                 ak._v2.contents.IndexedArray64,
#             ),
#         ):
#             fk = form_key(id=str(key_index), layout=layout)
#             key = key_format(form_key=fk, attribute="index", partition=str(part))
#             container[key] = little_endian(numpy.asarray(layout.index))
#             return ak.forms.IndexedForm(
#                 index_form(layout.index),
#                 fill(layout.content, part),
#                 has_identities,
#                 parameters,
#                 fk,
#             )

#         elif isinstance(
#             layout, (ak._v2.contents.IndexedOptionArray32, ak._v2.contents.IndexedOptionArray64)
#         ):
#             fk = form_key(id=str(key_index), layout=layout)
#             key = key_format(form_key=fk, attribute="index", partition=str(part))
#             container[key] = little_endian(numpy.asarray(layout.index))
#             return ak.forms.IndexedOptionForm(
#                 index_form(layout.index),
#                 fill(layout.content, part),
#                 has_identities,
#                 parameters,
#                 fk,
#             )

#         elif isinstance(layout, ak._v2.contents.ByteMaskedArray):
#             fk = form_key(id=str(key_index), layout=layout)
#             key = key_format(form_key=fk, attribute="mask", partition=str(part))
#             container[key] = little_endian(numpy.asarray(layout.mask))
#             return ak.forms.ByteMaskedForm(
#                 index_form(layout.mask),
#                 fill(layout.content, part),
#                 layout.valid_when,
#                 has_identities,
#                 parameters,
#                 fk,
#             )

#         elif isinstance(layout, ak._v2.contents.BitMaskedArray):
#             fk = form_key(id=str(key_index), layout=layout)
#             key = key_format(form_key=fk, attribute="mask", partition=str(part))
#             container[key] = little_endian(numpy.asarray(layout.mask))
#             return ak.forms.BitMaskedForm(
#                 index_form(layout.mask),
#                 fill(layout.content, part),
#                 layout.valid_when,
#                 layout.lsb_order,
#                 has_identities,
#                 parameters,
#                 fk,
#             )

#         elif isinstance(layout, ak._v2.contents.UnmaskedArray):
#             return ak.forms.UnmaskedForm(
#                 fill(layout.content, part),
#                 has_identities,
#                 parameters,
#                 form_key(id=str(key_index), layout=layout),
#             )

#         elif isinstance(
#             layout,
#             (ak._v2.contents.ListArray32, ak._v2.contents.ListArrayU32, ak._v2.contents.ListArray64),
#         ):
#             fk = form_key(id=str(key_index), layout=layout)
#             key = key_format(form_key=fk, attribute="starts", partition=str(part))
#             container[key] = little_endian(numpy.asarray(layout.starts))
#             key = key_format(form_key=fk, attribute="stops", partition=str(part))
#             container[key] = little_endian(numpy.asarray(layout.stops))
#             return ak.forms.ListForm(
#                 index_form(layout.starts),
#                 index_form(layout.stops),
#                 fill(layout.content, part),
#                 has_identities,
#                 parameters,
#                 fk,
#             )

#         elif isinstance(
#             layout,
#             (
#                 ak._v2.contents.ListOffsetArray32,
#                 ak._v2.contents.ListOffsetArrayU32,
#                 ak._v2.contents.ListOffsetArray64,
#             ),
#         ):
#             fk = form_key(id=str(key_index), layout=layout)
#             key = key_format(form_key=fk, attribute="offsets", partition=str(part))
#             container[key] = little_endian(numpy.asarray(layout.offsets))
#             return ak.forms.ListOffsetForm(
#                 index_form(layout.offsets),
#                 fill(layout.content, part),
#                 has_identities,
#                 parameters,
#                 fk,
#             )

#         elif isinstance(layout, ak._v2.contents.NumpyArray):
#             fk = form_key(id=str(key_index), layout=layout)
#             key = key_format(form_key=fk, attribute="data", partition=str(part))
#             array = numpy.asarray(layout)
#             container[key] = little_endian(array)
#             form = ak.forms.Form.from_numpy(array.dtype)
#             return ak.forms.NumpyForm(
#                 layout.shape[1:],
#                 form.itemsize,
#                 form.format,
#                 has_identities,
#                 parameters,
#                 fk,
#             )

#         elif isinstance(layout, ak._v2.contents.RecordArray):
#             if layout.istuple:
#                 forms = [fill(x, part) for x in layout.contents]
#                 keys = None
#             else:
#                 forms = []
#                 keys = []
#                 for k in layout.keys():
#                     forms.append(fill(layout[k], part))
#                     keys.append(k)

#             return ak.forms.RecordForm(
#                 forms,
#                 keys,
#                 has_identities,
#                 parameters,
#                 form_key(id=str(key_index), layout=layout),
#             )

#         elif isinstance(layout, ak._v2.contents.RegularArray):
#             return ak.forms.RegularForm(
#                 fill(layout.content, part),
#                 layout.size,
#                 has_identities,
#                 parameters,
#                 form_key(id=str(key_index), layout=layout),
#             )

#         elif isinstance(
#             layout,
#             (
#                 ak._v2.contents.UnionArray8_32,
#                 ak._v2.contents.UnionArray8_U32,
#                 ak._v2.contents.UnionArray8_64,
#             ),
#         ):
#             forms = []
#             for x in layout.contents:
#                 forms.append(fill(x, part))

#             fk = form_key(id=str(key_index), layout=layout)
#             key = key_format(form_key=fk, attribute="tags", partition=str(part))
#             container[key] = little_endian(numpy.asarray(layout.tags))
#             key = key_format(form_key=fk, attribute="index", partition=str(part))
#             container[key] = little_endian(numpy.asarray(layout.index))
#             return ak.forms.UnionForm(
#                 index_form(layout.tags),
#                 index_form(layout.index),
#                 forms,
#                 has_identities,
#                 parameters,
#                 fk,
#             )

#         elif isinstance(layout, ak._v2.contents.VirtualArray):
#             if virtual == "materialize":
#                 return fill(layout.array, part)
#             elif virtual == "pass":
#                 return ak.forms.VirtualForm(
#                     layout.form,
#                     layout.generator.length is not None,
#                     layout.identities is not None,
#                     layout.parameters,
#                     None,
#                 )
#             else:
#                 raise ValueError(
#                     "unrecognized value for 'virtual': "
#                     + str(virtual)
#
#                 )

#         else:
#             raise AssertionError(
#                 "unrecognized layout node type: "
#                 + str(type(layout))
#
#             )

#     layout = to_layout(array, allow_record=False, allow_other=False)

#     if isinstance(layout, ak.partition.PartitionedArray):
#         form = None
#         length = []
#         for part, content in enumerate(layout.partitions):
#             num_form_keys[0] = 0

#             f = fill(content, partition_start + part)

#             if form is None:
#                 form = f
#             elif form != f:
#                 raise ValueError(
#                     """the Form of partition {0}:

#     {1}

# differs from the first Form:

#     {2}""".format(
#                         partition_start + part,
#                         f.tojson(True, False),
#                         form.tojson(True, False),
#                     )
#
#                 )
#             length.append(len(content))

#     else:
#         form = fill(layout, partition_start)
#         length = len(layout)

#     return form, length, container
