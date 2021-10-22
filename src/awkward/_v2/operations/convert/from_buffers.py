# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def from_buffers(
    form,
    length,
    container,
    partition_start=0,
    key_format="part{partition}-{form_key}-{attribute}",
    lazy=False,
    lazy_cache="new",
    lazy_cache_key=None,
    highlevel=True,
    behavior=None,
):
    pass


#     """
#     Args:
#         form (#ak.forms.Form or str/dict equivalent): The form of the Awkward
#             Array to reconstitute from named buffers.
#         length (int or iterable of int): Length of the array to reconstitute as a
#             non-partitioned array or the lengths (plural) of partitions in a
#             partitioned array.
#         container (Mapping, such as dict): The str \u2192 Python buffers that
#             represent the decomposed Awkward Array. This `container` is only
#             assumed to have a `__getitem__` method that accepts strings as keys.
#         partition_start (int): First (or only) partition number to get from the
#             `container`.
#         key_format (str or callable): Python format string containing
#             `"{partition}"`, `"{form_key}"`, and/or `"{attribute}"` or a function
#             that takes these as keyword arguments and returns a string to use
#             as keys for buffers in the `container`. The `partition` is a
#             partition number (non-negative integer, passed as a string), the
#             `form_key` is a string associated with each node in the Form, and the
#             `attribute` is a hard-coded string representing the buffer's function
#             (e.g. `"data"`, `"offsets"`, `"index"`).
#         lazy (bool): If True, read the array or its partitions on demand (as
#             #ak.layout.VirtualArray, possibly in #ak.partition.PartitionedArray
#             if `num_partitions` is not None); if False, read all requested data
#             immediately. Any RecordArray child nodes will additionally be
#             read on demand.
#         lazy_cache (None, "new", or MutableMapping): If lazy, pass this
#             cache to the VirtualArrays. If "new", a new dict (keep-forever cache)
#             is created. If None, no cache is used.
#         lazy_cache_key (None or str): If lazy, pass this cache_key to the
#             VirtualArrays. If None, a process-unique string is constructed.
#         highlevel (bool): If True, return an #ak.Array; otherwise, return
#             a low-level #ak.layout.Content subclass.
#         behavior (None or dict): Custom #ak.behavior for the output array, if
#             high-level.

#     Reconstitutes an Awkward Array from a Form, length, and a collection of memory
#     buffers, so that data can be losslessly read from file formats and storage
#     devices that only map names to binary blobs (such as a filesystem directory).

#     The first three arguments of this function are the return values of
#     #ak.to_buffers, so a full round-trip is

#         >>> reconstituted = ak.from_buffers(*ak.to_buffers(original))

#     The `container` argument lets you specify your own Mapping, which might be
#     an interface to some storage format or device (e.g. h5py). It's okay if
#     the `container` dropped NumPy's `dtype` and `shape` information, leaving
#     raw bytes, since `dtype` and `shape` can be reconstituted from the
#     #ak.forms.NumpyForm.

#     The `key_format` should be the same as the one used in #ak.to_buffers.

#     The arguments that begin with `lazy_` are only needed if `lazy` is True.
#     The `lazy_cache` and `lazy_cache_key` determine how the array or its
#     partitions are cached after being read from the `container` (in a no-eviction
#     dict attached to the output #ak.Array as `cache` if not specified).

#     See #ak.to_buffers for examples.
#     """

#     if isinstance(form, str) or (ak._v2._util.py27 and isinstance(form, ak._v2._util.unicode)):
#         form = ak.forms.Form.fromjson(form)
#     elif isinstance(form, dict):
#         form = ak.forms.Form.fromjson(json.dumps(form))

#     if isinstance(key_format, str):

#         def generate_key_format(key_format):
#             def kf(**v):
#                 return key_format.format(**v)

#             return kf

#         key_format = generate_key_format(key_format)

#     hold_cache = None
#     if lazy:
#         form = _wrap_record_with_virtual(form)

#         if lazy_cache == "new":
#             hold_cache = ak._v2._util.MappingProxy({})
#             lazy_cache = ak._v2.contents.ArrayCache(hold_cache)
#         elif lazy_cache is not None and not isinstance(
#             lazy_cache, ak._v2.contents.ArrayCache
#         ):
#             hold_cache = ak._v2._util.MappingProxy.maybe_wrap(lazy_cache)
#             if not isinstance(hold_cache, MutableMapping):
#                 raise TypeError("lazy_cache must be a MutableMapping")
#             lazy_cache = ak._v2.contents.ArrayCache(hold_cache)

#         if lazy_cache_key is None:
#             lazy_cache_key = "ak.from_buffers:{0}".format(_from_buffers_key())

#     if length is None or isinstance(length, (numbers.Integral, np.integer)):
#         if length is None:
#             raise TypeError(
#                 "length must be an integer or an iterable of integers"
#
#             )

#         args = (form, container, str(partition_start), key_format, length)

#         if lazy:
#             generator = ak._v2.contents.ArrayGenerator(
#                 _form_to_layout,
#                 args + (lazy_cache, lazy_cache_key),
#                 form=form,
#                 length=length,
#             )
#             out = ak._v2.contents.VirtualArray(generator, lazy_cache, lazy_cache_key)

#         else:
#             out = _form_to_layout(*(args + (None, None)))

#     elif isinstance(length, Iterable):
#         partitions = []
#         offsets = [0]

#         for part, partlen in enumerate(length):
#             partnum = str(partition_start + part)
#             args = (form, container, partnum, key_format)

#             if lazy:
#                 lazy_cache_key_part = "{0}[{1}]".format(lazy_cache_key, partnum)
#                 generator = ak._v2.contents.ArrayGenerator(
#                     _form_to_layout,
#                     args + (partlen, lazy_cache, lazy_cache_key_part),
#                     form=form,
#                     length=length[part],
#                 )

#                 partitions.append(
#                     ak._v2.contents.VirtualArray(generator, lazy_cache, lazy_cache_key_part)
#                 )
#                 offsets.append(offsets[-1] + length[part])

#             else:
#                 partitions.append(_form_to_layout(*(args + (partlen, None, None))))
#                 offsets.append(offsets[-1] + len(partitions[-1]))

#         out = ak.partition.IrregularlyPartitionedArray(partitions, offsets[1:])

#     else:
#         raise TypeError(
#             "length must be an integer or an iterable of integers, not "
#             + repr(length)
#
#         )

#     return ak._v2._util.maybe_wrap(out, behavior, highlevel)


# _index_form_to_dtype = _index_form_to_index = _form_to_layout_class = None


# def _asbuf(obj):
#     try:
#         tmp = numpy.asarray(obj)
#     except Exception:
#         return numpy.frombuffer(obj, np.uint8)
#     else:
#         return tmp.reshape(-1).view(np.uint8)


# def _form_to_layout(
#     form,
#     container,
#     partnum,
#     key_format,
#     length,
#     lazy_cache,
#     lazy_cache_key,
# ):
#     global _index_form_to_dtype, _index_form_to_index, _form_to_layout_class

#     if _index_form_to_dtype is None:
#         _index_form_to_dtype = {
#             "i8": np.dtype("<i1"),
#             "u8": np.dtype("<u1"),
#             "i32": np.dtype("<i4"),
#             "u32": np.dtype("<u4"),
#             "i64": np.dtype("<i8"),
#         }

#         _index_form_to_index = {
#             "i8": ak._v2.index.Index8,
#             "u8": ak._v2.index.IndexU8,
#             "i32": ak._v2.index.Index32,
#             "u32": ak._v2.index.IndexU32,
#             "i64": ak._v2.index.Index64,
#         }

#         _form_to_layout_class = {
#             (ak.forms.IndexedForm, "i32"): ak._v2.contents.IndexedArray32,
#             (ak.forms.IndexedForm, "u32"): ak._v2.contents.IndexedArrayU32,
#             (ak.forms.IndexedForm, "i64"): ak._v2.contents.IndexedArray64,
#             (ak.forms.IndexedOptionForm, "i32"): ak._v2.contents.IndexedOptionArray32,
#             (ak.forms.IndexedOptionForm, "i64"): ak._v2.contents.IndexedOptionArray64,
#             (ak.forms.ListForm, "i32"): ak._v2.contents.ListArray32,
#             (ak.forms.ListForm, "u32"): ak._v2.contents.ListArrayU32,
#             (ak.forms.ListForm, "i64"): ak._v2.contents.ListArray64,
#             (ak.forms.ListOffsetForm, "i32"): ak._v2.contents.ListOffsetArray32,
#             (ak.forms.ListOffsetForm, "u32"): ak._v2.contents.ListOffsetArrayU32,
#             (ak.forms.ListOffsetForm, "i64"): ak._v2.contents.ListOffsetArray64,
#             (ak.forms.UnionForm, "i32"): ak._v2.contents.UnionArray8_32,
#             (ak.forms.UnionForm, "u32"): ak._v2.contents.UnionArray8_U32,
#             (ak.forms.UnionForm, "i64"): ak._v2.contents.UnionArray8_64,
#         }

#     if form.has_identities:
#         raise NotImplementedError(
#             "ak.from_buffers for an array with Identities"
#
#         )
#     else:
#         identities = None

#     parameters = form.parameters
#     fk = form.form_key

#     if isinstance(form, ak.forms.BitMaskedForm):
#         raw_mask = _asbuf(
#             container[key_format(form_key=fk, attribute="mask", partition=partnum)]
#         )
#         mask = _index_form_to_index[form.mask](
#             raw_mask.view(_index_form_to_dtype[form.mask])
#         )

#         content = _form_to_layout(
#             form.content,
#             container,
#             partnum,
#             key_format,
#             length,
#             lazy_cache,
#             lazy_cache_key,
#         )

#         if length is None:
#             length = len(content)
#         if length > len(mask) * 8:
#             raise ValueError(
#                 "mask is too short for BitMaskedArray: content length "
#                 "is {0}, mask length * 8 is {1}".format(length, len(mask) * 8)
#
#             )

#         return ak._v2.contents.BitMaskedArray(
#             mask,
#             content,
#             form.valid_when,
#             length,
#             form.lsb_order,
#             identities,
#             parameters,
#         )

#     elif isinstance(form, ak.forms.ByteMaskedForm):
#         raw_mask = _asbuf(
#             container[key_format(form_key=fk, attribute="mask", partition=partnum)]
#         )
#         mask = _index_form_to_index[form.mask](
#             raw_mask.view(_index_form_to_dtype[form.mask])
#         )

#         if length is None:
#             length = len(mask)
#         elif length > len(mask):
#             raise ValueError(
#                 "mask is too short for ByteMaskedArray: expected {0}, mask length is {1}".format(
#                     length, len(mask)
#                 )
#
#             )

#         content = _form_to_layout(
#             form.content,
#             container,
#             partnum,
#             key_format,
#             length,
#             lazy_cache,
#             lazy_cache_key,
#         )

#         return ak._v2.contents.ByteMaskedArray(
#             mask, content, form.valid_when, identities, parameters
#         )

#     elif isinstance(form, ak.forms.EmptyForm):
#         if length is not None and length != 0:
#             raise ValueError(
#                 "EmptyArray found in node with non-zero expected length: expected {0}".format(
#                     length
#                 )
#
#             )
#         return ak._v2.contents.EmptyArray(identities, parameters)

#     elif isinstance(form, ak.forms.IndexedForm):
#         raw_index = _asbuf(
#             container[key_format(form_key=fk, attribute="index", partition=partnum)]
#         )
#         index = _index_form_to_index[form.index](
#             raw_index.view(_index_form_to_dtype[form.index])
#         )

#         if length is None:
#             length = len(index)
#         elif length > len(index):
#             raise ValueError(
#                 "index too short for IndexedArray: expected {0}, index length is {1}".format(
#                     length, len(index)
#                 )
#
#             )

#         content = _form_to_layout(
#             form.content,
#             container,
#             partnum,
#             key_format,
#             0 if len(index) == 0 else numpy.max(index) + 1,
#             lazy_cache,
#             lazy_cache_key,
#         )

#         return _form_to_layout_class[type(form), form.index](
#             index, content, identities, parameters
#         )

#     elif isinstance(form, ak.forms.IndexedOptionForm):
#         raw_index = _asbuf(
#             container[key_format(form_key=fk, attribute="index", partition=partnum)]
#         )
#         index = _index_form_to_index[form.index](
#             raw_index.view(_index_form_to_dtype[form.index])
#         )

#         if length is None:
#             length = len(index)
#         elif length > len(index):
#             raise ValueError(
#                 "index too short for IndexedOptionArray: expected {0}, index length is {1}".format(
#                     length, len(index)
#                 )
#
#             )

#         content = _form_to_layout(
#             form.content,
#             container,
#             partnum,
#             key_format,
#             0 if len(index) == 0 else max(0, numpy.max(index) + 1),
#             lazy_cache,
#             lazy_cache_key,
#         )

#         return _form_to_layout_class[type(form), form.index](
#             index, content, identities, parameters
#         )

#     elif isinstance(form, ak.forms.ListForm):
#         raw_starts = _asbuf(
#             container[key_format(form_key=fk, attribute="starts", partition=partnum)]
#         )
#         starts = _index_form_to_index[form.starts](
#             raw_starts.view(_index_form_to_dtype[form.starts])
#         )
#         raw_stops = _asbuf(
#             container[key_format(form_key=fk, attribute="stops", partition=partnum)]
#         )
#         stops = _index_form_to_index[form.stops](
#             raw_stops.view(_index_form_to_dtype[form.stops])
#         )

#         if length is None:
#             length = len(starts)
#         elif length > len(starts):
#             raise ValueError(
#                 "starts too short for ListArray: expected {0}, starts length is {1}".format(
#                     length, len(starts)
#                 )
#
#             )
#         elif length > len(stops):
#             raise ValueError(
#                 "stops too short for ListArray: expected {0}, stops length is {1}".format(
#                     length, len(stops)
#                 )
#
#             )

#         array_starts = numpy.asarray(starts)
#         if len(array_starts) != length:
#             array_starts = array_starts[:length]
#         array_stops = numpy.asarray(stops)
#         if len(array_stops) != length:
#             array_stops = array_stops[:length]
#         array_stops = array_stops[array_starts != array_stops]
#         content = _form_to_layout(
#             form.content,
#             container,
#             partnum,
#             key_format,
#             0 if len(array_stops) == 0 else numpy.max(array_stops),
#             lazy_cache,
#             lazy_cache_key,
#         )

#         return _form_to_layout_class[type(form), form.starts](
#             starts, stops, content, identities, parameters
#         )

#     elif isinstance(form, ak.forms.ListOffsetForm):
#         raw_offsets = _asbuf(
#             container[key_format(form_key=fk, attribute="offsets", partition=partnum)]
#         )
#         offsets = _index_form_to_index[form.offsets](
#             raw_offsets.view(_index_form_to_dtype[form.offsets])
#         )

#         if length is None:
#             length = len(offsets) - 1
#         elif length > len(offsets) - 1:
#             raise ValueError(
#                 "offsets too short for ListOffsetArray: expected {0}, offsets length - 1 is {1}".format(
#                     length, len(offsets) - 1
#                 )
#
#             )

#         content = _form_to_layout(
#             form.content,
#             container,
#             partnum,
#             key_format,
#             offsets[-1],
#             lazy_cache,
#             lazy_cache_key,
#         )

#         return _form_to_layout_class[type(form), form.offsets](
#             offsets, content, identities, parameters
#         )

#     elif isinstance(form, ak.forms.NumpyForm):
#         raw_array = _asbuf(
#             container[key_format(form_key=fk, attribute="data", partition=partnum)]
#         )
#         dtype_inner_shape = form.to_numpy()
#         if dtype_inner_shape.subdtype is None:
#             dtype, inner_shape = dtype_inner_shape, ()
#         else:
#             dtype, inner_shape = dtype_inner_shape.subdtype

#         # If the buffer is required to have a length, check that it's not too short.
#         # If the array has no itemsize, then the buffer can never be "too short"!
#         if (length is not None) and (dtype_inner_shape.itemsize > 0):
#             actual = len(raw_array) // dtype_inner_shape.itemsize
#             if length > actual:
#                 raise ValueError(
#                     "buffer is too short for NumpyArray: expected {0}, buffer "
#                     "has {1} items ({2} bytes)".format(length, actual, len(raw_array))
#
#                 )

#         # NumPy can only infer the length of the array from the buffer
#         # if the inner shape has a nonzero item-size.
#         if dtype_inner_shape.itemsize:
#             leading_shape = (-1,)
#         elif length is not None:
#             leading_shape = (length,)
#         # Given that `length=None` only occurs when reading legacy data
#         # This already failed, so we can fail if `length` is not given
#         else:
#             raise ValueError(
#                 "buffer is empty, and no length was given. It is not possible to "
#                 "determine the correct size of the array from the buffer alone."
#             )

#         array = raw_array.view(dtype).reshape(leading_shape + inner_shape)

#         return ak._v2.contents.NumpyArray(array, identities, parameters)

#     elif isinstance(form, ak.forms.RecordForm):
#         items = list(form.contents.items())
#         if form.istuple:
#             items.sort(key=lambda x: int(x[0]))
#         contents = []
#         minlength = None
#         keys = []
#         for key, content_form in items:
#             keys.append(key)
#             content = _form_to_layout(
#                 content_form,
#                 container,
#                 partnum,
#                 key_format,
#                 length,
#                 lazy_cache,
#                 lazy_cache_key,
#             )
#             if minlength is None:
#                 minlength = len(content)
#             else:
#                 minlength = min(minlength, len(content))
#             contents.append(content)

#         if length is None:
#             length = minlength
#         elif minlength is not None and length > minlength:
#             raise ValueError(
#                 "RecordArray length mismatch: expected {0}, minimum content is {1}".format(
#                     length, minlength
#                 )
#
#             )

#         return ak._v2.contents.RecordArray(
#             contents,
#             None if form.istuple else keys,
#             length,
#             identities,
#             parameters,
#         )

#     elif isinstance(form, ak.forms.RegularForm):
#         if length is None:
#             length = 0

#         content = _form_to_layout(
#             form.content,
#             container,
#             partnum,
#             key_format,
#             length * form.size,
#             lazy_cache,
#             lazy_cache_key,
#         )

#         return ak._v2.contents.RegularArray(
#             content, form.size, length, identities, parameters
#         )

#     elif isinstance(form, ak.forms.UnionForm):
#         raw_tags = _asbuf(
#             container[key_format(form_key=fk, attribute="tags", partition=partnum)]
#         )
#         tags = _index_form_to_index[form.tags](
#             raw_tags.view(_index_form_to_dtype[form.tags])
#         )
#         raw_index = _asbuf(
#             container[key_format(form_key=fk, attribute="index", partition=partnum)]
#         )
#         index = _index_form_to_index[form.index](
#             raw_index.view(_index_form_to_dtype[form.index])
#         )

#         if length is None:
#             length = len(tags)
#         elif length > len(tags):
#             raise ValueError(
#                 "tags too short for UnionArray: expected {0}, tags length is {1}".format(
#                     length, len(tags)
#                 )
#
#             )
#         elif length > len(index):
#             raise ValueError(
#                 "index too short for UnionArray: expected {0}, index length is {1}".format(
#                     length, len(index)
#                 )
#
#             )

#         array_tags = numpy.asarray(tags)
#         if len(array_tags) != length:
#             array_tags = array_tags[:length]
#         array_index = numpy.asarray(index)
#         if len(array_index) != length:
#             array_index = array_index[:length]

#         contents = []
#         for i, content_form in enumerate(form.contents):
#             mine = array_index[numpy.equal(array_tags, i)]
#             contents.append(
#                 _form_to_layout(
#                     content_form,
#                     container,
#                     partnum,
#                     key_format,
#                     0 if len(mine) == 0 else numpy.max(mine) + 1,
#                     lazy_cache,
#                     lazy_cache_key,
#                 )
#             )

#         return _form_to_layout_class[type(form), form.index](
#             tags, index, contents, identities, parameters
#         )

#     elif isinstance(form, ak.forms.UnmaskedForm):
#         content = _form_to_layout(
#             form.content,
#             container,
#             partnum,
#             key_format,
#             length,
#             lazy_cache,
#             lazy_cache_key,
#         )

#         return ak._v2.contents.UnmaskedArray(content, identities, parameters)

#     elif isinstance(form, ak.forms.VirtualForm):
#         args = (
#             form.form,
#             container,
#             partnum,
#             key_format,
#             length,
#             lazy_cache,
#             lazy_cache_key,
#         )
#         generator = ak._v2.contents.ArrayGenerator(
#             _form_to_layout,
#             args,
#             form=form.form,
#             length=length,
#         )
#         node_cache_key = key_format(
#             form_key=form.form.form_key, attribute="virtual", partition=partnum
#         )

#         if isinstance(form.form, (ak.forms.NumpyForm, ak.forms.EmptyForm)):
#             # If it's a leaf node, the node_cache_key completely determines
#             # uniqueness of the subtree (because it's the whole subtree).
#             nested_cache_key = "{0}({1})".format(lazy_cache_key, node_cache_key)
#         else:
#             # Otherwise, the node_cache_key for the root of the subtree might
#             # be the same in two places whereas the nested content differs.
#             nested_cache_key = "{0}({1}:{2})".format(
#                 lazy_cache_key, node_cache_key, _from_buffers_key()
#             )

#         return ak._v2.contents.VirtualArray(generator, lazy_cache, nested_cache_key)

#     else:
#         raise AssertionError(
#             "unexpected form node type: "
#             + str(type(form))
#
#         )


# _from_buffers_key_number = 0
# _from_buffers_key_lock = threading.Lock()


# def _from_buffers_key():
#     global _from_buffers_key_number
#     with _from_buffers_key_lock:
#         out = _from_buffers_key_number
#         _from_buffers_key_number += 1
#     return out


# def _wrap_record_with_virtual(input_form):
#     def modify(form):
#         if form["class"] == "RecordArray":
#             for item in form["contents"].values():
#                 modify(item)
#         elif form["class"].startswith("UnionArray"):
#             for item in form["contents"]:
#                 modify(item)
#         elif "content" in form:
#             modify(form["content"])

#         if form["class"] == "RecordArray":
#             for key in form["contents"].keys():
#                 form["contents"][key] = {
#                     "class": "VirtualArray",
#                     "has_length": True,
#                     "form": form["contents"][key],
#                 }

#     form = json.loads(input_form.tojson())
#     modify(form)
#     return ak.forms.Form.fromjson(json.dumps(form))
