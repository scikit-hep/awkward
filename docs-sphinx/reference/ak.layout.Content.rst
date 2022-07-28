ak.layout.Content
-----------------

.. py:currentmodule:: ak.layout

Superclass of array nodes that build up the structure of an
:class:`ak.Array`; the array layout returned by the
`ak.Array.layout <_auto/ak.Array.html#ak-array-layout>`__ property.

The array node types are listed below.

   * :class:`ak.layout.EmptyArray`: array with zero length and
     :doc:`ak.types.UnknownType`.
   * :class:`ak.layout.NumpyArray`: complete equivalent to a NumPy np.ndarray,
     usually with :doc:`ak.types.PrimitiveType`.
   * :class:`ak.layout.RegularArray`: nested lists of equal length; has
     :doc:`ak.types.RegularType`.
   * :class:`ak.layout.ListArray` (64-bit, 32-bit, and unsigned 32-bit
     specializations): nested lists of any length (built with two indexes,
     ``starts`` and ``stops``), has :doc:`ak.types.ListType`.
   * :class:`ak.layout.ListOffsetArray` (64-bit, 32-bit, and unsigned 32-bit
     specializations): nested lists of any length (built with one index,
     ``offsets``), has :doc:`ak.types.ListType`.
   * :class:`ak.layout.RecordArray`: array of records or tuples; has
     :doc:`ak.types.RecordType`.
   * :class:`ak.layout.IndexedArray` (64-bit, 32-bit, and unsigned 32-bit
     specializations): contains any array and reorders/duplicates it with an
     arbitrary integer ``index``; has the same type as its content.
   * :class:`ak.layout.IndexedOptionArray` (64-bit and 32-bit specializations):
     same as above but ``-1`` values in the ``index`` are interpreted as None;
     has :doc:`ak.types.OptionType`.
   * :class:`ak.layout.ByteMaskedArray`: boolean bytes in a ``mask`` are
     interpreted as valid contents or None; has :doc:`ak.types.OptionType`.
   * :class:`ak.layout.BitMaskedArray`: boolean bits in a ``mask`` are
     interpreted as valid contents or None; has :doc:`ak.types.OptionType`.
   * :class:`ak.layout.UnmaskedArray`: formally has :doc:`ak.types.OptionType`,
     but all data are valid (no ``mask``).
   * :class:`ak.layout.UnionArray` (8-bit signed ``tags`` with 64-bit, 32-bit, and
     unsigned 32-bit ``index`` specializations): heterogeneous data represented
     as a tagged union; has :doc:`ak.types.UnionType`.

In Python, :class:`ak.layout.Record` is not a subclass of
:class:`ak.layout.Content` (though it is in C++ for technical reasons).

All :class:`ak.layout.Content` nodes have the following properties and methods
in common.

.. py:class:: Content

.. _ak.layout.Content.identities:

.. py:attribute:: Content.identities

Returns the :class:`ak.layout.Identities` object associated with this array node
(if any).

.. _ak.layout.Content.identity:

.. py:attribute:: Content.identity

Returns the single element of an :class:`ak.layout.Identities` associated with
this array node (if any).

.. _ak.layout.Content.nbytes:

.. py:attribute:: Content.nbytes

The total number of bytes in all the :class:`ak.layout.Index`,
:class:`ak.layout.Identities`, and :class:`ak.layout.NumpyArray` buffers in this
array tree.

Note: this calculation takes overlapping buffers into account, to the
extent that overlaps are not double-counted, but overlaps are currently
assumed to be complete subsets of one another, and so it is
theoretically possible (though unlikely) that this number is an
underestimate of the true usage.

It also does not count buffers that must be kept in memory because
of ownership, but are not directly used in the array. Nor does it count
the (small) C++ nodes or Python objects that reference the (large)
array buffers.

.. _ak.layout.Content.numfields:

.. py:attribute:: Content.numfields

Number of fields in the outermost records or tuples, or `-1` if the array does
not contain records or tuples.

.. _ak.layout.Content.parameters:

.. py:attribute:: Content.parameters

Free-form parameters associated with every array node as a dict from parameter
name to its JSON-like value. Some parameters are special and are used to assign
behaviors to the data.

Note that the dict returned by this property is a *copy* of the array node's
parameters. *Changing the dict will not change the array!*

See :doc:`ak.behavior` and :class:`ak.Array`.

.. _ak.layout.Content.purelist_depth:

.. py:attribute:: Content.purelist_depth

Number of dimensions of nested lists, not counting anything deeper than the
first record or tuple layer, if any. The depth of a one-dimensional array is
`1`.

If the array contains :doc:`ak.types.UnionType` data and its contents have
equal depths, the return value is that depth. If they do not have equal
depths, the return value is `-1`.

.. _ak.layout.Content.purelist_isregular:

.. py:attribute:: Content.purelist_isregular

Returns True if all dimensions down to the first record or tuple layer have
:doc:`ak.types.RegularType`; False otherwise.

.. _ak.layout.Content.__getitem__:

.. py:method:: Content.__getitem__(where)

See `ak.Array.__getitem__ <_auto/ak.Array.html#ak-array-getitem>`_.

.. _ak.layout.Content.__iter__:

.. py:method:: Content.__iter__()

See `ak.Array.__iter__ <_auto/ak.Array.html#ak-array-iter>`_.

.. _ak.layout.Content.__len__:

.. py:method:: Content.__len__()

See `ak.Array.__len__ <_auto/ak.Array.html#ak-array-len>`_.

.. _ak.layout.Content.__repr__:

.. py:method:: Content.__repr__()

A multi-line XML representation of the array structure.

See (for contrast) `ak.Array.__repr__ <_auto/ak.Array.html#ak-array-repr>`_.

.. _ak.layout.Content.all:

.. py:method:: Content.all(axis=-1, mask=False, keepdims=False)

Implements :func:`ak.all`.

.. _ak.layout.Content.any:

.. py:method:: Content.any(axis=-1, mask=False, keepdims=False)

Implements :func:`ak.any`.

.. _ak.layout.Content.argmax:

.. py:method:: Content.argmax(axis=-1, mask=True, keepdims=False)

Implements :func:`ak.argmax`.

.. _ak.layout.Content.argmin:

.. py:method:: Content.argmin(axis=-1, mask=True, keepdims=False)

Implements :func:`ak.argmin`.

.. _ak.layout.Content.combinations:

.. py:method:: Content.combinations(n, replacement=False, keys=None, parameters=None, axis=1)

Implements :func:`ak.combinations`.

.. _ak.layout.Content.count:

.. py:method:: Content.count(axis=-1, mask=False, keepdims=False)

Implements :func:`ak.count`.

.. _ak.layout.Content.count_nonzero:

.. py:method:: Content.count_nonzero(axis=-1, mask=False, keepdims=False)

Implements :func:`ak.count_nonzero`.

.. _ak.layout.Content.deep_copy:

.. py:method:: Content.deep_copy(copyarrays=True, copyindexes=True, copyidentities=True)

Returns a copy of the array node and its children.

   * If ``copyarrays``, then :class:`ak.layout.NumpyArray` buffers are also
     copied.
   * If ``copyindexes``, then :class:`ak.layout.Index` buffers are also copied.
   * If ``copyidentities``, then :class:`ak.layout.Identities` buffers are also
     copied.

If all three flags are False, then only (small) C++ and Pyhton objects are
copied, not (large) array buffers.

.. _ak.layout.Content.fieldindex:

.. py:method:: Content.fieldindex(key)

Returns the ``fieldindex`` (int) associated with a ``key`` (str) of the
outermost record or tuple. If the array does not contain records or tuples,
this method raises an error.

.. _ak.layout.Content.fillna:

.. py:method:: Content.fillna(value)

Implements :func:`ak.fill_none`.

.. _ak.layout.Content.flatten:

.. py:method:: Content.flatten(axis=1)

Implements :func:`ak.flatten`.

.. _ak.layout.Content.getitem_nothing:

.. py:method:: Content.getitem_nothing()

Returns an empty array with this array structure. Used for a corner-case of
``__getitem__``.

.. _ak.layout.Content.haskey:

.. py:method:: Content.haskey(key)

Returns True if the outermost record or tuple has a given ``key``; False
otherwise (including the case of not containing records or tuples).

.. _ak.layout.Content.key:

.. py:method:: Content.key(fieldindex)

Returns the ``key`` (str) associated with a ``fieldindex`` (int) of the
outermost record or tuple. If the array does not contain records or tuples,
this method raises an error.

.. _ak.layout.Content.keys:

.. py:method:: Content.keys()

Returns the keys of the outermost record or tuple or an empty list.

.. _ak.layout.Content.localindex:

.. py:method:: Content.localindex(axis=1)

Returns nested lists of integers (down to the chosen ``axis``) that count
from `0` to the `length - 1` of the innermost list.

This is used internally to generate :func:`ak.argcartesian` from
:func:`ak.cartesian`, etc.

.. _ak.layout.Content.max:

.. py:method:: Content.max(axis=-1, mask=True, keepdims=False)

Implements :func:`ak.max`.

.. _ak.layout.Content.merge:

.. py:method:: Content.merge(other)

Concatenate this array node with the ``other`` array node (``axis=0``) by
sharing buffers; i.e. without using a :class:`ak.layout.UnionArray`. If this
is not possible, this method raises an error.

.. _ak.layout.Content.merge_as_union:

.. py:method:: Content.merge_as_union(other)

Concatenate this array node with the ``other`` array node (``axis=0``) using
a :class:`ak.layout.UnionArray` instead of attempting to share buffers.

.. _ak.layout.Content.mergeable:

.. py:method:: Content.mergeable(other, mergebool=False)

If True, this array node can be concatenated (``axis=0``) with the ``other``
array node without resorting to a :class:`ak.layout.UnionArray`; otherwise,
they cannot.

If ``mergebool`` is True, consider booleans to be a numeric type that can
be merged with numeric arrays (integers and floating-point).

.. _ak.layout.Content.min:

.. py:method:: Content.min(axis=-1, mask=True, keepdims=False)

Implements :func:`ak.min`.

.. _ak.layout.Content.num:

.. py:method:: Content.num(axis=1)

Implements :func:`ak.num`.

.. _ak.layout.Content.offsets_and_flatten:

.. py:method:: Content.offsets_and_flatten(axis=1)

Implements :func:`ak.flatten`, though it returns a set of ``offsets``
along with the flattened array.

.. _ak.layout.Content.parameter:

.. py:method:: Content.parameter(key)

Get one parameter by its ``key`` (outermost node only). If a ``key`` is not
found, None is returned.

.. _ak.layout.Content.prod:

.. py:method:: Content.prod(axis=-1, mask=False, keepdims=False)

Implements :func:`ak.prod`.

.. _ak.layout.Content.purelist_parameter:

.. py:method:: Content.purelist_parameter(key)

Return the value of the outermost parameter matching ``key`` in a sequence
of nested lists, stopping at the first record or tuple layer.

If a layer has :doc:`ak.types.UnionType`, the value is only returned if all
possibilities have the same value.

.. _ak.layout.Content.rpad:

.. py:method:: Content.rpad(arg0, arg1)

Implements :func:`ak.pad_none` with ``clip=False``.

.. _ak.layout.Content.rpad_and_clip:

.. py:method:: Content.rpad_and_clip(arg0, arg1)

Implements :func:`ak.pad_none` with ``clip=True``.

.. _ak.layout.Content.setidentities:

.. py:method:: Content.setidentities()

.. py:method:: Content.setidentities(identities)

Sets identities in-place.

**Do not use this function:** it is deprecated and will be removed. Assign
:class:`ak.layout.Identities` in the constructor only.

.. _ak.layout.Content.setparameter:

.. py:method:: Content.setparameter(key, value)

Sets one parameter in-place.

**Do not use this function:** it is deprecated and will be removed. Assign
parameters in the constructor only.

.. _ak.layout.Content.simplify:

.. py:method:: Content.simplify()

Flattens one extraneous level of :doc:`ak.types.OptionType` or
:doc:`ak.types.UnionType`. If there is no such level, this is a pass-through.
In all cases, the output has the same logical meaning as the input.

.. _ak.layout.Content.sum:

.. py:method:: Content.sum(axis=-1, mask=False, keepdims=False)

Implements :func:`ak.sum`.

.. _ak.layout.Content.toRegularArray:

.. py:method:: Content.toRegularArray()

Converts the data to a :class:`ak.layout.RegularArray`, if possible.

.. _ak.layout.Content.tojson:

.. py:method:: Content.tojson(pretty=False, maxdecimals=None)

Converts this array node to JSON and returns it as a string.

See :func:`ak.to_json`.

.. py:method:: Content.tojson(destination, pretty=False, maxdecimals=None, buffersize=65536)

Converts this array node to JSON and writes it to a file (``destination``).

See :func:`ak.to_json`.

.. _ak.layout.Content.type:

.. py:method:: Content.type()

Returns the high-level :doc:`ak.types.Type` of this array node.
