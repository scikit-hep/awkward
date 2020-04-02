ak.layout.Content
-----------------

Superclass of array nodes that build up the structure of an
:doc:`_auto/ak.Array`; the array layout returned by the
`ak.Array.layout <_auto/ak.Array.html#ak-array-layout>`_ property.

The array node types are listed below.

   * :doc:`ak.layout.EmptyArray`: array with zero length and
     :doc:`ak.types.UnknownType`.
   * :doc:`ak.layout.NumpyArray`: complete equivalent to a NumPy np.ndarray,
     usually with :doc:`ak.types.PrimitiveType`.
   * :doc:`ak.layout.RegularArray`: nested lists of equal length; has
     :doc:`ak.types.RegularType`.
   * :doc:`ak.layout.ListArray` (64-bit, 32-bit, and unsigned 32-bit
     specializations): nested lists of any length (built with two indexes,
     ``starts`` and ``stops``), has :doc:`ak.types.ListType`.
   * :doc:`ak.layout.ListOffsetArray` (64-bit, 32-bit, and unsigned 32-bit
     specializations): nested lists of any length (built with one index,
     ``offsets``), has :doc:`ak.types.ListType`.
   * :doc:`ak.layout.RecordArray`: array of records or tuples; has
     :doc:`ak.types.RecordType`.
   * :doc:`ak.layout.IndexedArray` (64-bit, 32-bit, and unsigned 32-bit
     specializations): contains any array and reorders/duplicates it with an
     arbitrary integer ``index``; has the same type as its content.
   * :doc:`ak.layout.IndexedOptionArray` (64-bit and 32-bit specializations):
     same as above but ``-1`` values in the ``index`` are interpreted as None;
     has :doc:`ak.types.OptionType`.
   * :doc:`ak.layout.ByteMaskedArray`: boolean bytes in a ``mask`` are
     interpreted as valid contents or None; has :doc:`ak.types.OptionType`.
   * :doc:`ak.layout.BitMaskedArray`: boolean bits in a ``mask`` are
     interpreted as valid contents or None; has :doc:`ak.types.OptionType`.
   * :doc:`ak.layout.UnmaskedArray`: formally has :doc:`ak.types.OptionType`,
     but all data are valid (no ``mask``).
   * :doc:`ak.layout.UnionArray` (8-bit signed ``tags`` with 64-bit, 32-bit, and
     unsigned 32-bit ``index`` specializations): heterogeneous data represented
     as a tagged union; has :doc:`ak.types.UnionType`.

In Python, :doc:`ak.layout.Record` is not a subclass of
:doc:`ak.layout.Content` (though it is in C++ for technical reasons).

All :doc:`ak.layout.Content` nodes have the following properties and methods
in common.

ak.layout.Content.identities
============================

.. py:attribute:: ak.layout.Content.identities

Returns the :doc:`ak.layout.Identities` object associated with this array node
(if any).

ak.layout.Content.identity
==========================

.. py:attribute:: ak.layout.Content.identity

Returns the single element of an :doc:`ak.layout.Identities` associated with
this array node (if any).

ak.layout.Content.nbytes
========================

.. py:attribute:: ak.layout.Content.nbytes

The total number of bytes in all the :doc:`ak.layout.Index`,
:doc:`ak.layout.Identities`, and :doc:`ak.layout.NumpyArray` buffers in this
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

ak.layout.Content.numfields
===========================

.. py:attribute:: ak.layout.Content.numfields

Number of fields in the outermost records or tuples, or `-1` if the array does
not contain records or tuples.

ak.layout.Content.parameters
============================

.. py:attribute:: ak.layout.Content.parameters

Free-form parameters associated with every array node as a dict from parameter
name to its JSON-like value. Some parameters are special and are used to assign
behaviors to the data.

Note that the dict returned by this property is a *copy* of the array node's
parameters. *Changing the dict will not change the array!*

See :doc:`ak.behavior` and :doc:`_auto/ak.Array`.

ak.layout.Content.purelist_depth
================================

.. py:attribute:: ak.layout.Content.purelist_depth

Number of dimensions of nested lists, not counting anything deeper than the
first record or tuple layer, if any. The depth of a one-dimensional array is
`1`.

If the array contains :doc:`ak.types.UnionType` data and its contents have
equal depths, the return value is that depth. If they do not have equal
depths, the return value is `-1`.

ak.layout.Content.purelist_isregular
====================================

.. py:attribute:: ak.layout.Content.purelist_isregular

Returns True if all dimensions down to the first record or tuple layer have
:doc:`ak.types.RegularType`; False otherwise.

ak.layout.Content.__getitem__
=============================

.. py:method:: ak.layout.Content.__getitem__(where)

See `ak.Array.__getitem__ <_auto/ak.Array.html#ak-array-getitem>`_.

ak.layout.Content.__iter__
==========================

.. py:method:: ak.layout.Content.__iter__()

See `ak.Array.__iter__ <_auto/ak.Array.html#ak-array-iter>`_.

ak.layout.Content.__len__
=========================

.. py:method:: ak.layout.Content.__len__()

See `ak.Array.__len__ <_auto/ak.Array.html#ak-array-len>`_.

ak.layout.Content.__repr__
==========================

.. py:method:: ak.layout.Content.__repr__()

A multi-line XML representation of the array structure.

See (for contrast) `ak.Array.__repr__ <_auto/ak.Array.html#ak-array-repr>`_.

ak.layout.Content.all
=====================

.. py:method:: ak.layout.Content.all(axis=-1, mask=False, keepdims=False)

Implements :doc:`_auto/ak.all`.

ak.layout.Content.any
=====================

.. py:method:: ak.layout.Content.any(axis=-1, mask=False, keepdims=False)

Implements :doc:`_auto/ak.any`.

ak.layout.Content.argmax
========================

.. py:method:: ak.layout.Content.argmax(axis=-1, mask=True, keepdims=False)

Implements :doc:`_auto/ak.argmax`.

ak.layout.Content.argmin
========================

.. py:method:: ak.layout.Content.argmin(axis=-1, mask=True, keepdims=False)

Implements :doc:`_auto/ak.argmin`.

ak.layout.Content.combinations
==============================

.. py:method:: ak.layout.Content.combinations(n, diagonal=False, keys=None, parameters=None, axis=1)

Implements :doc:`_auto/ak.combinations`.

ak.layout.Content.count
=======================

.. py:method:: ak.layout.Content.count(axis=-1, mask=False, keepdims=False)

Implements :doc:`_auto/ak.count`.

ak.layout.Content.count_nonzero
===============================

.. py:method:: ak.layout.Content.count_nonzero(axis=-1, mask=False, keepdims=False)

Implements :doc:`_auto/ak.count_nonzero`.

ak.layout.Content.deep_copy
===========================

.. py:method:: ak.layout.Content.deep_copy(copyarrays=True, copyindexes=True, copyidentities=True)

Returns a copy of the array node and its children.

   * If ``copyarrays``, then :doc:`ak.layout.NumpyArray` buffers are also
     copied.
   * If ``copyindexes``, then :doc:`ak.layout.Index` buffers are also copied.
   * If ``copyidentities``, then :doc:`ak.layout.Identities` buffers are also
     copied.

If all three flags are False, then only (small) C++ and Pyhton objects are
copied, not (large) array buffers.

ak.layout.Content.fieldindex
============================

.. py:method:: ak.layout.Content.fieldindex(key)

Returns the ``fieldindex`` (int) associated with a ``key`` (str) of the
outermost record or tuple. If the array does not contain records or tuples,
this method raises an error.

ak.layout.Content.fillna
========================

.. py:method:: ak.layout.Content.fillna(value)

Implements :doc:`_auto/ak.fillna`.

ak.layout.Content.flatten
=========================

.. py:method:: ak.layout.Content.flatten(axis=1)

Implements :doc:`_auto/ak.flatten`.

ak.layout.Content.getitem_nothing
=================================

.. py:method:: ak.layout.Content.getitem_nothing()

Returns an empty array with this array structure. Used for a corner-case of
``__getitem__``.

ak.layout.Content.haskey
========================

.. py:method:: ak.layout.Content.haskey(key)

Returns True if the outermost record or tuple has a given ``key``; False
otherwise (including the case of not containing records or tuples).

ak.layout.Content.key
=====================

.. py:method:: ak.layout.Content.key(fieldindex)

Returns the ``key`` (str) associated with a ``fieldindex`` (int) of the
outermost record or tuple. If the array does not contain records or tuples,
this method raises an error.

ak.layout.Content.keys
======================

.. py:method:: ak.layout.Content.keys()

Returns the keys of the outermost record or tuple or an empty list.

ak.layout.Content.localindex
============================

.. py:method:: ak.layout.Content.localindex(axis=1)

Returns nested lists of integers (down to the chosen ``axis``) that count
from `0` to the `length - 1` of the innermost list.

This is used internally to generate :doc:`_auto/ak.argcross` from
:doc:`_auto/ak.cross`, etc.

ak.layout.Content.max
=====================

.. py:method:: ak.layout.Content.max(axis=-1, mask=True, keepdims=False)

Implements :doc:`_auto/ak.max`.

ak.layout.Content.merge
=======================

.. py:method:: ak.layout.Content.merge(other)

Concatenate this array node with the ``other`` array node (``axis=0``) by
sharing buffers; i.e. without using a :doc:`ak.layout.UnionArray`. If this
is not possible, this method raises an error.

ak.layout.Content.merge_as_union
================================

.. py:method:: ak.layout.Content.merge_as_union(other)

Concatenate this array node with the ``other`` array node (``axis=0``) using
a :doc:`ak.layout.UnionArray` instead of attempting to share buffers.

ak.layout.Content.mergeable
===========================

.. py:method:: ak.layout.Content.mergeable(other, mergebool=False)

If True, this array node can be concatenated (``axis=0``) with the ``other``
array node without resorting to a :doc:`ak.layout.UnionArray`; otherwise,
they cannot.

If ``mergebool`` is True, consider booleans to be a numeric type that can
be merged with numeric arrays (integers and floating-point).

ak.layout.Content.min
=====================

.. py:method:: ak.layout.Content.min(axis=-1, mask=True, keepdims=False)

Implements :doc:`_auto/ak.min`.

ak.layout.Content.num
=====================

.. py:method:: ak.layout.Content.num(axis=1)

Implements :doc:`_auto/ak.num`.

ak.layout.Content.offsets_and_flatten
=====================================

.. py:method:: ak.layout.Content.offsets_and_flatten(axis=1)

Implements :doc:`_auto/ak.flatten`, though it returns a set of ``offsets``
along with the flattened array.

ak.layout.Content.parameter
===========================

.. py:method:: ak.layout.Content.parameter(key)

Get one parameter by its ``key`` (outermost node only). If a ``key`` is not
found, None is returned.

ak.layout.Content.prod
======================

.. py:method:: ak.layout.Content.prod(axis=-1, mask=False, keepdims=False)

Implements :doc:`_auto/ak.prod`.

ak.layout.Content.purelist_parameter
====================================

.. py:method:: ak.layout.Content.purelist_parameter(key)

Return the value of the outermost parameter matching ``key`` in a sequence
of nested lists, stopping at the first record or tuple layer.

If a layer has :doc:`ak.types.UnionType`, the value is only returned if all
possibilities have the same value.

ak.layout.Content.rpad
======================

.. py:method:: ak.layout.Content.rpad(arg0, arg1)

Implements :doc:`_auto/ak.rpad` with ``clip=False``.

ak.layout.Content.rpad_and_clip
===============================

.. py:method:: ak.layout.Content.rpad_and_clip(arg0, arg1)

Implements :doc:`_auto/ak.rpad` with ``clip=True``.

ak.layout.Content.setidentities
===============================

.. py:method:: ak.layout.Content.setidentities()

.. py:method:: ak.layout.Content.setidentities(identities)

Sets identities in-place.

**Do not use this function:** it is deprecated and will be removed. Assign
:doc:`ak.layout.Identities` in the constructor only.

ak.layout.Content.setparameter
==============================

.. py:method:: ak.layout.Content.setparameter(key, value)

Sets one parameter in-place.

**Do not use this function:** it is deprecated and will be removed. Assign
parameters in the constructor only.

ak.layout.Content.simplify
==========================

.. py:method:: ak.layout.Content.simplify()

Flattens one extraneous level of :doc:`ak.types.OptionType` or
:doc:`ak.types.UnionType`. If there is no such level, this is a pass-through.
In all cases, the output has the same logical meaning as the input.

ak.layout.Content.sum
=====================

.. py:method:: ak.layout.Content.sum(axis=-1, mask=False, keepdims=False)

Implements :doc:`_auto/ak.sum`.

ak.layout.Content.toRegularArray
================================

.. py:method:: ak.layout.Content.toRegularArray()

Converts the data to a :doc:`ak.layout.RegularArray`, if possible.

ak.layout.Content.tojson
========================

.. py:method:: ak.layout.Content.tojson(pretty=False, maxdecimals=None)

Converts this array node to JSON and returns it as a string.

See :doc:`_auto/ak.to_json`.

.. py:method:: ak.layout.Content.tojson(destination, pretty=False, maxdecimals=None, buffersize=65536)

Converts this array node to JSON and writes it to a file (``destination``).

See :doc:`_auto/ak.to_json`.

ak.layout.Content.type
======================

.. py:method:: ak.layout.Content.type()

Returns the high-level :doc:`ak.types.Type` of this array node.
