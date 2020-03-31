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

ak.layout.Content.parameters
============================

.. py:attribute:: ak.layout.Content.parameters

ak.layout.Content.purelist_depth
================================

.. py:attribute:: ak.layout.Content.purelist_depth

ak.layout.Content.purelist_isregular
====================================

.. py:attribute:: ak.layout.Content.purelist_isregular

ak.layout.Content.__getitem__
=============================

.. py:method:: ak.layout.Content.__getitem__(where)

ak.layout.Content.__iter__
==========================

.. py:method:: ak.layout.Content.__iter__()

ak.layout.Content.__len__
=========================

.. py:method:: ak.layout.Content.__len__()

ak.layout.Content.__repr__
==========================

.. py:method:: ak.layout.Content.__repr__()

ak.layout.Content.all
=====================

.. py:method:: ak.layout.Content.all(axis=-1, mask=False, keepdims=False)

ak.layout.Content.any
=====================

.. py:method:: ak.layout.Content.any(axis=-1, mask=False, keepdims=False)

ak.layout.Content.argmax
========================

.. py:method:: ak.layout.Content.argmax(axis=-1, mask=True, keepdims=False)

ak.layout.Content.argmin
========================

.. py:method:: ak.layout.Content.argmin(axis=-1, mask=True, keepdims=False)

ak.layout.Content.broadcast_tooffsets64
=======================================

.. py:method:: ak.layout.Content.broadcast_tooffsets64(arg0)

ak.layout.Content.choose
========================

.. py:method:: ak.layout.Content.choose(n, diagonal=False, keys=None, parameters=None, axis=1)

ak.layout.Content.compact_offsets64
===================================

.. py:method:: ak.layout.Content.compact_offsets64(start_at_zero=True)

ak.layout.Content.count
=======================

.. py:method:: ak.layout.Content.count(axis=-1, mask=False, keepdims=False)

ak.layout.Content.count_nonzero
===============================

.. py:method:: ak.layout.Content.count_nonzero(axis=-1, mask=False, keepdims=False)

ak.layout.Content.deep_copy
===========================

.. py:method:: ak.layout.Content.deep_copy(copyarrays=True, copyindexes=True, copyidentities=True)

ak.layout.Content.fieldindex
============================

.. py:method:: ak.layout.Content.fieldindex(arg0)

ak.layout.Content.fillna
========================

.. py:method:: ak.layout.Content.fillna(arg0)

ak.layout.Content.flatten
=========================

.. py:method:: ak.layout.Content.flatten(axis=1)

ak.layout.Content.getitem_nothing
=================================

.. py:method:: ak.layout.Content.getitem_nothing()

ak.layout.Content.haskey
========================

.. py:method:: ak.layout.Content.haskey(arg0)

ak.layout.Content.key
=====================

.. py:method:: ak.layout.Content.key(arg0)

ak.layout.Content.keys
======================

.. py:method:: ak.layout.Content.keys()

ak.layout.Content.localindex
============================

.. py:method:: ak.layout.Content.localindex(axis=1)

ak.layout.Content.max
=====================

.. py:method:: ak.layout.Content.max(axis=-1, mask=True, keepdims=False)

ak.layout.Content.merge
=======================

.. py:method:: ak.layout.Content.merge(arg0)

ak.layout.Content.merge_as_union
================================

.. py:method:: ak.layout.Content.merge_as_union(arg0)

ak.layout.Content.mergeable
===========================

.. py:method:: ak.layout.Content.mergeable(other, mergebool=False)

ak.layout.Content.min
=====================

.. py:method:: ak.layout.Content.min(axis=-1, mask=True, keepdims=False)

ak.layout.Content.num
=====================

.. py:method:: ak.layout.Content.num(axis=1)

ak.layout.Content.offsets_and_flatten
=====================================

.. py:method:: ak.layout.Content.offsets_and_flatten(axis=1)

ak.layout.Content.parameter
===========================

.. py:method:: ak.layout.Content.parameter(arg0)

ak.layout.Content.prod
======================

.. py:method:: ak.layout.Content.prod(axis=-1, mask=False, keepdims=False)

ak.layout.Content.purelist_parameter
====================================

.. py:method:: ak.layout.Content.purelist_parameter(arg0)

ak.layout.Content.rpad
======================

.. py:method:: ak.layout.Content.rpad(arg0, arg1)

ak.layout.Content.rpad_and_clip
===============================

.. py:method:: ak.layout.Content.rpad_and_clip(arg0, arg1)

ak.layout.Content.setidentities
===============================

.. py:method:: ak.layout.Content.setidentities(arg0)

ak.layout.Content.setparameter
==============================

.. py:method:: ak.layout.Content.setparameter(arg0, arg1)

ak.layout.Content.simplify
==========================

.. py:method:: ak.layout.Content.simplify()

ak.layout.Content.sum
=====================

.. py:method:: ak.layout.Content.sum(axis=-1, mask=False, keepdims=False)

ak.layout.Content.toRegularArray
================================

.. py:method:: ak.layout.Content.toRegularArray()

ak.layout.Content.tojson
========================

.. py:method:: ak.layout.Content.tojson(pretty=False, maxdecimals=None)

ak.layout.Content.tojson
========================

.. py:method:: ak.layout.Content.tojson(destination, pretty=False, maxdecimals=None, buffersize=65536)

ak.layout.Content.type
======================

.. py:method:: ak.layout.Content.type()
