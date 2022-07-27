ak.types.Type
-------------

.. py:currentmodule:: ak.types

Superclass of type nodes that describe a high-level data type (see
:func:`ak.type` and
`ak.Array.layout <_auto/ak.Array.html#ak-array-layout>`_).

Types are rendered as `Datashape <https://datashape.readthedocs.io/>`__ strings,
whenever possible. In some cases, they have more information than this
specification can express, so we extend the Datashape language in natural ways.

The type subclasses are listed below.

   * :doc:`ak.types.ArrayType`: type of a high-level :func:`ak.Array`,
     which includes the length of the array (and hence is not composable with
     other types).
   * :doc:`ak.types.UnknownType`: a type that is not known, for example in
     :doc:`ak.layout.EmptyArray`.
   * :doc:`ak.types.PrimitiveType`: numbers and booleans.
   * :doc:`ak.types.RegularType`: nested lists, each with the same length.
   * :doc:`ak.types.ListType`: nested lists with unconstrained lengths.
   * :doc:`ak.types.RecordType`: records or tuples.
   * :doc:`ak.types.OptionType`: data that might be missing (None in Python).
   * :doc:`ak.types.UnionType`: heterogeneous data; one of several types.

All :doc:`ak.types.Type` instances have the following properties and methods
in common.

.. py:class:: Type

ak.types.Type.__eq__
====================

.. py:method:: Type.__eq__(other)

True if two types are equal; False otherwise.

ak.types.Type.__ne__
====================

.. py:method:: Type.__ne__()

True if two types are not equal; False otherwise.

ak.types.Type.__repr__
======================

.. py:method:: Type.__repr__()

String representation of the type, mostly following the
`Datashape <https://datashape.readthedocs.io/>`__ grammar.

ak.types.Type.__getstate__
==========================

.. py:method:: Type.__getstate__()

Types can be pickled.

ak.types.Type.__setstate__
==========================

.. py:method:: Type.__setstate__(arg0)

Types can be pickled.

ak.types.Type.empty
===================

.. py:method:: Type.empty()

Creates an empty :doc:`ak.layout.Content` array with this type.

ak.types.Type.fieldindex
========================

.. py:method:: Type.fieldindex(key)

Returns the index position of a ``key`` if the type contains
:doc:`ak.types.RecordType` and ``key`` is in the record.

ak.types.Type.haskey
====================

.. py:method:: Type.haskey(key)

Returns True if the type contains :doc:`ak.types.RecordType` and ``key`` is
in the record; False otherwise.

ak.types.Type.key
=================

.. py:method:: Type.key(fieldindex)

Returns the ``key`` name at a given index position in the record if the
type contains :doc:`ak.types.RecordType` with more than ``fieldindex``
fields.

ak.types.Type.keys
==================

.. py:method:: Type.keys()

Returns a list of keys in the record if the type contains
:doc:`ak.types.RecordType`.

ak.types.Type.setparameter
==========================

.. py:method:: Type.setparameter(key, value)

Sets a parameter.

**Do not use this method!** Mutable parameters are deprecated.

ak.types.Type.numfields
=======================

.. py:attribute:: Type.numfields

Returns the number of fields in the record if this type contains a
:doc:`ak.types.RecordType`.

ak.types.Type.parameters
========================

.. py:attribute:: Type.parameters

Returns the parameters associated with this type.

ak.types.Type.typestr
=====================

.. py:attribute:: Type.typestr

Returns the custom type string if overridden with :doc:`ak.behavior`.

See `Custom type names <ak.behavior.html#custom-type-names>`_.
