ak.types.Type
-------------

Superclass of type nodes that describe a high-level data type (see
:doc:`_auto/ak.Type` and
`ak.Array.layout <_auto/ak.Array.html#ak-array-layout>`_).

Types are rendered as `Datashape <https://datashape.readthedocs.io/>`__ strings.

The type subclasses are listed below.

   * :doc:`ak.types.ArrayType`: type of a high-level :doc:`_auto/ak.Array`,
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

ak.types.Type.__eq__
====================

.. py:method:: ak.types.Type.__eq__()

ak.types.Type.__ne__
====================

.. py:method:: ak.types.Type.__ne__()

ak.types.Type.__repr__
======================

.. py:method:: ak.types.Type.__repr__()

ak.types.Type.__getstate__
==========================

.. py:method:: ak.types.Type.__getstate__()

ak.types.Type.__setstate__
==========================

.. py:method:: ak.types.Type.__setstate__(arg0)

ak.types.Type.empty
===================

.. py:method:: ak.types.Type.empty()

ak.types.Type.fieldindex
========================

.. py:method:: ak.types.Type.fieldindex(key)

ak.types.Type.haskey
====================

.. py:method:: ak.types.Type.haskey(key)

ak.types.Type.key
=================

.. py:method:: ak.types.Type.key(fieldindex)

ak.types.Type.keys
==================

.. py:method:: ak.types.Type.keys()

ak.types.Type.setparameter
==========================

.. py:method:: ak.types.Type.setparameter(arg0, arg1)

ak.types.Type.numfields
=======================

.. py:attribute:: ak.types.Type.numfields

ak.types.Type.parameters
========================

.. py:attribute:: ak.types.Type.parameters

ak.types.Type.typestr
=====================

.. py:attribute:: ak.types.Type.typestr
