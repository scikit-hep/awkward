.. py:currentmodule:: ak.types

ak.types.UnionType
------------------

The type of a :doc:`ak.layout.UnionArray`.

In addition to the properties and methods described in :doc:`ak.types.Type`,
a UnionType has the following.

.. py:class:: UnionType(types, parameters=None, typestr=None)

ak.types.UnionType.__init__
===========================

.. py:method:: UnionType.__init__(types, parameters=None, typestr=None)

ak.types.UnionType.types
========================

.. py:attribute:: UnionType.types

ak.types.UnionType.__getitem__
==============================

.. py:method:: UnionType.__getitem__(where)

ak.types.UnionType.type
=======================

.. py:method:: UnionType.type(index)

ak.types.UnionType.numtypes
===========================

.. py:attribute:: UnionType.numtypes
