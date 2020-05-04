ak.types.UnknownType
--------------------

The type of a :doc:`ak.layout.EmptyArray`.

Types can be unknown if created with an :doc:`_auto/ak.ArrayBuilder` and
no instances are encountered to resolve the type.

In addition to the properties and methods described in :doc:`ak.types.Type`,
an UnknownType has the following.

ak.types.UnknownType.__init__
=============================

.. py:method:: ak.types.UnknownType.__init__(type, size, parameters=None, typestr=None)

ak.types.UnknownType.type
=========================

.. py:attribute:: ak.types.UnknownType.type

ak.types.UnknownType.size
=========================

.. py:attribute:: ak.types.UnknownType.size
