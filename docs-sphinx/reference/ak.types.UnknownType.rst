.. py:currentmodule:: ak.types

ak.types.UnknownType
--------------------

The type of a :class:`ak.layout.EmptyArray`.

Types can be unknown if created with an :class:`ak.ArrayBuilder` and
no instances are encountered to resolve the type.

In addition to the properties and methods described in :doc:`ak.types.Type`,
an UnknownType has the following.

.. py:class:: UnknownType(parameters=None, typestr=None)

.. _ak.types.UnknownType.__init__:

.. py:method:: UnknownType.__init__(parameters=None, typestr=None)
