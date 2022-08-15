.. py:currentmodule:: ak.types

ak.types.UnknownType
--------------------

The type of a :class:`ak.layout.EmptyArray`.

Types can be unknown if created with an :class:`ak.ArrayBuilder` and
no instances are encountered to resolve the type.

In addition to the properties and methods described in :class:`ak.types.Type`,
an UnknownType has the following.

.. py:class:: UnknownType(parameters=None, typestr=None)

