.. py:currentmodule:: ak.types

ak.types.UnionType
------------------

The type of a :class:`ak.layout.UnionArray`.

In addition to the properties and methods described in :class:`ak.types.Type`,
a UnionType has the following.

.. py:class:: UnionType(types, parameters=None, typestr=None)

.. _ak.types.UnionType.__init__:

.. py:method:: UnionType.__init__(types, parameters=None, typestr=None)

.. _ak.types.UnionType.types:

.. py:attribute:: UnionType.types

.. _ak.types.UnionType.__getitem__:

.. py:method:: UnionType.__getitem__(where)

.. _ak.types.UnionType.type:

.. py:method:: UnionType.type(index)

.. _ak.types.UnionType.numtypes:

.. py:attribute:: UnionType.numtypes
