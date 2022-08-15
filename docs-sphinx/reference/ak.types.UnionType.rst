.. py:currentmodule:: ak.types

ak.types.UnionType
------------------

The type of a :class:`ak.layout.UnionArray`.

In addition to the properties and methods described in :class:`ak.types.Type`,
a UnionType has the following.

.. py:class:: UnionType(types, parameters=None, typestr=None)

    .. py:method:: UnionType.__init__(types, parameters=None, typestr=None)
        
    .. py:attribute:: UnionType.types
        
    .. py:method:: UnionType.__getitem__(where)
        
    .. py:method:: UnionType.type(index)
        