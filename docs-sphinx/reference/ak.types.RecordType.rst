.. py:currentmodule:: ak.types

ak.types.RecordType
-------------------

The type of a :class:`ak.layout.RecordArray`.

In addition to the properties and methods described in :class:`ak.types.Type`,
a RecordType has the following.

.. py:class:: RecordType(types, keys=None, parameters=None, typestr=None)

    .. py:method:: __init__(types, keys=None, parameters=None, typestr=None)
        
    .. py:attribute:: types
        
    .. py:method:: __getitem__(where)
        
    .. py:method:: field(fieldindex)
        
    .. py:method:: field(key)
        
    .. py:method:: fielditems()
        
    .. py:method:: fields()
        