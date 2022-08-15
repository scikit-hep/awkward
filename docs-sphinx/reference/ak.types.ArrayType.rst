.. py:currentmodule:: ak.types

ak.types.ArrayType
------------------

Type of a high-level :class:`ak.Array`, which includes the length of the
array.

Consider, for instance the distinction between this :class:`ak.Array` and
the :class:`ak.layout.Content` in its
`ak.Array.layout <_auto/ak.Array.html#ak-array-layout>`_:

.. code-block:: python

    >>> array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    >>> ak.type(array)
    3 * var * float64
    >>> ak.type(array.layout)
    var * float64

This distinction is between an :class:`ak.types.ArrayType` and the
:class:`ak.types.ListType` it contains.

.. code-block:: python

    >>> type(ak.type(array))
    <class 'awkward.types.ArrayType'>
    >>> type(ak.type(array.layout))
    <class 'awkward.types.ListType'>

In addition to the properties and methods described in :class:`ak.types.Type`,
an ArrayType has the following.

.. py:class:: ArrayType(type, length, parameters=None, typestr=None)

    .. py:method:: ArrayType.__init__(type, length, parameters=None, typestr=None)
        
    .. py:attribute:: ArrayType.type
        