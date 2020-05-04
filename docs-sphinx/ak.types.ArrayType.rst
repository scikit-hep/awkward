ak.types.ArrayType
------------------

Type of a high-level :doc:`_auto/ak.Array`, which includes the length of the
array.

Consider, for instance the distinction between this :doc:`_auto/ak.Array` and
the :doc:`ak.layout.Content` in its
`ak.Array.layout <_auto/ak.Array.html#ak-array-layout>`_:

.. code-block:: python

    >>> array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    >>> ak.type(array)
    3 * var * float64
    >>> ak.type(array.layout)
    var * float64

This distinction is between an :doc:`ak.types.ArrayType` and the
:doc:`ak.types.ListType` it contains.

.. code-block:: python

    >>> type(ak.type(array))
    <class 'awkward1.types.ArrayType'>
    >>> type(ak.type(array.layout))
    <class 'awkward1.types.ListType'>

In addition to the properties and methods described in :doc:`ak.types.Type`,
an ArrayType has the following.

ak.types.ArrayType.__init__
===========================

.. py:method:: ak.types.ArrayType.__init__(type, length, parameters=None, typestr=None)

ak.types.ArrayType.type
=======================

.. py:attribute:: ak.types.ArrayType.type

ak.types.ArrayType.length
=========================

.. py:attribute:: ak.types.ArrayType.length
