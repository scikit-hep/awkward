ak.types.Type
-------------

.. py:currentmodule:: ak.types

Superclass of type nodes that describe a high-level data type (see
:func:`ak.type` and
`ak.Array.layout <_auto/ak.Array.html#ak-array-layout>`_).

Types are rendered as `Datashape <https://datashape.readthedocs.io/>`__ strings,
whenever possible. In some cases, they have more information than this
specification can express, so we extend the Datashape language in natural ways.

The type subclasses are listed below.

   * :class:`ak.types.ArrayType`: type of a high-level :class:`ak.Array`,
     which includes the length of the array (and hence is not composable with
     other types).
   * :class:`ak.types.UnknownType`: a type that is not known, for example in
     :class:`ak.layout.EmptyArray`.
   * :class:`ak.types.PrimitiveType`: numbers and booleans.
   * :class:`ak.types.RegularType`: nested lists, each with the same length.
   * :class:`ak.types.ListType`: nested lists with unconstrained lengths.
   * :class:`ak.types.RecordType`: records or tuples.
   * :class:`ak.types.OptionType`: data that might be missing (None in Python).
   * :class:`ak.types.UnionType`: heterogeneous data; one of several types.

All :class:`ak.types.Type` instances have the following properties and methods
in common.

.. py:class:: Type

    .. py:method:: Type.__eq__(other)
        
        True if two types are equal; False otherwise.
        
    .. py:method:: Type.__ne__()
        
        True if two types are not equal; False otherwise.
        
    .. py:method:: Type.__repr__()
        
        String representation of the type, mostly following the
        `Datashape <https://datashape.readthedocs.io/>`__ grammar.
        
    .. py:method:: Type.__getstate__()
        
        Types can be pickled.
        
    .. py:method:: Type.__setstate__(arg0)
        
        Types can be pickled.
        
    .. py:method:: Type.empty()
        
        Creates an empty :class:`ak.layout.Content` array with this type.
        
    .. py:method:: Type.fieldindex(key)
        
        Returns the index position of a ``key`` if the type contains
        :class:`ak.types.RecordType` and ``key`` is in the record.
        
    .. py:method:: Type.haskey(key)
        
        Returns True if the type contains :class:`ak.types.RecordType` and ``key`` is
        in the record; False otherwise.
        
    .. py:method:: Type.key(fieldindex)
        
        Returns the ``key`` name at a given index position in the record if the
        type contains :class:`ak.types.RecordType` with more than ``fieldindex``
        fields.
        
    .. py:method:: Type.keys()
        
        Returns a list of keys in the record if the type contains
        :class:`ak.types.RecordType`.
        
    .. py:method:: Type.setparameter(key, value)
        
        Sets a parameter.
        
        **Do not use this method!** Mutable parameters are deprecated.
        
    .. py:attribute:: Type.numfields
        
        Returns the number of fields in the record if this type contains a
        :class:`ak.types.RecordType`.
        
    .. py:attribute:: Type.parameters
        
        Returns the parameters associated with this type.
        
Returns the custom type string if overridden with :data:`ak.behavior`.

See `Custom type names <ak.behavior.html#custom-type-names>`_.
