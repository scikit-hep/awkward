.. py:currentmodule:: ak.layout

ak.layout.Record
----------------

Represents a single value from a :class:`ak.layout.RecordArray`.

As this is a columnar representation, the Record contains a
:class:`ak.layout.RecordArray`, rather than the other way around.
Its two fields are

   * ``array``: the :class:`ak.layout.RecordArray` and
   * ``at``: the index posiion where this Record is found.

The Record shares a reference with its :class:`ak.layout.RecordArray`;
it is not a copy (not even a shallow-copied node object).

A :class:`ak.layout.Record` node has the following properties and methods.
See their definitions in :class:`ak.layout.Content` and :class:`ak.layout.RecordArray`
as a guide. They pass through to the underlying ``array`` with adjustments
in some cases because Record is a scalar.

.. py:class:: Record(array, at)

.. py:method:: Record.__init__(array, at)

.. py:attribute:: Record.array

.. py:attribute:: Record.at

.. py:method:: Record.__getitem__(where)

.. py:method:: Record.__repr__()

.. py:method:: Record.field(index)

.. py:method:: Record.field(key)

.. py:method:: Record.fieldindex(key)

.. py:method:: Record.fielditems()

.. py:method:: Record.fields()

.. py:method:: Record.haskey(key)

.. py:method:: Record.keys()

.. py:method:: Record.parameter(arg0)

.. py:method:: Record.purelist_parameter(arg0)

.. py:method:: Record.setparameter(arg0, arg1)

.. py:method:: Record.simplify()

.. py:method:: Record.tojson(pretty=False, maxdecimals=None)

.. py:method:: Record.tojson(destination, pretty=False, maxdecimals=None, buffersize=65536)

.. py:method:: Record.type(arg0)

.. py:attribute:: Record.astuple

.. py:attribute:: Record.identities

.. py:attribute:: Record.identity

.. py:attribute:: Record.istuple

.. py:attribute:: Record.numfields

.. py:attribute:: Record.parameters
