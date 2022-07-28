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

.. _ak.layout.Record.__init__:

.. py:method:: Record.__init__(array, at)

.. _ak.layout.Record.array:

.. py:attribute:: Record.array

.. _ak.layout.Record.at:

.. py:attribute:: Record.at

.. _ak.layout.Record.__getitem__:

.. py:method:: Record.__getitem__(where)

.. _ak.layout.Record.__repr__:

.. py:method:: Record.__repr__()

.. _ak.layout.Record.field:

.. py:method:: Record.field(index)

.. _ak.layout.Record.field:

.. py:method:: Record.field(key)

.. _ak.layout.Record.fieldindex:

.. py:method:: Record.fieldindex(key)

.. _ak.layout.Record.fielditems:

.. py:method:: Record.fielditems()

.. _ak.layout.Record.fields:

.. py:method:: Record.fields()

.. _ak.layout.Record.haskey:

.. py:method:: Record.haskey(key)

.. _ak.layout.Record.keys:

.. py:method:: Record.keys()

.. _ak.layout.Record.parameter:

.. py:method:: Record.parameter(arg0)

.. _ak.layout.Record.purelist_parameter:

.. py:method:: Record.purelist_parameter(arg0)

.. _ak.layout.Record.setparameter:

.. py:method:: Record.setparameter(arg0, arg1)

.. _ak.layout.Record.simplify:

.. py:method:: Record.simplify()

.. _ak.layout.Record.tojson:

.. py:method:: Record.tojson(pretty=False, maxdecimals=None)

.. _ak.layout.Record.tojson:

.. py:method:: Record.tojson(destination, pretty=False, maxdecimals=None, buffersize=65536)

.. _ak.layout.Record.type:

.. py:method:: Record.type(arg0)

.. _ak.layout.Record.astuple:

.. py:attribute:: Record.astuple

.. _ak.layout.Record.identities:

.. py:attribute:: Record.identities

.. _ak.layout.Record.identity:

.. py:attribute:: Record.identity

.. _ak.layout.Record.istuple:

.. py:attribute:: Record.istuple

.. _ak.layout.Record.numfields:

.. py:attribute:: Record.numfields

.. _ak.layout.Record.parameters:

.. py:attribute:: Record.parameters
