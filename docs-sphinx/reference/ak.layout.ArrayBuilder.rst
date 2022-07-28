.. py:currentmodule:: ak.layout

ak.layout.ArrayBuilder
----------------------

The low-level ArrayBuilder that builds :class:`ak.layout.Content` arrays. This
object is wrapped by :class:`ak.ArrayBuilder`.

(Method names in the high-level interface have been chnaged to include
underscores after "begin" and "end," but that hasn't happened in the
low-level interface, yet or possibly at all.)

.. py:class:: ArrayBuilder(initial=1024, resize=1.5)

.. _ak.layout.ArrayBuilder.__getitem__:

.. py:method:: ArrayBuilder.__getitem__(where)

.. _ak.layout.ArrayBuilder.__init__:

.. py:method:: ArrayBuilder.__init__(initial=1024, resize=1.5)

.. _ak.layout.ArrayBuilder.__iter__:

.. py:method:: ArrayBuilder.__iter__()

.. _ak.layout.ArrayBuilder.__len__:

.. py:method:: ArrayBuilder.__len__()

.. _ak.layout.ArrayBuilder.__repr__:

.. py:method:: ArrayBuilder.__repr__()

.. _ak.layout.ArrayBuilder.beginlist:

.. py:method:: ArrayBuilder.beginlist()

.. _ak.layout.ArrayBuilder.beginrecord:

.. py:method:: ArrayBuilder.beginrecord(name=None)

.. _ak.layout.ArrayBuilder.begintuple:

.. py:method:: ArrayBuilder.begintuple(arg0)

.. _ak.layout.ArrayBuilder.boolean:

.. py:method:: ArrayBuilder.boolean(arg0)

.. _ak.layout.ArrayBuilder.bytestring:

.. py:method:: ArrayBuilder.bytestring(arg0)

.. _ak.layout.ArrayBuilder.clear:

.. py:method:: ArrayBuilder.clear()

.. _ak.layout.ArrayBuilder.endlist:

.. py:method:: ArrayBuilder.endlist()

.. _ak.layout.ArrayBuilder.endrecord:

.. py:method:: ArrayBuilder.endrecord()

.. _ak.layout.ArrayBuilder.endtuple:

.. py:method:: ArrayBuilder.endtuple()


.. _ak.layout.ArrayBuilder.field:

.. py:method:: ArrayBuilder.field(arg0)

.. _ak.layout.ArrayBuilder.fromiter:

.. py:method:: ArrayBuilder.fromiter(arg0)

.. _ak.layout.ArrayBuilder.index:

.. py:method:: ArrayBuilder.index(arg0)

.. _ak.layout.ArrayBuilder.integer:

.. py:method:: ArrayBuilder.integer(arg0)

.. _ak.layout.ArrayBuilder.null:

.. py:method:: ArrayBuilder.null()

.. _ak.layout.ArrayBuilder.real:

.. py:method:: ArrayBuilder.real(arg0)

.. _ak.layout.ArrayBuilder.snapshot:

.. py:method:: ArrayBuilder.snapshot()

.. _ak.layout.ArrayBuilder.string:

.. py:method:: ArrayBuilder.string(arg0)

.. _ak.layout.ArrayBuilder.type:

.. py:method:: ArrayBuilder.type(arg0)
