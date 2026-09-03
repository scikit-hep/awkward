.. py:currentmodule:: ak.layout

ak._ext.ArrayBuilder
----------------------

The low-level ArrayBuilder that builds :class:`ak.contents.Content` arrays. This
object is wrapped by :class:`ak.ArrayBuilder`.

(Method names in the high-level interface have been changed to include
underscores after "begin" and "end," but that hasn't happened in the
low-level interface, yet or possibly at all.)

.. py:class:: ArrayBuilder(initial=1024, resize=8)

.. py:method:: ArrayBuilder.__getitem__(where)

.. py:method:: ArrayBuilder.__init__(initial=1024, resize=8)

.. py:method:: ArrayBuilder.__iter__()

.. py:method:: ArrayBuilder.__len__()

.. py:method:: ArrayBuilder.__repr__()

.. py:method:: ArrayBuilder.beginlist()

.. py:method:: ArrayBuilder.beginrecord(name=None)

.. py:method:: ArrayBuilder.begintuple(arg0)

.. py:method:: ArrayBuilder.boolean(arg0)

.. py:method:: ArrayBuilder.bytestring(arg0)

.. py:method:: ArrayBuilder.clear()

.. py:method:: ArrayBuilder.endlist()

.. py:method:: ArrayBuilder.endrecord()

.. py:method:: ArrayBuilder.endtuple()

.. py:method:: ArrayBuilder.field(arg0)

.. py:method:: ArrayBuilder.fromiter(arg0)

.. py:method:: ArrayBuilder.index(arg0)

.. py:method:: ArrayBuilder.integer(arg0)

.. py:method:: ArrayBuilder.null()

.. py:method:: ArrayBuilder.real(arg0)

.. py:method:: ArrayBuilder.snapshot()

.. py:method:: ArrayBuilder.string(arg0)

.. py:method:: ArrayBuilder.type(arg0)
