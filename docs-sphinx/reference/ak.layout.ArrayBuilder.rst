.. py:currentmodule:: ak.layout

ak.layout.ArrayBuilder
----------------------

The low-level ArrayBuilder that builds :doc:`ak.layout.Content` arrays. This
object is wrapped by :func:`ak.ArrayBuilder`.

(Method names in the high-level interface have been chnaged to include
underscores after "begin" and "end," but that hasn't happened in the
low-level interface, yet or possibly at all.)

.. py:class:: ArrayBuilder(initial=1024, resize=1.5)

ak.layout.ArrayBuilder.__getitem__
==================================

.. py:method:: ArrayBuilder.__getitem__(where)

ak.layout.ArrayBuilder.__init__
===============================

.. py:method:: ArrayBuilder.__init__(initial=1024, resize=1.5)

ak.layout.ArrayBuilder.__iter__
===============================

.. py:method:: ArrayBuilder.__iter__()

ak.layout.ArrayBuilder.__len__
==============================

.. py:method:: ArrayBuilder.__len__()

ak.layout.ArrayBuilder.__repr__
===============================

.. py:method:: ArrayBuilder.__repr__()

ak.layout.ArrayBuilder.beginlist
================================

.. py:method:: ArrayBuilder.beginlist()

ak.layout.ArrayBuilder.beginrecord
==================================

.. py:method:: ArrayBuilder.beginrecord(name=None)

ak.layout.ArrayBuilder.begintuple
=================================

.. py:method:: ArrayBuilder.begintuple(arg0)

ak.layout.ArrayBuilder.boolean
==============================

.. py:method:: ArrayBuilder.boolean(arg0)

ak.layout.ArrayBuilder.bytestring
=================================

.. py:method:: ArrayBuilder.bytestring(arg0)

ak.layout.ArrayBuilder.clear
============================

.. py:method:: ArrayBuilder.clear()

ak.layout.ArrayBuilder.endlist
==============================

.. py:method:: ArrayBuilder.endlist()

ak.layout.ArrayBuilder.endrecord
================================

.. py:method:: ArrayBuilder.endrecord()

ak.layout.ArrayBuilder.endtuple
===============================

.. py:method:: ArrayBuilder.endtuple()


ak.layout.ArrayBuilder.field
============================

.. py:method:: ArrayBuilder.field(arg0)

ak.layout.ArrayBuilder.fromiter
===============================

.. py:method:: ArrayBuilder.fromiter(arg0)

ak.layout.ArrayBuilder.index
============================

.. py:method:: ArrayBuilder.index(arg0)

ak.layout.ArrayBuilder.integer
==============================

.. py:method:: ArrayBuilder.integer(arg0)

ak.layout.ArrayBuilder.null
===========================

.. py:method:: ArrayBuilder.null()

ak.layout.ArrayBuilder.real
===========================

.. py:method:: ArrayBuilder.real(arg0)

ak.layout.ArrayBuilder.snapshot
===============================

.. py:method:: ArrayBuilder.snapshot()

ak.layout.ArrayBuilder.string
=============================

.. py:method:: ArrayBuilder.string(arg0)

ak.layout.ArrayBuilder.type
===========================

.. py:method:: ArrayBuilder.type(arg0)
