ak.layout.ListArray
-------------------

The ListArray concept is implemented in 3 specialized classes:

    * ``ak.layout.ListArray64``: ``starts`` and ``stops`` values are 64-bit
      signed integers.
    * ``ak.layout.ListArray32``: ``starts`` and ``stops`` values are 32-bit
      signed integers.
    * ``ak.layout.ListArrayU32``: ``starts`` and ``stops`` values are 32-bit
      unsigned integers.

HERE

In addition to the properties and methods described in :doc:`ak.layout.Content`,
a ListArray has the following.

ak.layout.ListArray.__init__
============================

.. py:method:: ak.layout.ListArray.__init__(starts, stops, content, identities=None, parameters=None)

ak.layout.ListArray.starts
==========================

.. py:attribute:: ak.layout.ListArray.starts

ak.layout.ListArray.stops
=========================

.. py:attribute:: ak.layout.ListArray.stops

ak.layout.ListArray.content
===========================

.. py:attribute:: ak.layout.ListArray.content

ak.layout.ListArray.compact_offsets64
=====================================

.. py:method:: ak.layout.ListArray.compact_offsets64(start_at_zero=True)

ak.layout.ListArray.broadcast_tooffsets64
=========================================

.. py:method:: ak.layout.ListArray.broadcast_tooffsets64(offsets)

ak.layout.ListArray.toRegularArray
==================================

.. py:method:: ak.layout.ListArray.toRegularArray()

ak.layout.ListArray.simplify
============================

.. py:method:: ak.layout.ListArray.simplify()
