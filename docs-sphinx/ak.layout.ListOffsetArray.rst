ak.layout.ListOffsetArray
-------------------------

The ListOffsetArray concept is implemented in 3 specialized classes:

    * ``ak.layout.ListOffsetArray64``: ``offsets`` values are 64-bit signed
      integers.
    * ``ak.layout.ListOffsetArray32``: ``offsets`` values are 32-bit signed
      integers.
    * ``ak.layout.ListOffsetArrayU32``: ``offsets`` values are 32-bit
      unsigned integers.

HERE

In addition to the properties and methods described in :doc:`ak.layout.Content`,
a ListOffsetArray has the following.

ak.layout.ListOffsetArray.__init__
==================================

.. py:method:: ak.layout.ListOffsetArray.__init__(offsets, content, identities=None, parameters=None)

ak.layout.ListOffsetArray.offsets
=================================

.. py:attribute:: ak.layout.ListOffsetArray.offsets

ak.layout.ListOffsetArray.content
=================================

.. py:attribute:: ak.layout.ListOffsetArray.content

ak.layout.ListOffsetArray.starts
================================

.. py:attribute:: ak.layout.ListOffsetArray.starts

ak.layout.ListOffsetArray.stops
===============================

.. py:attribute:: ak.layout.ListOffsetArray.stops

ak.layout.ListOffsetArray.compact_offsets64
===========================================

.. py:method:: ak.layout.ListOffsetArray.compact_offsets64(start_at_zero=True)

ak.layout.ListOffsetArray.broadcast_tooffsets64
===============================================

.. py:method:: ak.layout.ListOffsetArray.broadcast_tooffsets64(offsets)

ak.layout.ListOffsetArray.toRegularArray
========================================

.. py:method:: ak.layout.ListOffsetArray.toRegularArray()

ak.layout.ListOffsetArray.simplify
==================================

.. py:method:: ak.layout.ListOffsetArray.simplify()
