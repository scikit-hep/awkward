ak.layout.IndexedArray
----------------------

The IndexedArray concept is implemented in 3 specialized classes:

    * ``ak.layout.IndexArray64``: ``index`` values are 64-bit signed integers.
    * ``ak.layout.IndexArray32``: ``index`` values are 32-bit signed integers.
    * ``ak.layout.IndexArrayU32``: ``index`` values are 32-bit unsigned
      integers.

HERE

In addition to the properties and methods described in :doc:`ak.layout.Content`,
an IndexedArray has the following.

ak.layout.IndexedArray.__init__
===============================

.. py:method:: ak.layout.IndexedArray.__init__(index, content, identities=None, parameters=None)

ak.layout.IndexedArray.index
============================

.. py:attribute:: ak.layout.IndexedArray.index

ak.layout.IndexedArray.content
==============================

.. py:attribute:: ak.layout.IndexedArray.content

ak.layout.IndexedArray.isoption
===============================

.. py:attribute:: ak.layout.IndexedArray.isoption

ak.layout.IndexedArray.project
==============================

.. py:method:: ak.layout.IndexedArray.project(mask=None)

ak.layout.IndexedArray.bytemask
===============================

.. py:method:: ak.layout.IndexedArray.bytemask()

ak.layout.IndexedArray.simplify
===============================

.. py:method:: ak.layout.IndexedArray.simplify()
