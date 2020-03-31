ak.layout.IndexedOptionArray
----------------------------

The IndexedOptionArray concept is implemented in 2 specialized classes:

    * ``ak.layout.IndexOptionArray64``: ``index`` values 64-bit signed integers.
    * ``ak.layout.IndexOptionArray32``: ``index`` values are 32-bit signed
      integers.

HERE

In addition to the properties and methods described in :doc:`ak.layout.Content`,
an IndexedOptionArray has the following.

ak.layout.IndexedOptionArray.__init__
=====================================

.. py:method:: ak.layout.IndexedOptionArray.__init__(index, content, identities=None, parameters=None)

ak.layout.IndexedOptionArray.index
==================================

.. py:attribute:: ak.layout.IndexedOptionArray.index

ak.layout.IndexedOptionArray.content
====================================

.. py:attribute:: ak.layout.IndexedOptionArray.content

ak.layout.IndexedOptionArray.isoption
=====================================

.. py:attribute:: ak.layout.IndexedOptionArray.isoption

ak.layout.IndexedOptionArray.project
====================================

.. py:method:: ak.layout.IndexedOptionArray.project(mask=None)

ak.layout.IndexedOptionArray.bytemask
=====================================

.. py:method:: ak.layout.IndexedOptionArray.bytemask()

ak.layout.IndexedOptionArray.simplify
=====================================

.. py:method:: ak.layout.IndexedOptionArray.simplify()
