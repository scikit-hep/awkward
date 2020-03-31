ak.layout.UnionArray
--------------------

The UnionArray concept is implemented in 3 specialized classes:

    * ``ak.layout.UnionArray8_64``: ``tags`` values are 8-bit signed integers
      and ``index`` values are 64-bit signed integers.
    * ``ak.layout.UnionArray8_32``: ``tags`` values are 8-bit signed integers
      and ``index`` values are 32-bit signed integers.
    * ``ak.layout.UnionArray8_U32``: ``tags`` values are 8-bit signed integers
      and ``index`` values are 32-bit unsigned integers.

HERE

In addition to the properties and methods described in :doc:`ak.layout.Content`,
a UnionArray has the following.

ak.layout.UnionArray.__init__
=============================

.. py:method:: ak.layout.UnionArray.__init__(tags, index, contents, identities=None, parameters=None)

ak.layout.UnionArray.regular_index
==================================

.. py:method:: ak.layout.UnionArray.regular_index(tags)

ak.layout.UnionArray.tags
=========================

.. py:attribute:: ak.layout.UnionArray.tags

ak.layout.UnionArray.index
==========================

.. py:attribute:: ak.layout.UnionArray.index

ak.layout.UnionArray.contents
=============================

.. py:attribute:: ak.layout.UnionArray.contents

ak.layout.UnionArray.numcontents
================================

.. py:attribute:: ak.layout.UnionArray.numcontents

ak.layout.UnionArray.content
============================

.. py:method:: ak.layout.UnionArray.content(index)

ak.layout.UnionArray.project
============================

.. py:method:: ak.layout.UnionArray.project()

ak.layout.UnionArray.simplify
=============================

.. py:method:: ak.layout.UnionArray.simplify(mergebool=False)
