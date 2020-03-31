ak.layout.NumpyArray
--------------------

HERE

NumpyArray supports the buffer protocol, so it can be directly cast as a
NumPy array.

In addition to the properties and methods described in :doc:`ak.layout.Content`,
a NumpyArray has the following.

ak.layout.NumpyArray.__init__
=============================

.. py:method:: ak.layout.NumpyArray.__init__(array, identities=None, parameters=None)

ak.layout.NumpyArray.shape
==========================

.. py:attribute:: ak.layout.NumpyArray.shape

ak.layout.NumpyArray.strides
============================

.. py:attribute:: ak.layout.NumpyArray.strides

ak.layout.NumpyArray.itemsize
=============================

.. py:attribute:: ak.layout.NumpyArray.itemsize

ak.layout.NumpyArray.format
===========================

.. py:attribute:: ak.layout.NumpyArray.format

ak.layout.NumpyArray.ndim
=========================

.. py:attribute:: ak.layout.NumpyArray.ndim

ak.layout.NumpyArray.isscalar
=============================

.. py:attribute:: ak.layout.NumpyArray.isscalar

ak.layout.NumpyArray.isempty
============================

.. py:attribute:: ak.layout.NumpyArray.isempty

ak.layout.NumpyArray.iscontiguous
=================================

.. py:attribute:: ak.layout.NumpyArray.iscontiguous

ak.layout.NumpyArray.toRegularArray
===================================

.. py:method:: ak.layout.NumpyArray.toRegularArray()

ak.layout.NumpyArray.contiguous
===============================

.. py:method:: ak.layout.NumpyArray.contiguous()

ak.layout.NumpyArray.simplify
=============================

.. py:method:: ak.layout.NumpyArray.simplify()
