ak.layout.SliceGenerator
------------------------

An SliceGenerator takes a :doc:`ak.layout.Content` array and a
slice (any Python object that can be used in an array's
``__getitem__``), and optionally
an expected :doc:`ak.forms.Form` and/or ``length``.

It can only be used in a :doc:`ak.layout.VirtualArray` to generate
arrays. When executed, it applies the slice to its
:doc:`ak.layout.Content`.

If a :doc:`ak.forms.Form` is given, there are fewer
circumstances in which the generator would need to be executed,
but it is an error if the generated array does not match this
:doc:`ak.forms.Form`.

If a ``length`` is given (as a non-negative
int), there are fewer circumstances in which the generator would
need to be executed, but it is an error if the generated array does
not match this ``length``.

ak.layout.SliceGenerator.__init__
=================================

.. py:method:: ak.layout.SliceGenerator.__init__(content, slice, form=None, length=None)

ak.layout.SliceGenerator.content
================================

.. py:attribute:: ak.layout.SliceGenerator.content

ak.layout.SliceGenerator.form
=============================

.. py:attribute:: ak.layout.SliceGenerator.form

ak.layout.SliceGenerator.length
===============================

.. py:attribute:: ak.layout.SliceGenerator.length

ak.layout.SliceGenerator.__call__
=================================

.. py:method:: ak.layout.SliceGenerator.__call__()

Generates the array and checks it against the :doc:`ak.forms.Form` and
``length``, if given.

ak.layout.SliceGenerator.__repr__
=================================

.. py:method:: ak.layout.SliceGenerator.__repr__()
