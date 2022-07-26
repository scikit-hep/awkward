.. py:currentmodule:: ak.layout

ak.layout.ArrayGenerator
------------------------

An ArrayGenerator takes a Python callable, arguments, and optionally
an expected :doc:`ak.forms.Form` and/or ``length``.

It can only be used in a :doc:`ak.layout.VirtualArray` to generate
arrays.

If a :doc:`ak.forms.Form` is given, there are fewer
circumstances in which the generator would need to be executed,
but it is an error if the generated array does not match this
:doc:`ak.forms.Form`.

If a ``length`` is given (as a non-negative
int), there are fewer circumstances in which the generator would
need to be executed, but it is an error if the generated array does
not match this ``length``.

.. py:class:: ArrayGenerator(callable, args=()

ak.layout.ArrayGenerator.__init__
=================================

.. py:method:: ArrayGenerator.__init__(callable, args=(), kwargs={}, form=None, length=None)

ak.layout.ArrayGenerator.callable
=================================

.. py:attribute:: ArrayGenerator.callable

ak.layout.ArrayGenerator.args
=============================

.. py:attribute:: ArrayGenerator.args

ak.layout.ArrayGenerator.kwargs
===============================

.. py:attribute:: ArrayGenerator.kwargs

ak.layout.ArrayGenerator.form
=============================

.. py:attribute:: ArrayGenerator.form

ak.layout.ArrayGenerator.length
===============================

.. py:attribute:: ArrayGenerator.length

ak.layout.ArrayGenerator.__call__
=================================

.. py:method:: ArrayGenerator.__call__()

Generates the array and checks it against the :doc:`ak.forms.Form` and
``length``, if given.

ak.layout.ArrayGenerator.__repr__
=================================

.. py:method:: ArrayGenerator.__repr__()
