.. py:currentmodule:: ak.layout

ak.layout.ArrayGenerator
------------------------

An ArrayGenerator takes a Python callable, arguments, and optionally
an expected :class:`ak.forms.Form` and/or ``length``.

It can only be used in a :class:`ak.layout.VirtualArray` to generate
arrays.

If a :class:`ak.forms.Form` is given, there are fewer
circumstances in which the generator would need to be executed,
but it is an error if the generated array does not match this
:class:`ak.forms.Form`.

If a ``length`` is given (as a non-negative
int), there are fewer circumstances in which the generator would
need to be executed, but it is an error if the generated array does
not match this ``length``.

.. py:class:: ArrayGenerator(callable, args=()

.. py:method:: ArrayGenerator.__init__(callable, args=(), kwargs={}, form=None, length=None)

.. py:attribute:: ArrayGenerator.callable

.. py:attribute:: ArrayGenerator.args

.. py:attribute:: ArrayGenerator.kwargs

.. py:attribute:: ArrayGenerator.form

.. py:attribute:: ArrayGenerator.length

.. py:method:: ArrayGenerator.__call__()

Generates the array and checks it against the :class:`ak.forms.Form` and
``length``, if given.

.. py:method:: ArrayGenerator.__repr__()
