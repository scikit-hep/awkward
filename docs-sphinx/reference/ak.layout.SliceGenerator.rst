.. py:currentmodule:: ak.layout

ak.layout.SliceGenerator
------------------------

An SliceGenerator takes a :class:`ak.layout.Content` array and a
slice (any Python object that can be used in an array's
``__getitem__``), and optionally
an expected :class:`ak.forms.Form` and/or ``length``.

It can only be used in a :class:`ak.layout.VirtualArray` to generate
arrays. When executed, it applies the slice to its
:class:`ak.layout.Content`.

If a :class:`ak.forms.Form` is given, there are fewer
circumstances in which the generator would need to be executed,
but it is an error if the generated array does not match this
:class:`ak.forms.Form`.

If a ``length`` is given (as a non-negative
int), there are fewer circumstances in which the generator would
need to be executed, but it is an error if the generated array does
not match this ``length``.

.. py:class:: SliceGenerator(content, slice, form=None, length=None)

.. py:method:: SliceGenerator.__init__(content, slice, form=None, length=None)

.. py:attribute:: SliceGenerator.content

.. py:attribute:: SliceGenerator.form

.. py:attribute:: SliceGenerator.length

.. py:method:: SliceGenerator.__call__()

Generates the array and checks it against the :class:`ak.forms.Form` and
``length``, if given.

.. py:method:: SliceGenerator.__repr__()
