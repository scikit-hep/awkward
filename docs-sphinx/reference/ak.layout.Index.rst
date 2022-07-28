.. py:currentmodule:: ak.layout

ak.layout.Index
---------------

The Index concept is implemented in 5 specialized classes:

    * ``ak.layout.Index64``: values are 64-bit signed integers.
    * ``ak.layout.Index32``: values are 32-bit signed integers.
    * ``ak.layout.IndexU32``: values are 32-bit unsigned integers.
    * ``ak.layout.Index8``: values are 8-bit signed integers.
    * ``ak.layout.IndexU8``: values are 8-bit unsigned integers.

Index supports the buffer protocol, so it can be directly cast as a NumPy array.

An Index represents a contiguous, one-dimensional array of integers that is
fully interpreted in the C++ layer. The structure of all
:class:`ak.layout.Content` nodes is implemented in Indexes, such as a
:class:`ak.layout.ListArray`'s ``starts`` and ``stops``. Only data at the leaves
of a structure, such as :class:`ak.layout.NumpyArray` content, are not implemented
in an Index. (This allows :class:`ak.layout.NumpyArray` to handle non-contiguous,
non-integer data.)

This distinction is made because the C++ layer needs to make stronger
assumptions about the arrays used to define structure than any content at the
leaves of the structure.

.. py:class:: Index(numpy_array)

.. py:method:: Index.__init__(numpy_array)

Creates an Index from a NumPy array. If the array has the wrong type for
this specialization, or if it is not C-contiguous, the array is copied.
Otherwise, it is a view (shares data).

.. py:method:: Index.__getitem__(at)

Extracts one element from the Index.

.. py:method:: Index.__getitem__(start, stop)

Slices the Index between ``start`` and ``stop``.

.. py:method:: Index.__len__()

The length of the Index.

.. py:method:: Index.__repr__()

A string representation of the Index (single-line XML, trucated middle).
