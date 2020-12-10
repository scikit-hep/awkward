ak.layout.RegularArray
----------------------

The RegularArray class describes lists that all have the same length, the single
integer ``size``. Its underlying ``content`` is a flattened view of the data;
that is, each list is not stored separately in memory, but is inferred as a
subinterval of the underlying data.

If the ``content`` length is not an integer multiple of ``size``, then the length
of the RegularArray is truncated to the largest integer multiple.

An extra field ``length`` is ignored unless the ``size`` is zero. This sets the
length of the RegularArray in only those cases, so that it is possible for an
array to contain a non-zero number of zero-length lists with regular type.

A multidimensional :doc:`ak.layout.NumpyArray` is equivalent to a one-dimensional
:doc:`ak.layout.NumpyArray` nested within several RegularArrays, one for each
dimension. However, RegularArrays can be used to make lists of any other type.

RegularArray corresponds to an Apache Arrow `Tensor <https://arrow.apache.org/docs/python/generated/pyarrow.Tensor.html>`__.

Below is a simplified implementation of a RegularArray class in pure Python
that exhaustively checks validity in its constructor (see
:doc:`_auto/ak.is_valid`) and can generate random valid arrays. The
``random_number()`` function returns a random float and the
``random_length(minlen)`` function returns a random int that is at least
``minlen``. The ``RawArray`` class represents simple, one-dimensional data.

.. code-block:: python

    class RegularArray(Content):
        def __init__(self, content, size, length=0):
            assert isinstance(content, Content)
            assert isinstance(size, int)
            assert isinstance(length, int)
            assert size >= 0
            if size != 0:
                length = len(self.content) // self.size   # floor division
            self.content = content
            self.size = size
            self.length = length

        @staticmethod
        def random(minlen, choices):
            size = random_length(0, 5)
            length = random_length(0, 5)
            content = random.choice(choices).random(random_length(minlen) * size, choices)
            return RegularArray(content, size, length)

        def __len__(self):
            return self.length

        def __getitem__(self, where):
            if isinstance(where, int):
                assert 0 <= where < len(self)
                return self.content[(where) * self.size:(where + 1) * self.size]
            elif isinstance(where, slice) and where.step is None:
                start = where.start * self.size
                stop = where.stop * self.size
                length = where.stop - where.start
                return RegularArray(self.content[start:stop], self.size, length)
            elif isinstance(where, str):
                return RegularArray(self.content[where], self.size, self.length)
            else:
                raise AssertionError(where)

        def __repr__(self):
            if size == 0:
                length = ", " + repr(self.length)
            else:
                length = ""
            return "RegularArray(" + repr(self.content) + ", " + repr(self.size) + length + ")"

        def xml(self, indent="", pre="", post=""):
            out = indent + pre + "<RegularArray>\n"
            out += self.content.xml(indent + "    ", "<content>", "</content>\n")
            out += indent + "    <size>" + str(self.size) + "</size>\n"
            if size == 0:
                out += indent + "    <length>" + str(self.length) + "</length>\n"
            out += indent + "</RegularArray>" + post
            return out

Here is an example:

.. code-block:: python

    RegularArray(RawArray([7.4, -0.0, 6.6, 6.6, 5.2, 4.6, 9.6, 4.2, 2.3, 6.5, 4.2, 1.3, 2.2, 4.1,
                           1.9, 3.9, 2.3, 2.3, 0.7, 6.9, 1.4, 9.6, 11.8, 6.8, 8.2, 10.5, 8.2,
                           7.5, 6.3, 5.4, 0.5, 1.0, 5.5, 4.1, 5.9, 7.9, 6.7, 7.3, 5.6, 5.5, 2.2,
                           2.2, -0.3, 3.5, 11.2, 13.4, 6.7, -1.0, 6.4, 1.3, 6.8, 5.1, 3.2, 9.5,
                           2.8]),
                 5)

.. code-block:: xml

    <RegularArray>
        <content><RawArray>
            <ptr>7.4 -0.0 6.6 6.6 5.2 4.6 9.6 4.2 2.3 6.5 4.2 1.3 2.2 4.1 1.9 3.9 2.3 2.3 0.7 6.9
                 1.4 9.6 11.8 6.8 8.2 10.5 8.2 7.5 6.3 5.4 0.5 1.0 5.5 4.1 5.9 7.9 6.7 7.3 5.6
                 5.5 2.2 2.2 -0.3 3.5 11.2 13.4 6.7 -1.0 6.4 1.3 6.8 5.1 3.2 9.5 2.8</ptr>
        </RawArray></content>
        <size>5</size>
    </RegularArray>

which represents the following logical data.

.. code-block:: python

    [[7.4, -0.0, 6.6, 6.6, 5.2],
     [4.6, 9.6, 4.2, 2.3, 6.5],
     [4.2, 1.3, 2.2, 4.1, 1.9],
     [3.9, 2.3, 2.3, 0.7, 6.9],
     [1.4, 9.6, 11.8, 6.8, 8.2],
     [10.5, 8.2, 7.5, 6.3, 5.4],
     [0.5, 1.0, 5.5, 4.1, 5.9],
     [7.9, 6.7, 7.3, 5.6, 5.5],
     [2.2, 2.2, -0.3, 3.5, 11.2],
     [13.4, 6.7, -1.0, 6.4, 1.3],
     [6.8, 5.1, 3.2, 9.5, 2.8]]

In addition to the properties and methods described in :doc:`ak.layout.Content`,
a RegularArray has the following.

ak.layout.RegularArray.__init__
===============================

.. py:method:: ak.layout.RegularArray.__init__(content, size, identities=None, parameters=None)

ak.layout.RegularArray.content
==============================

.. py:attribute:: ak.layout.RegularArray.content

ak.layout.RegularArray.size
===========================

.. py:attribute:: ak.layout.RegularArray.size

ak.layout.RegularArray.compact_offsets64
========================================

.. py:method:: ak.layout.RegularArray.compact_offsets64(start_at_zero=True)

Returns a 64-bit :doc:`ak.layout.Index` of ``offsets`` by prefix summing
in steps of ``size``.

ak.layout.RegularArray.broadcast_tooffsets64
============================================

.. py:method:: ak.layout.RegularArray.broadcast_tooffsets64(offsets)

Shifts ``contents`` to match a given set of ``offsets`` (if possible) and
returns a :doc:`ak.layout.ListOffsetArray` with the results. This is used in
broadcasting because a set of :doc:`ak.types.ListType` and :doc:`ak.types.RegularType`
arrays have to be reordered to a common ``offsets`` before they can be directly
operated upon.

ak.layout.RegularArray.simplify
===============================

.. py:method:: ak.layout.RegularArray.simplify()

Pass-through; returns the original array.
