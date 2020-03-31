ak.layout.NumpyArray
--------------------

A NumpyArray describes multidimensional data with the same set of parameters
as a NumPy ``np.ndarray``.

   * ``ptr``: The data themselves, a raw buffer.
   * ``shape``: Non-negative integers, at least one. Each represents the
     number of components in a dimension; a shape of length *N* represents an
     *N* dimensional tensor. The number of items in the array is the product
     of all values in shape (may be zero).
   * ``strides``: The same number of integers as ``shape``. The strides
     describes how many items in ``ptr`` to skip per element in a dimension.
     (Strides can be negative or zero.)
   * ``offset``: the number of items in ptr to skip before the first element
     of the array.
   * ``itemsize``: the number of bytes per item (i.e. 1 for characters, 4 for
     int32, 8 for double types).

If the ``shape`` is one-dimensional, a NumpyArray corresponds to an Apache
Arrow `Primitive array <https://arrow.apache.org/docs/format/Columnar.html#fixed-size-primitive-layout>`__.

Below is a simplified implementation of a NumpyArray class in pure Python
that exhaustively checks validity in its constructor (see
:doc:`_auto/ak.isvalid`) and can generate random valid arrays. The
``random_number()`` function returns a random float and the
``random_length(minlen)`` function returns a random int that is at least
``minlen``. The ``RawArray`` class represents simple, one-dimensional data.

.. code-block:: python

    class NumpyArray(Content):
        def __init__(self, ptr, shape, strides, offset):
            assert isinstance(ptr, list)
            assert isinstance(shape, list)
            assert isinstance(strides, list)
            for x in ptr:
                assert isinstance(x, (bool, int, float))
            assert len(shape) > 0
            assert len(strides) == len(shape)
            for x in shape:
                assert isinstance(x, int)
                assert x >= 0
            for x in strides:
                assert isinstance(x, int)
            assert isinstance(offset, int)
            if all(x != 0 for x in shape):
                assert 0 <= offset < len(ptr)
                last = offset
                for sh, st in zip(shape, strides):
                    last += (sh - 1) * st
                assert last <= len(ptr)
            self.ptr = ptr
            self.shape = shape
            self.strides = strides
            self.offset = offset

        @staticmethod
        def random(minlen, choices):
            shape = [random_length(minlen)]
            for i in range(random_length(0, 2)):
                shape.append(random_length(1, 3))
            strides = [1]
            for x in shape[:0:-1]:
                skip = random_length(0, 2)
                strides.insert(0, x * strides[0] + skip)
            offset = random_length()
            ptr = [random_number() for i in range(shape[0] * strides[0] + offset)]
            return NumpyArray(ptr, shape, strides, offset)

        def __len__(self):
            return self.shape[0]

        def __getitem__(self, where):
            if isinstance(where, int):
                assert 0 <= where < len(self)
                offset = self.offset + self.strides[0] * where
                if len(self.shape) == 1:
                    return self.ptr[offset]
                else:
                    return NumpyArray(self.ptr, self.shape[1:], self.strides[1:], offset)
            elif isinstance(where, slice) and where.step is None:
                offset = self.offset + self.strides[0] * where.start
                shape = [where.stop - where.start] + self.shape[1:]
                return NumpyArray(self.ptr, shape, self.strides, offset)
            elif isinstance(where, str):
                raise ValueError("field " + repr(where) + " not found")
            else:
                raise AssertionError(where)

        def __repr__(self):
            return ("NumpyArray(" + repr(self.ptr) + ", " + repr(self.shape) + ", " +
                    repr(self.strides) + ", " + repr(self.offset) + ")")

        def xml(self, indent="", pre="", post=""):
            out = indent + pre + "<NumpyArray>\n"
            out += indent + "    <ptr>" + " ".join(str(x) for x in self.ptr) + "</ptr>\n"
            out += indent + "    <shape>" + " ".join(str(x) for x in self.shape) + "</shape>\n"
            out += indent + "    <strides>" + " ".join(str(x) for x in self.strides)
            out += "</strides>\n"
            out += indent + "    <offset>" + str(self.offset) + "</offset>\n"
            out += indent + "</NumpyArray>" + post
            return out

Here is an example:

.. code-block:: python

    NumpyArray([2.4, 9.6, -0.2, 7.1, 10.2, 3.3, 7.9, 4.5, 2.1, 5.4, 8.4, 2.3, 12.0, 5.6, 6.2,
                11.4, 4.4, 3.0, 4.7, 7.8, 2.4, 2.2, 0.8, 10.6, 8.2, 5.4, 6.7, 4.5, 5.1, 11.2,
                11.4, 9.2, 6.6, 2.1, -2.4, 6.8, 8.8, 8.2, 5.4, 2.9, 8.2, 7.0, 2.2, 4.8, 5.3,
                6.4, 4.1, 5.1, 8.6, 9.4, 5.1, 6.0],
               [17, 2],
               [2, 1],
               18)

.. code-block:: xml

    <NumpyArray>
        <ptr>2.4 9.6 -0.2 7.1 10.2 3.3 7.9 4.5 2.1 5.4 8.4 2.3 12.0 5.6 6.2 11.4 4.4 3.0 4.7 7.8
             2.4 2.2 0.8 10.6 8.2 5.4 6.7 4.5 5.1 11.2 11.4 9.2 6.6 2.1 -2.4 6.8 8.8 8.2 5.4 2.9
             8.2 7.0 2.2 4.8 5.3 6.4 4.1 5.1 8.6 9.4 5.1 6.0</ptr>
        <shape>17 2</shape>
        <strides>2 1</strides>
        <offset>18</offset>
    </NumpyArray>

which represents the following logical data.

.. code-block:: python

    [[4.7, 7.8],
     [2.4, 2.2],
     [0.8, 10.6],
     [8.2, 5.4],
     [6.7, 4.5],
     [5.1, 11.2],
     [11.4, 9.2],
     [6.6, 2.1],
     [-2.4, 6.8],
     [8.8, 8.2],
     [5.4, 2.9],
     [8.2, 7.0],
     [2.2, 4.8],
     [5.3, 6.4],
     [4.1, 5.1],
     [8.6, 9.4],
     [5.1, 6.0]]

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
