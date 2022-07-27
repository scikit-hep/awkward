.. py:currentmodule:: ak.layout

ak.layout.EmptyArray
--------------------

An EmptyArray is used whenever an array's type is not known because it is empty
(such as data from :func:`ak.ArrayBuilder` without enough sample points
to resolve the type).

EmptyArray has no equivalent in Apache Arrow.

Below is a simplified implementation of an EmptyArray class in pure Python
that exhaustively checks validity in its constructor (see
:func:`ak.is_valid`) and can generate random valid arrays. The
``random_number()`` function returns a random float and the
``random_length(minlen)`` function returns a random int that is at least
``minlen``. The ``RawArray`` class represents simple, one-dimensional data.

.. code-block:: python

    class EmptyArray(Content):
        def __init__(self):
            pass

        @staticmethod
        def random(minlen, choices):
            assert minlen == 0
            return EmptyArray()

        def __len__(self):
            return 0

        def __getitem__(self, where):
            if isinstance(where, int):
                assert False
            elif isinstance(where, slice) and where.step is None:
                return EmptyArray()
            elif isinstance(where, str):
                raise ValueError("field " + repr(where) + " not found")
            else:
                raise AssertionError(where)

        def __repr__(self):
            return "EmptyArray()"

        def xml(self, indent="", pre="", post=""):
            return indent + pre + "<EmptyArray/>" + post

Here is an example:

.. code-block:: python

    EmptyArray()

.. code-block:: xml

    <EmptyArray/>

which represents the following logical data.

.. code-block:: python

    []

In addition to the properties and methods described in :doc:`ak.layout.Content`,
an EmptyArray has the following.

.. py:class:: EmptyArray(identities=None, parameters=None)

.. _ak.layout.EmptyArray.__init__:

.. py:method:: EmptyArray.__init__(identities=None, parameters=None)

.. _ak.layout.EmptyArray.toNumpyArray:

.. py:method:: EmptyArray.toNumpyArray()

Converts this EmptyArray into a :doc:`ak.layout.NumpyArray` with 64-bit
floating-point type.

.. _ak.layout.EmptyArray.simplify:

.. py:method:: EmptyArray.simplify()

Pass-through; returns the original array.
