ak.layout.IndexedArray
----------------------

The IndexedArray concept is implemented in 3 specialized classes:

    * ``ak.layout.IndexArray64``: ``index`` values are 64-bit signed integers.
    * ``ak.layout.IndexArray32``: ``index`` values are 32-bit signed integers.
    * ``ak.layout.IndexArrayU32``: ``index`` values are 32-bit unsigned
      integers.

The IndexedArray class is a general-purpose tool for changing the order of
and/or duplicating some ``content``. Its ``index`` array is a lazily applied
`np.take <https://docs.scipy.org/doc/numpy/reference/generated/numpy.take.html>`__
(integer-array slice, also known as "advanced indexing").

It has many uses:

   * representing a lazily applied slice.
   * simulating pointers into another collection.
   * emulating the dictionary encoding of Apache Arrow and Parquet.

IndexedArray doesn't have a direct equivalent in Apache Arrow.

Below is a simplified implementation of a IndexedArray class in pure Python
that exhaustively checks validity in its constructor (see
:doc:`_auto/ak.isvalid`) and can generate random valid arrays. The
``random_number()`` function returns a random float and the
``random_length(minlen)`` function returns a random int that is at least
``minlen``. The ``RawArray`` class represents simple, one-dimensional data.

.. code-block:: python

    class IndexedArray(Content):
        def __init__(self, index, content):
            assert isinstance(index, list)
            assert isinstance(content, Content)
            for x in index:
                assert isinstance(x, int)
                assert 0 <= x < len(content)   # index[i] must not be negative
            self.index = index
            self.content = content

        @staticmethod
        def random(minlen, choices):
            if minlen == 0:
                content = random.choice(choices).random(0, choices)
            else:
                content = random.choice(choices).random(1, choices)
            if len(content) == 0:
                index = []
            else:
                index = [random.randint(0, len(content) - 1)
                             for i in range(random_length(minlen))]
            return IndexedArray(index, content)

        def __len__(self):
            return len(self.index)

        def __getitem__(self, where):
            if isinstance(where, int):
                assert 0 <= where < len(self)
                return self.content[self.index[where]]
            elif isinstance(where, slice) and where.step is None:
                return IndexedArray(self.index[where.start:where.stop], self.content)
            elif isinstance(where, str):
                return IndexedArray(self.index, self.content[where])
            else:
                raise AssertionError(where)

        def __repr__(self):
            return "IndexedArray(" + repr(self.index) + ", " + repr(self.content) + ")"

        def xml(self, indent="", pre="", post=""):
            out = indent + pre + "<IndexedArray>\n"
            out += indent + "    <index>" + " ".join(str(x) for x in self.index) + "</index>\n"
            out += self.content.xml(indent + "    ", "<content>", "</content>\n")
            out += indent + "</IndexedArray>\n"
            return out

Here is an example:

.. code-block:: python

    IndexedArray([3, 5, 1, 1, 5, 3],
                 RawArray([8.9, 3.2, 5.4, 9.8, 7.5, 1.9]))

.. code-block:: xml

    <IndexedArray>
        <index>3 5 1 1 5 3</index>
        <content><RawArray>
            <ptr>8.9 3.2 5.4 9.8 7.5 1.9</ptr>
        </RawArray></content>
    </IndexedArray>

which represents the following logical data.

.. code-block:: python

    [9.8, 1.9, 3.2, 3.2, 1.9, 9.8]

In addition to the properties and methods described in :doc:`ak.layout.Content`,
an IndexedArray has the following.

ak.layout.IndexedArray.__init__
===============================

.. py:method:: ak.layout.IndexedArray.__init__(index, content, identities=None, parameters=None)

ak.layout.IndexedArray.index
============================

.. py:attribute:: ak.layout.IndexedArray.index

ak.layout.IndexedArray.content
==============================

.. py:attribute:: ak.layout.IndexedArray.content

ak.layout.IndexedArray.isoption
===============================

.. py:attribute:: ak.layout.IndexedArray.isoption

Returns False because this is not an IndexedOptionArray.

ak.layout.IndexedArray.project
==============================

.. py:method:: ak.layout.IndexedArray.project(mask=None)

Returns an array with the ``index`` applied to reorder/duplicate elements.

If ``mask`` is a signed 8-bit :doc:`ak.layout.Index` in which ``0`` means valid
and ``1`` means missing, only valid elements according to this ``mask`` are
returned.

ak.layout.IndexedArray.bytemask
===============================

.. py:method:: ak.layout.IndexedArray.bytemask()

Returns an 8-bit signed :doc:`ak.layout.Index` of all zeros, because this
IndexedArray does not have :doc:`ak.types.OptionType`.

ak.layout.IndexedArray.simplify
===============================

.. py:method:: ak.layout.IndexedArray.simplify()

Combines this node with its ``content`` if the ``content`` also has
:doc:`ak.types.OptionType` or is an :doc:`ak.layout.IndexedArray`; otherwise, this is
a pass-through.  In all cases, the output has the same logical meaning as the input.

This method only operates one level deep.
