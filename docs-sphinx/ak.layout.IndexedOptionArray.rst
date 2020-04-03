ak.layout.IndexedOptionArray
----------------------------

The IndexedOptionArray concept is implemented in 2 specialized classes:

    * ``ak.layout.IndexOptionArray64``: ``index`` values 64-bit signed integers.
    * ``ak.layout.IndexOptionArray32``: ``index`` values are 32-bit signed
      integers.

The IndexedOptionArray class is an :doc:`ak.layout.IndexedArray` for which
negative values in the index are interpreted as missing. It represents
:doc:`ak.types.OptionType` data like :doc:`ak.layout.ByteMaskedArray`,
:doc:`ak.layout.BitMaskedArray`, and :doc:`ak.layout.UnmaskedArray`, but
the flexibility of the arbitrary ``index`` makes it a common output of
many operations.

IndexedOptionArray doesn't have a direct equivalent in Apache Arrow.

Below is a simplified implementation of a IndexedOptionArray class in pure Python
that exhaustively checks validity in its constructor (see
:doc:`_auto/ak.is_valid`) and can generate random valid arrays. The
``random_number()`` function returns a random float and the
``random_length(minlen)`` function returns a random int that is at least
``minlen``. The ``RawArray`` class represents simple, one-dimensional data.

.. code-block:: python

    class IndexedOptionArray(Content):
        def __init__(self, index, content):
            assert isinstance(index, list)
            assert isinstance(content, Content)
            for x in index:
                assert isinstance(x, int)
                assert x < len(content)   # index[i] may be negative
            self.index = index
            self.content = content

        @staticmethod
        def random(minlen, choices):
            content = random.choice(choices).random(0, choices)
            index = []
            for i in range(random_length(minlen)):
                if len(content) == 0 or random.randint(0, 4) == 0:
                    index.append(-random_length(1))   # a random number, but not necessarily -1
                else:
                    index.append(random.randint(0, len(content) - 1))
            return IndexedOptionArray(index, content)

        def __len__(self):
            return len(self.index)

        def __getitem__(self, where):
            if isinstance(where, int):
                assert 0 <= where < len(self)
                if self.index[where] < 0:
                    return None
                else:
                    return self.content[self.index[where]]
            elif isinstance(where, slice) and where.step is None:
                return IndexedOptionArray(self.index[where.start:where.stop], self.content)
            elif isinstance(where, str):
                return IndexedOptionArray(self.index, self.content[where])
            else:
                raise AssertionError(where)

        def __repr__(self):
            return "IndexedOptionArray(" + repr(self.index) + ", " + repr(self.content) + ")"

        def xml(self, indent="", pre="", post=""):
            out = indent + pre + "<IndexedOptionArray>\n"
            out += indent + "    <index>" + " ".join(str(x) for x in self.index) + "</index>\n"
            out += self.content.xml(indent + "    ", "<content>", "</content>\n")
            out += indent + "</IndexedOptionArray>\n"
            return out

Here is an example:

.. code-block:: python

    IndexedOptionArray([-30, 19, 6, 7, -3, 21, 13, 22, 17, 9, -12, 16],
                       RawArray([5.2, 1.7, 6.7, -0.4, 4.0, 7.8, 3.8, 6.8, 4.2, 0.3, 4.6, 6.2,
                                 6.9, -0.7, 3.9, 1.6, 8.7, -0.7, 3.2, 4.3, 4.0, 5.8, 4.2, 7.0,
                                 5.6, 3.8]))

.. code-block:: xml

    <IndexedOptionArray>
        <index>-30 19 6 7 -3 21 13 22 17 9 -12 16</index>
        <content><RawArray>
            <ptr>5.2 1.7 6.7 -0.4 4.0 7.8 3.8 6.8 4.2 0.3 4.6 6.2 6.9 -0.7 3.9 1.6 8.7 -0.7 3.2
                 4.3 4.0 5.8 4.2 7.0 5.6 3.8</ptr>
        </RawArray></content>
    </IndexedOptionArray>

which represents the following logical data.

.. code-block:: python

    [None, 4.3, 3.8, 6.8, None, 5.8, -0.7, 4.2, -0.7, 0.3, None, 8.7]

In addition to the properties and methods described in :doc:`ak.layout.Content`,
an IndexedOptionArray has the following.

ak.layout.IndexedOptionArray.__init__
=====================================

.. py:method:: ak.layout.IndexedOptionArray.__init__(index, content, identities=None, parameters=None)

ak.layout.IndexedOptionArray.index
==================================

.. py:attribute:: ak.layout.IndexedOptionArray.index

ak.layout.IndexedOptionArray.content
====================================

.. py:attribute:: ak.layout.IndexedOptionArray.content

ak.layout.IndexedOptionArray.isoption
=====================================

.. py:attribute:: ak.layout.IndexedOptionArray.isoption

Returns True because this is an IndexedOptionArray.

ak.layout.IndexedOptionArray.project
====================================

.. py:method:: ak.layout.IndexedOptionArray.project(mask=None)

Returns a non-:doc:`ak.types.OptionType` array containing only the valid elements
with the ``index`` applied to reorder/duplicate elements.

If ``mask`` is a signed 8-bit :doc:`ak.layout.Index` in which ``0`` means valid
and ``1`` means missing, this ``mask`` is unioned with the BitMaskedArray's
mask (after converting to 8-bit and to ``valid_when=False`` to match this ``mask``).

ak.layout.IndexedOptionArray.bytemask
=====================================

.. py:method:: ak.layout.IndexedOptionArray.bytemask()

Returns an array of 8-bit values in which ``0`` means valid and ``1`` means missing.

ak.layout.IndexedOptionArray.simplify
=====================================

.. py:method:: ak.layout.IndexedOptionArray.simplify()

Combines this node with its ``content`` if the ``content`` also has
:doc:`ak.types.OptionType` or is an :doc:`ak.layout.IndexedArray`; otherwise, this is
a pass-through.  In all cases, the output has the same logical meaning as the input.

This method only operates one level deep.
