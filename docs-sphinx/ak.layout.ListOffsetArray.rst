ak.layout.ListOffsetArray
-------------------------

The ListOffsetArray concept is implemented in 3 specialized classes:

    * ``ak.layout.ListOffsetArray64``: ``offsets`` values are 64-bit signed
      integers.
    * ``ak.layout.ListOffsetArray32``: ``offsets`` values are 32-bit signed
      integers.
    * ``ak.layout.ListOffsetArrayU32``: ``offsets`` values are 32-bit
      unsigned integers.

The ListOffsetArray class describes unequal-length lists (often called a
"jagged" or "ragged" array). Like :doc:`ak.layout.RegularArray`, the
underlying data for all lists are in a contiguous ``content``. It is
subdivided into lists according to an ``offsets`` array, which specifies
the starting and stopping index of each list.

The ``offsets`` must have at least length 1 (corresponding to an empty array),
but it need not start with ``0`` or include all of the ``content``. Just as
:doc:`ak.layout.RegularArray` can have unreachable ``content`` if it is not
an integer multiple of ``size``, a ListOffsetArray can have unreachable
content before the first list and after the last list.

ListOffsetArray corresponds to Apache Arrow `List type <https://arrow.apache.org/docs/format/Columnar.html#variable-size-list-layout>`__.

Below is a simplified implementation of a ListOffsetArray class in pure Python
that exhaustively checks validity in its constructor (see
:doc:`_auto/ak.is_valid`) and can generate random valid arrays. The
``random_number()`` function returns a random float and the
``random_length(minlen)`` function returns a random int that is at least
``minlen``. The ``RawArray`` class represents simple, one-dimensional data.

.. code-block:: python

    class ListOffsetArray(Content):
        def __init__(self, offsets, content):
            assert isinstance(offsets, list)
            assert isinstance(content, Content)
            assert len(offsets) != 0
            for i in range(len(offsets) - 1):
                start = offsets[i]
                stop = offsets[i + 1]
                assert isinstance(start, int)
                assert isinstance(stop, int)
                if start != stop:
                    assert start < stop   # i.e. start <= stop
                    assert start >= 0
                    assert stop <= len(content)
            self.offsets = offsets
            self.content = content

        @staticmethod
        def random(minlen, choices):
            counts = [random_length() for i in range(random_length(minlen))]
            offsets = [random_length()]
            for x in counts:
                offsets.append(offsets[-1] + x)
            return ListOffsetArray(offsets, random.choice(choices).random(offsets[-1], choices))
            
        def __len__(self):
            return len(self.offsets) - 1

        def __getitem__(self, where):
            if isinstance(where, int):
                assert 0 <= where < len(self)
                return self.content[self.offsets[where]:self.offsets[where + 1]]
            elif isinstance(where, slice) and where.step is None:
                offsets = self.offsets[where.start : where.stop + 1]
                if len(offsets) == 0:
                    offsets = [0]
                return ListOffsetArray(offsets, self.content)
            elif isinstance(where, str):
                return ListOffsetArray(self.offsets, self.content[where])
            else:
                raise AssertionError(where)

        def __repr__(self):
            return "ListOffsetArray(" + repr(self.offsets) + ", " + repr(self.content) + ")"

        def xml(self, indent="", pre="", post=""):
            out = indent + pre + "<ListOffsetArray>\n"
            out += indent + "    <offsets>" + " ".join(str(x) for x in self.offsets)
            out += "</offsets>\n"
            out += self.content.xml(indent + "    ", "<content>", "</content>\n")
            out += indent + "</ListOffsetArray>" + post
            return out

Here is an example:

.. code-block:: python

    ListOffsetArray([0, 2, 4, 11, 19],
                    RawArray([5.9, 3.5, 2.2, 5.8, 7.4, 3.4, 2.7, 7.2, 6.6, 8.6, 8.2, 5.5, 3.8,
                              3.0, 8.4, 5.1, 1.2, -0.9, 3.7, 4.2, 0.8, 9.5, 4.0, 4.2, 4.2]))

.. code-block:: xml

    <ListOffsetArray>
        <offsets>0 2 4 11 19</offsets>
        <content><RawArray>
            <ptr>5.9 3.5 2.2 5.8 7.4 3.4 2.7 7.2 6.6 8.6 8.2 5.5 3.8 3.0 8.4 5.1 1.2 -0.9 3.7
                 4.2 0.8 9.5 4.0 4.2 4.2</ptr>
        </RawArray></content>
    </ListOffsetArray>

which represents the following logical data.

.. code-block:: python

    [[5.9, 3.5],
     [2.2, 5.8],
     [7.4, 3.4, 2.7, 7.2, 6.6, 8.6, 8.2],
     [5.5, 3.8, 3.0, 8.4, 5.1, 1.2, -0.9, 3.7]]

In addition to the properties and methods described in :doc:`ak.layout.Content`,
a ListOffsetArray has the following.

ak.layout.ListOffsetArray.__init__
==================================

.. py:method:: ak.layout.ListOffsetArray.__init__(offsets, content, identities=None, parameters=None)

ak.layout.ListOffsetArray.offsets
=================================

.. py:attribute:: ak.layout.ListOffsetArray.offsets

ak.layout.ListOffsetArray.content
=================================

.. py:attribute:: ak.layout.ListOffsetArray.content

ak.layout.ListOffsetArray.starts
================================

.. py:attribute:: ak.layout.ListOffsetArray.starts

Derives ``starts`` as a view of ``offsets``:

.. code-block:: python

    starts = offsets[:-1]

ak.layout.ListOffsetArray.stops
===============================

.. py:attribute:: ak.layout.ListOffsetArray.stops

Derives ``stops`` as a view of ``offsets``:

.. code-block:: python

    stops = offsets[1:]

ak.layout.ListOffsetArray.compact_offsets64
===========================================

.. py:method:: ak.layout.ListOffsetArray.compact_offsets64(start_at_zero=True)

Returns a 64-bit :doc:`ak.layout.Index` of ``offsets`` that represent the same lengths
of this array's ``offsets``. If this ``offsets[0] == 0 or not start_at_zero``, the
return value is a view of this array's ``offsets``.

ak.layout.ListOffsetArray.broadcast_tooffsets64
===============================================

.. py:method:: ak.layout.ListOffsetArray.broadcast_tooffsets64(offsets)

Shifts ``contents`` to match a given set of ``offsets`` (if possible) and
returns a :doc:`ak.layout.ListOffsetArray` with the results. This is used in
broadcasting because a set of :doc:`ak.types.ListType` and :doc:`ak.types.RegularType`
arrays have to be reordered to a common ``offsets`` before they can be directly
operated upon.

ak.layout.ListOffsetArray.toRegularArray
========================================

.. py:method:: ak.layout.ListOffsetArray.toRegularArray()

Converts this :doc:`ak.types.ListType` into a :doc:`ak.types.RegularType` array
if possible.

ak.layout.ListOffsetArray.simplify
==================================

.. py:method:: ak.layout.ListOffsetArray.simplify()

Pass-through; returns the original array.
