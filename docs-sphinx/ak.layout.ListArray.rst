ak.layout.ListArray
-------------------

The ListArray concept is implemented in 3 specialized classes:

    * ``ak.layout.ListArray64``: ``starts`` and ``stops`` values are 64-bit
      signed integers.
    * ``ak.layout.ListArray32``: ``starts`` and ``stops`` values are 32-bit
      signed integers.
    * ``ak.layout.ListArrayU32``: ``starts`` and ``stops`` values are 32-bit
      unsigned integers.

The ListArray class generalizes :doc:`ak.layout.ListOffsetArray` by not
requiring its ``content`` to be in increasing order and by allowing it to
have unreachable elements between lists. Instead of a single ``offsets`` array,
ListArray has

   * ``starts``: The starting index of each list.
   * ``stops``: The stopping index of each list.

:doc:`ak.layout.ListOffsetArray` ``offsets`` may be related to ``starts`` and
``stops``:

.. code-block:: python

    starts = offsets[:-1]
    stops = offsets[1:]

ListArrays are a common by-product of structure manipulation: as a result of
some operation, we might want to view slices or permutations of the ``content``
without copying it to make a contiguous version of it. For that reason,
ListArrays are more useful in a data-manipulation library like Awkward Array
than in a data-representation library like Apache Arrow.

There is not equivalent of ListArray in Apache Arrow.

Below is a simplified implementation of a ListArray class in pure Python
that exhaustively checks validity in its constructor (see
:doc:`_auto/ak.isvalid`) and can generate random valid arrays. The
``random_number()`` function returns a random float and the
``random_length(minlen)`` function returns a random int that is at least
``minlen``. The ``RawArray`` class represents simple, one-dimensional data.

.. code-block:: python

    class ListArray(Content):
        def __init__(self, starts, stops, content):
            assert isinstance(starts, list)
            assert isinstance(stops, list)
            assert isinstance(content, Content)
            assert len(stops) >= len(starts)   # usually equal
            for i in range(len(starts)):
                start = starts[i]
                stop = stops[i]
                assert isinstance(start, int)
                assert isinstance(stop, int)
                if start != stop:
                    assert start < stop   # i.e. start <= stop
                    assert start >= 0
                    assert stop <= len(content)
            self.starts = starts
            self.stops = stops
            self.content = content

        @staticmethod
        def random(minlen, choices):
            content = random.choice(choices).random(0, choices)
            length = random_length(minlen)
            if len(content) == 0:
                starts = [random.randint(0, 10) for i in range(length)]
                stops = list(starts)
            else:
                starts = [random.randint(0, len(content) - 1) for i in range(length)]
                stops = [x + min(random_length(), len(content) - x) for x in starts]
            return ListArray(starts, stops, content)
            
        def __len__(self):
            return len(self.starts)

        def __getitem__(self, where):
            if isinstance(where, int):
                assert 0 <= where < len(self)
                return self.content[self.starts[where]:self.stops[where]]
            elif isinstance(where, slice) and where.step is None:
                starts = self.starts[where.start:where.stop]
                stops = self.stops[where.start:where.stop]
                return ListArray(starts, stops, self.content)
            elif isinstance(where, str):
                return ListArray(self.starts, self.stops, self.content[where])
            else:
                raise AssertionError(where)

        def __repr__(self):
            return ("ListArray(" + repr(self.starts) + ", " + repr(self.stops) + ", "
                    + repr(self.content) + ")")

        def xml(self, indent="", pre="", post=""):
            out = indent + pre + "<ListArray>\n"
            out += indent + "    <starts>" + " ".join(str(x) for x in self.starts)
            out += "</starts>\n"
            out += indent + "    <stops>" + " ".join(str(x) for x in self.stops) + "</stops>\n"
            out += self.content.xml(indent + "    ", "<content>", "</content>\n")
            out += indent + "</ListArray>" + post
            return out

Here is an example:

.. code-block:: python

    ListArray([5, 1, 4, 1, 1, 1, 0, 0, 4, 3, 5],
              [6, 2, 5, 6, 6, 1, 6, 6, 6, 3, 6],
              RawArray([13.3, 3.8, 5.9, 5.9, 9.2, 9.3]))

.. code-block:: xml

    <ListArray>
        <starts>5 1 4 1 1 1 0 0 4 3 5</starts>
        <stops>6 2 5 6 6 1 6 6 6 3 6</stops>
        <content><RawArray>
            <ptr>13.3 3.8 5.9 5.9 9.2 9.3</ptr>
        </RawArray></content>
    </ListArray>

which represents the following logical data.

.. code-block:: python

    [[9.3],
     [3.8],
     [9.2],
     [3.8, 5.9, 5.9, 9.2, 9.3],
     [3.8, 5.9, 5.9, 9.2, 9.3],
     [],
     [13.3, 3.8, 5.9, 5.9, 9.2, 9.3],
     [13.3, 3.8, 5.9, 5.9, 9.2, 9.3],
     [9.2, 9.3],
     [],
     [9.3]]

In addition to the properties and methods described in :doc:`ak.layout.Content`,
a ListArray has the following.

ak.layout.ListArray.__init__
============================

.. py:method:: ak.layout.ListArray.__init__(starts, stops, content, identities=None, parameters=None)

ak.layout.ListArray.starts
==========================

.. py:attribute:: ak.layout.ListArray.starts

ak.layout.ListArray.stops
=========================

.. py:attribute:: ak.layout.ListArray.stops

ak.layout.ListArray.content
===========================

.. py:attribute:: ak.layout.ListArray.content

ak.layout.ListArray.compact_offsets64
=====================================

.. py:method:: ak.layout.ListArray.compact_offsets64(start_at_zero=True)

Returns a 64-bit :doc:`ak.layout.Index` of ``offsets`` that represent the same lengths
of this array's ``starts`` and ``stops`` (though not the physical order in memory).

ak.layout.ListArray.broadcast_tooffsets64
=========================================

.. py:method:: ak.layout.ListArray.broadcast_tooffsets64(offsets)

Reorders ``contents`` to match a given set of ``offsets`` (if possible) and
returns a :doc:`ak.layout.ListOffsetArray` with the results. This is used in
broadcasting because a set of :doc:`ak.types.ListType` and :doc:`ak.types.RegularType`
arrays have to be reordered to a common ``offsets`` before they can be directly
operated upon.

ak.layout.ListArray.toRegularArray
==================================

.. py:method:: ak.layout.ListArray.toRegularArray()

Converts this :doc:`ak.types.ListType` into a :doc:`ak.types.RegularType` array
if possible.

ak.layout.ListArray.simplify
============================

.. py:method:: ak.layout.ListArray.simplify()

Pass-through; returns the original array.
