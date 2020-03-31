ak.layout.UnionArray
--------------------

The UnionArray concept is implemented in 3 specialized classes:

    * ``ak.layout.UnionArray8_64``: ``tags`` values are 8-bit signed integers
      and ``index`` values are 64-bit signed integers.
    * ``ak.layout.UnionArray8_32``: ``tags`` values are 8-bit signed integers
      and ``index`` values are 32-bit signed integers.
    * ``ak.layout.UnionArray8_U32``: ``tags`` values are 8-bit signed integers
      and ``index`` values are 32-bit unsigned integers.

The UnionArray class represents data drawn from an ordered list of ``contents``,
which can have different types, using

   * ``tags``: array of integers indicating which content each array element
     draws from.
   * ``index``: array of integers indicating which element from the content
     to draw from.

UnionArrays correspond to Apache Arrow's
`dense union type <https://arrow.apache.org/docs/format/Columnar.html#dense-union>`__.
Awkward Array has no direct equivalent for Apache Arrow's
`sparse union type <https://arrow.apache.org/docs/format/Columnar.html#sparse-union>`__,
but an appropriate index may be generated with
`sparse_index <#ak.layout.UnionArray.sparse_index>`_.

Below is a simplified implementation of a RecordArray class in pure Python
that exhaustively checks validity in its constructor (see
:doc:`_auto/ak.isvalid`) and can generate random valid arrays. The
``random_number()`` function returns a random float and the
``random_length(minlen)`` function returns a random int that is at least
``minlen``. The ``RawArray`` class represents simple, one-dimensional data.

.. code-block:: python

    class UnionArray(Content):
        def __init__(self, tags, index, contents):
            assert isinstance(tags, list)
            assert isinstance(index, list)
            assert isinstance(contents, list)
            assert len(index) >= len(tags)   # usually equal
            for x in tags:
                assert isinstance(x, int)
                assert 0 <= x < len(contents)
            for i, x in enumerate(index):
                assert isinstance(x, int)
                assert 0 <= x < len(contents[tags[i]])
            self.tags = tags
            self.index = index
            self.contents = contents

        @staticmethod
        def random(minlen, choices):
            contents = []
            unshuffled_tags = []
            unshuffled_index = []
            for i in range(random.randint(1, 3)):
                if minlen == 0:
                    contents.append(random.choice(choices).random(0, choices))
                else:
                    contents.append(random.choice(choices).random(1, choices))
                if len(contents[-1]) != 0:
                    thisindex = [random.randint(0, len(contents[-1]) - 1)
                                     for i in range(random_length(minlen))]
                    unshuffled_tags.extend([i] * len(thisindex))
                    unshuffled_index.extend(thisindex)
            permutation = list(range(len(unshuffled_tags)))
            random.shuffle(permutation)
            tags = [unshuffled_tags[i] for i in permutation]
            index = [unshuffled_index[i] for i in permutation]
            return UnionArray(tags, index, contents)

        def __len__(self):
            return len(self.tags)

        def __getitem__(self, where):
            if isinstance(where, int):
                assert 0 <= where < len(self)
                return self.contents[self.tags[where]][self.index[where]]
            elif isinstance(where, slice) and where.step is None:
                return UnionArray(self.tags[where], self.index[where], self.contents)
            elif isinstance(where, str):
                return UnionArray(self.tags, self.index, [x[where] for x in self.contents])
            else:
                raise AssertionError(where)

        def __repr__(self):
            return ("UnionArray(" + repr(self.tags) + ", " + repr(self.index)
                    + ", [" + ", ".join(repr(x) for x in self.contents) + "])")

        def xml(self, indent="", pre="", post=""):
            out = indent + pre + "<UnionArray>\n"
            out += indent + "    <tags>" + " ".join(str(x) for x in self.tags) + "</tags>\n"
            out += indent + "    <index>" + " ".join(str(x) for x in self.index) + "</index>\n"
            for i, content in enumerate(self.contents):
                out += content.xml(indent + "    ", "<content i=\"" + str(i) + "\">",
                                   "</content>\n")
            out += indent + "</UnionArray>" + post
            return out

Here is an example:

.. code-block:: python

    UnionArray([0, 1, 2, 0, 2, 2, 1],
               [0, 16, 9, 0, 10, 0, 13],
               [ListOffsetArray([10, 21, 22, 50, 54, 55, 59, 89, 92, 101, 111, 119, 120, 131,
                                 138, 158, 165, 171, 173],
                RawArray([0.5, 4.8, 8.6, -1.3, 4.0, 2.5, 5.0, 3.3, 5.0, 1.5, 9.3, 2.5, 5.4, 2.1,
                          7.1, 5.3, 10.8, -2.1, 6.4, 7.6, 5.6, 6.2, 4.9, 8.0, 6.2, 4.1, 6.6,
                          -1.3, 4.0, 3.8, 0.3, 5.7, 9.9, 5.6, 9.9, 9.4, 1.4, 3.9, 6.2, 6.3, 3.4,
                          6.2, 10.1, 3.7, 8.3, -0.6, 2.8, 9.7, 3.3, 6.5, 6.5, 2.1, 4.9, 5.8, 1.0,
                          6.8, 2.7, 3.2, 6.0, 6.4, 1.9, 8.1, 5.5, 6.3, 4.8, 5.5, 1.1, 0.1, 4.0,
                          1.8, 10.0, 3.8, 3.9, 2.5, 1.8, 6.0, 5.2, 6.0, 9.6, 11.7, 6.4, 7.9, 4.3,
                          5.3, 4.4, 7.0, 8.6, 6.1, 11.2, 4.7, 5.9, 9.3, 7.0, 5.1, 8.0, 6.9, 8.4,
                          3.7, 5.8, 4.8, 1.6, -1.5, -0.9, 6.0, 2.8, -0.2, 8.1, 2.9, 7.6, 5.7,
                          8.3, 8.1, 5.5, 7.1, 6.5, 0.8, 4.3, 1.9, 0.2, 7.7, 5.6, -0.5, 2.1, 6.1,
                          7.1, 4.5, 4.5, 4.2, 9.1, 5.7, 2.2, 9.0, 2.6, 3.8, 7.2, 3.2, 5.1, 6.6,
                          3.0, 6.6, 6.3, 4.8, 2.6, 3.7, 7.0, 5.2, 1.8, 4.2, 5.9, 2.2, 7.1, 6.1,
                          1.8, 4.2, 3.6, 3.0, 5.7, 2.1, 7.7, 1.5, 3.8, 6.4, 5.1, 7.4, 2.8, 3.3,
                          10.1, 8.0, 2.3, 4.5, 5.9, 6.0, 4.2, 2.6, 1.1, 2.5, 12.2])),
                RawArray([3.8, 5.3, 2.2, 4.9, 6.9, 5.6, -0.6, 3.2, 2.5, 2.6, 3.6, 6.9, 7.7, 4.7,
                          4.0, 5.1, 0.5, 4.0]),
                RawArray([6.2, 7.6, 7.6, -1.2, 5.0, 6.3, 6.8, 6.0, 3.2, 5.6, 2.3, 9.4, 1.6, 5.2,
                          6.1, 1.2])])

.. code-block:: xml

    <UnionArray>
        <tags>0 1 2 0 2 2 1</tags>
        <index>0 16 9 0 10 0 13</index>
        <content i="0"><ListOffsetArray>
            <offsets>10 21 22 50 54 55 59 89 92 101 111 119 120 131 138 158 165 171 173</offsets>
            <content><RawArray>
                <ptr>0.5 4.8 8.6 -1.3 4.0 2.5 5.0 3.3 5.0 1.5 9.3 2.5 5.4 2.1 7.1 5.3 10.8 -2.1
                     6.4 7.6 5.6 6.2 4.9 8.0 6.2 4.1 6.6 -1.3 4.0 3.8 0.3 5.7 9.9 5.6 9.9 9.4 1.4
                     3.9 6.2 6.3 3.4 6.2 10.1 3.7 8.3 -0.6 2.8 9.7 3.3 6.5 6.5 2.1 4.9 5.8 1.0
                     6.8 2.7 3.2 6.0 6.4 1.9 8.1 5.5 6.3 4.8 5.5 1.1 0.1 4.0 1.8 10.0 3.8 3.9 2.5
                     1.8 6.0 5.2 6.0 9.6 11.7 6.4 7.9 4.3 5.3 4.4 7.0 8.6 6.1 11.2 4.7 5.9 9.3
                     7.0 5.1 8.0 6.9 8.4 3.7 5.8 4.8 1.6 -1.5 -0.9 6.0 2.8 -0.2 8.1 2.9 7.6 5.7
                     8.3 8.1 5.5 7.1 6.5 0.8 4.3 1.9 0.2 7.7 5.6 -0.5 2.1 6.1 7.1 4.5 4.5 4.2 9.1
                     5.7 2.2 9.0 2.6 3.8 7.2 3.2 5.1 6.6 3.0 6.6 6.3 4.8 2.6 3.7 7.0 5.2 1.8 4.2
                     5.9 2.2 7.1 6.1 1.8 4.2 3.6 3.0 5.7 2.1 7.7 1.5 3.8 6.4 5.1 7.4 2.8 3.3 10.1
                     8.0 2.3 4.5 5.9 6.0 4.2 2.6 1.1 2.5 12.2</ptr>
            </RawArray></content>
        </ListOffsetArray></content>
        <content i="1"><RawArray>
            <ptr>3.8 5.3 2.2 4.9 6.9 5.6 -0.6 3.2 2.5 2.6 3.6 6.9 7.7 4.7 4.0 5.1 0.5 4.0</ptr>
        </RawArray></content>
        <content i="2"><RawArray>
            <ptr>6.2 7.6 7.6 -1.2 5.0 6.3 6.8 6.0 3.2 5.6 2.3 9.4 1.6 5.2 6.1 1.2</ptr>
        </RawArray></content>
    </UnionArray>

which represents the following logical data.

.. code-block:: python

    [[9.3, 2.5, 5.4, 2.1, 7.1, 5.3, 10.8, -2.1, 6.4, 7.6, 5.6],
     0.5,
     5.6,
     [9.3, 2.5, 5.4, 2.1, 7.1, 5.3, 10.8, -2.1, 6.4, 7.6, 5.6],
     2.3,
     6.2,
     4.7]

In addition to the properties and methods described in :doc:`ak.layout.Content`,
a UnionArray has the following.

ak.layout.UnionArray.__init__
=============================

.. py:method:: ak.layout.UnionArray.__init__(tags, index, contents, identities=None, parameters=None)

ak.layout.UnionArray.sparse_index
=================================

.. py:method:: ak.layout.UnionArray.sparse_index(length)

ak.layout.UnionArray.regular_index
==================================

.. py:method:: ak.layout.UnionArray.regular_index(tags)

ak.layout.UnionArray.tags
=========================

.. py:attribute:: ak.layout.UnionArray.tags

ak.layout.UnionArray.index
==========================

.. py:attribute:: ak.layout.UnionArray.index

ak.layout.UnionArray.contents
=============================

.. py:attribute:: ak.layout.UnionArray.contents

ak.layout.UnionArray.numcontents
================================

.. py:attribute:: ak.layout.UnionArray.numcontents

ak.layout.UnionArray.content
============================

.. py:method:: ak.layout.UnionArray.content(index)

ak.layout.UnionArray.project
============================

.. py:method:: ak.layout.UnionArray.project()

ak.layout.UnionArray.simplify
=============================

.. py:method:: ak.layout.UnionArray.simplify(mergebool=False)
