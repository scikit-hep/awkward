ak.layout.UnmaskedArray
-----------------------

UnmaskedArray implements an :doc:`ak.types.OptionType` for which the values are
never, in fact, missing. It exists to satisfy systems that formally require this
high-level type without the overhead of generating an array of all True or all
False values.

This is like NumPy's
`masked arrays <https://docs.scipy.org/doc/numpy/reference/maskedarray.html>`__
with ``mask=None``.

It is also like Apache Arrow's
`validity bitmaps <https://arrow.apache.org/docs/format/Columnar.html#validity-bitmaps>`__
because the bitmap can be omitted when all values are valid.

Below is a simplified implementation of a RecordArray class in pure Python
that exhaustively checks validity in its constructor (see
:doc:`_auto/ak.isvalid`) and can generate random valid arrays. The
``random_number()`` function returns a random float and the
``random_length(minlen)`` function returns a random int that is at least
``minlen``. The ``RawArray`` class represents simple, one-dimensional data.

.. code-block:: python

    class UnmaskedArray(Content):
        def __init__(self, content):
            assert isinstance(content, Content)
            self.content = content

        @staticmethod
        def random(minlen, choices):
            content = random.choice(choices).random(minlen, choices)
            return UnmaskedArray(content)

        def __len__(self):
            return len(self.content)

        def __getitem__(self, where):
            return self.content[where]

        def __repr__(self):
            return "UnmaskedArray(" + repr(self.content) + ")"

        def xml(self, indent="", pre="", post=""):
            out = indent + pre + "<UnmaskedArray>\n"
            out += self.content.xml(indent + "    ", "<content>", "</content>\n")
            out += indent + "</UnmaskedArray>\n"
            return out

Here is an example:

.. code-block:: python

    UnmaskedArray(RawArray([6.0, 4.6, 4.2, 2.2, 2.4, 2.0, 8.3, 5.8, 6.8, 5.3, 0.4, 7.4, 0.9, 3.4,
                            7.6, 3.9, 8.9, 4.2, 4.0, 5.3, 1.9, 8.8]))

.. code-block:: xml

    <UnmaskedArray>
        <content><RawArray>
            <ptr>6.0 4.6 4.2 2.2 2.4 2.0 8.3 5.8 6.8 5.3 0.4 7.4 0.9 3.4 7.6 3.9 8.9 4.2 4.0 5.3
                 1.9 8.8</ptr>
        </RawArray></content>
    </UnmaskedArray>

which represents the following logical data.

.. code-block:: python

    [6.0, 4.6, 4.2, 2.2, 2.4, 2.0, 8.3, 5.8, 6.8, 5.3, 0.4, 7.4, 0.9, 3.4, 7.6, 3.9, 8.9, 4.2,
     4.0, 5.3, 1.9, 8.8]

In addition to the properties and methods described in :doc:`ak.layout.Content`,
a UnmaskedArray has the following.

ak.layout.UnmaskedArray.__init__
================================

.. py:method:: ak.layout.UnmaskedArray.__init__(content, identities=None, parameters=None)

ak.layout.UnmaskedArray.content
===============================

.. py:attribute:: ak.layout.UnmaskedArray.content

ak.layout.UnmaskedArray.project
===============================

.. py:method:: ak.layout.UnmaskedArray.project(mask=None)

ak.layout.UnmaskedArray.bytemask
================================

.. py:method:: ak.layout.UnmaskedArray.bytemask()

ak.layout.UnmaskedArray.simplify
================================

.. py:method:: ak.layout.UnmaskedArray.simplify()
