ak.layout.ByteMaskedArray
-------------------------

The ByteMaskedArray implements an :doc:`ak.types.OptionType` with two aligned
arrays, a boolean ``mask`` and ``content``. At any element ``i`` where
``mask[i] == valid_when``, the value can be found at ``content[i]``. If
``mask[i] != valid_when``, the value is missing (None).

This is equivalent to NumPy's
`masked arrays <https://docs.scipy.org/doc/numpy/reference/maskedarray.html>`__
if ``valid_when=False``.

There is no Apache Arrow equivalent because Arrow
`uses bitmaps <https://arrow.apache.org/docs/format/Columnar.html#validity-bitmaps>`__
to mask all node types.

Below is a simplified implementation of a RecordArray class in pure Python
that exhaustively checks validity in its constructor (see
:doc:`_auto/ak.is_valid`) and can generate random valid arrays. The
``random_number()`` function returns a random float and the
``random_length(minlen)`` function returns a random int that is at least
``minlen``. The ``RawArray`` class represents simple, one-dimensional data.

.. code-block:: python

    class ByteMaskedArray(Content):
        def __init__(self, mask, content, valid_when):
            assert isinstance(mask, list)
            assert isinstance(content, Content)
            assert isinstance(valid_when, bool)
            for x in mask:
                assert isinstance(x, bool)
            assert len(mask) <= len(content)
            self.mask = mask
            self.content = content
            self.valid_when = valid_when

        @staticmethod
        def random(minlen, choices):
            mask = []
            for i in range(random_length(minlen)):
                mask.append(bool(random.randint(0, 1)))
            content = random.choice(choices).random(len(mask), choices)
            return ByteMaskedArray(mask, content, bool(random.randint(0, 1)))

        def __len__(self):
            return len(self.mask)

        def __getitem__(self, where):
            if isinstance(where, int):
                assert 0 <= where < len(self)
                if self.mask[where] == self.valid_when:
                    return self.content[where]
                else:
                    return None
            elif isinstance(where, slice) and where.step is None:
                return ByteMaskedArray(self.mask[where.start:where.stop],
                                       self.content[where.start:where.stop])
            elif isinstance(where, str):
                return ByteMaskedArray(self.mask, self.content[where])
            else:
                raise AssertionError(where)

        def __repr__(self):
            return ("ByteMaskedArray(" + repr(self.mask) + ", " + repr(self.content)
                    + ", " + repr(self.valid_when) + ")")

        def xml(self, indent="", pre="", post=""):
            out = indent + pre + "<ByteMaskedArray>\n"
            out += indent + "    <valid_when>" + repr(self.valid_when) + "<valid_when>\n"
            out += indent + "    <mask>" + " ".join(str(x) for x in self.mask) + "</mask>\n"
            out += self.content.xml(indent + "    ", "<content>", "</content>\n")
            out += indent + "</ByteMaskedArray>\n"
            return out

Here is an example:

.. code-block:: python

    ByteMaskedArray([True, True, False, False, True, False, False, True, True, True, True, True],
                    RawArray([5.7, 4.5, 8.3, 4.1, 5.1, 4.1, 0.3, 6.4, 5.5, 9.5, 7.1, 7.7, 4.0,
                              4.8, 4.4, 2.9, 1.4, 4.8, 7.3, 4.9, 6.0, 0.6, 11.2, 6.1, 4.7, 4.1,
                              4.4, 5.9, 7.6, 6.3, 5.5, 11.0, 9.2, 5.3, 0.1, 1.2, 4.5, 6.4, 2.8,
                              1.4, 5.8]),
                    False)

.. code-block:: xml

    <ByteMaskedArray>
        <valid_when>False</valid_when>
        <mask>True True False False True False False True True True True True</mask>
        <content><RawArray>
            <ptr>5.7 4.5 8.3 4.1 5.1 4.1 0.3 6.4 5.5 9.5 7.1 7.7 4.0 4.8 4.4 2.9 1.4 4.8 7.3 4.9
                 6.0 0.6 11.2 6.1 4.7 4.1 4.4 5.9 7.6 6.3 5.5 11.0 9.2 5.3 0.1 1.2 4.5 6.4 2.8
                 1.4 5.8</ptr>
        </RawArray></content>
    </ByteMaskedArray>

which represents the following logical data.

.. code-block:: python

    [None, None, 8.3, 4.1, None, 4.1, 0.3, None, None, None, None, None]

In addition to the properties and methods described in :doc:`ak.layout.Content`,
a ByteMaskedArray has the following.

ak.layout.ByteMaskedArray.__init__
==================================

.. py:method:: ak.layout.ByteMaskedArray.__init__(mask, content, valid_when, identities=None, parameters=None)

ak.layout.ByteMaskedArray.mask
==============================

.. py:attribute:: ak.layout.ByteMaskedArray.mask

ak.layout.ByteMaskedArray.content
=================================

.. py:attribute:: ak.layout.ByteMaskedArray.content

ak.layout.ByteMaskedArray.valid_when
====================================

.. py:attribute:: ak.layout.ByteMaskedArray.valid_when

ak.layout.ByteMaskedArray.project
=================================

.. py:method:: ak.layout.ByteMaskedArray.project(mask=None)

Returns a non-:doc:`ak.types.OptionType` array containing only the valid elements.
If ``mask`` is a signed 8-bit :doc:`ak.layout.Index` in which ``0`` means valid
and ``1`` means missing, this ``mask`` is unioned with the ByteMaskedArray's
mask (after converting to ``valid_when=False`` to match this ``mask``).

ak.layout.ByteMaskedArray.bytemask
==================================

.. py:method:: ak.layout.ByteMaskedArray.bytemask()

Returns an array of 8-bit values in which ``0`` means valid and ``1`` means missing.

ak.layout.ByteMaskedArray.simplify
==================================

.. py:method:: ak.layout.ByteMaskedArray.simplify()

Combines this node with its ``content`` if the ``content`` also has
:doc:`ak.types.OptionType`; otherwise, this is a pass-through.
In all cases, the output has the same logical meaning as the input.

This method only operates one level deep.
