ak.layout.BitMaskedArray
------------------------

Like :doc:`ak.layout.ByteMaskedArray`, BitMaskedArray implements an
:doc:`ak.types.OptionType` with two arrays, ``mask`` and ``content``.
However, the boolean ``mask`` values are packed into a bitmap.

BitMaskedArray has an additional parameter, ``lsb_order``; if True,
the position of each bit is in
`Least-Significant Bit order <https://en.wikipedia.org/wiki/Bit_numbering>`__
(LSB):

.. code-block:: python

    is_valid[j] = bool(mask[j // 8] & (1 << (j % 8))) == valid_when

If False, the position of each bit is in Most-Significant Bit order
(MSB):

.. code-block:: python

    is_valid[j] = bool(mask[j // 8] & (128 >> (j % 8))) == valid_when

Note that NumPy's
`unpackbits <https://docs.scipy.org/doc/numpy/reference/generated/numpy.unpackbits.html>`__
function before version 1.17 has MSB order, but now it is configurable
(``bitorder="little"`` for LSB, ``bitorder="big"`` for MSB).

If the logical size of the array is not a multiple of 8, the ``mask``
has to be padded. Thus, an explicit ``length`` is also part of the
class's definition.

Below is a simplified implementation of a RecordArray class in pure Python
that exhaustively checks validity in its constructor (see
:doc:`_auto/ak.isvalid`) and can generate random valid arrays. The
``random_number()`` function returns a random float and the
``random_length(minlen)`` function returns a random int that is at least
``minlen``. The ``RawArray`` class represents simple, one-dimensional data.

.. code-block:: python

    class BitMaskedArray(Content):
        def __init__(self, mask, content, valid_when, length, lsb_order):
            assert isinstance(mask, np.ndarray)
            assert isinstance(content, Content)
            assert isinstance(valid_when, bool)
            assert isinstance(length, int) and length >= 0
            assert isinstance(lsb_order, bool)
            assert len(mask) <= len(content)
            self.mask = mask
            self.content = content
            self.valid_when = valid_when
            self.length = length
            self.lsb_order = lsb_order

        @staticmethod
        def random(minlen, choices):
            mask = []
            for i in range(random_length(minlen)):
                mask.append(bool(random.randint(0, 1)))
            lsb_order = bool(random.randint(0, 1))
            bitmask = np.packbits(np.array(mask, dtype=np.uint8),
                                  bitorder=("little" if lsb_order else "big"))
            content = random.choice(choices).random(len(mask), choices)
            return BitMaskedArray(bitmask, content, bool(random.randint(0, 1)),
                                  len(mask), lsb_order)

        def __len__(self):
            return self.length

        def __getitem__(self, where):
            if isinstance(where, int):
                assert 0 <= where < len(self)
                if self.lsb_order:
                    bit = bool(self.mask[where // 8] & (1 << (where % 8)))
                else:
                    bit = bool(self.mask[where // 8] & (128 >> (where % 8)))
                if bit == self.valid_when:
                    return self.content[where]
                else:
                    return None
            elif isinstance(where, slice) and where.step is None:
                # In general, slices must convert BitMaskedArray to ByteMaskedArray.
                bytemask = np.unpackbits(self.mask,
                               bitorder=("little" if self.lsb_order else "big"))
                return ByteMaskedArray(bytemask[where.start:where.stop],
                                       self.content[where.start:where.stop])
            elif isinstance(where, str):
                return BitMaskedArray(self.mask, self.content[where])
            else:
                raise AssertionError(where)

        def __repr__(self):
            return ("BitMaskedArray(" + repr(self.mask) + ", " + repr(self.content)
                    + ", " + repr(self.valid_when) + ", " + repr(self.length)
                    + ", " + repr(self.lsb_order) + ")")

        def xml(self, indent="", pre="", post=""):
            out = indent + pre + "<BitMaskedArray>\n"
            out += indent + "    <valid_when>" + repr(self.valid_when) + "<valid_when>\n"
            out += indent + "    <length>" + repr(self.length) + "<length>\n"
            out += indent + "    <lsb_order>" + repr(self.lsb_order) + "<lsb_order>\n"
            out += indent + "    <mask>" + " ".join(str(x) for x in self.mask) + "</mask>\n"
            out += self.content.xml(indent + "    ", "<content>", "</content>\n")
            out += indent + "</BitMaskedArray>\n"
            return out

Here is an example:

.. code-block:: python

    BitMaskedArray(np.array([ 40, 173,  59, 104, 182, 116], dtype=np.uint8),
                   RawArray([5.5, 6.6, 1.5, 3.2, 9.8, 0.4, 5.7, 1.5, 0.2, 6.1, 5.4, 4.3, 5.9,
                             10.1, -2.3, 5.8, 3.4, 5.6, 6.2, 8.8, 3.1, 7.0, 1.2, 7.3, 5.8, 8.3,
                             9.7, 5.2, 3.4, 5.8, 1.7, 4.3, 5.8, 1.2, 1.7, 3.6, 4.4, 9.7, 5.0,
                             4.3, 7.8, 6.1, 3.3, 7.9, 7.1, 6.5, -0.6, 8.2, 3.7, 4.6, 3.9, 7.5]),
                   False,
                   46,
                   False)

.. code-block:: xml

    <BitMaskedArray>
        <valid_when>False<valid_when>
        <length>46<length>
        <lsb_order>False<lsb_order>
        <mask>40 173 59 104 182 116</mask>
        <content><RawArray>
            <ptr>5.5 6.6 1.5 3.2 9.8 0.4 5.7 1.5 0.2 6.1 5.4 4.3 5.9 10.1 -2.3 5.8 3.4 5.6 6.2
                 8.8 3.1 7.0 1.2 7.3 5.8 8.3 9.7 5.2 3.4 5.8 1.7 4.3 5.8 1.2 1.7 3.6 4.4 9.7 5.0
                 4.3 7.8 6.1 3.3 7.9 7.1 6.5 -0.6 8.2 3.7 4.6 3.9 7.5</ptr>
        </RawArray></content>
    </BitMaskedArray>

which represents the following logical data.

.. code-block:: python

    [5.5, 6.6, None, 3.2, None, 0.4, 5.7, 1.5, None, 6.1, None, 4.3, None, None, -2.3, None, 3.4,
     5.6, None, None, None, 7.0, None, None, 5.8, None, None, 5.2, None, 5.8, 1.7, 4.3, None,
     1.2, None, None, 4.4, None, None, 4.3, 7.8, None, None, None, 7.1, None]

This is equivalent to *all* of Apache Arrow's array types because they all
`use bitmaps <https://arrow.apache.org/docs/format/Columnar.html#validity-bitmaps>`__
to mask their data, with ``valid_when=True`` and ``lsb_order=True``.

In addition to the properties and methods described in :doc:`ak.layout.Content`,
a BitMaskedArray has the following.

ak.layout.BitMaskedArray.__init__
=================================

.. py:method:: ak.layout.BitMaskedArray.__init__(mask, content, valid_when, length, lsb_order, identities=None, parameters=None)

ak.layout.BitMaskedArray.mask
=============================

.. py:attribute:: ak.layout.BitMaskedArray.mask

ak.layout.BitMaskedArray.content
================================

.. py:attribute:: ak.layout.BitMaskedArray.content

ak.layout.BitMaskedArray.valid_when
===================================

.. py:attribute:: ak.layout.BitMaskedArray.valid_when

ak.layout.BitMaskedArray.lsb_order
==================================

.. py:attribute:: ak.layout.BitMaskedArray.lsb_order

ak.layout.BitMaskedArray.project
================================

.. py:method:: ak.layout.BitMaskedArray.project(mask=None)

ak.layout.BitMaskedArray.bytemask
=================================

.. py:method:: ak.layout.BitMaskedArray.bytemask()

ak.layout.BitMaskedArray.simplify
=================================

.. py:method:: ak.layout.BitMaskedArray.simplify()

ak.layout.BitMaskedArray.toByteMaskedArray
==========================================

.. py:method:: ak.layout.BitMaskedArray.toByteMaskedArray()

ak.layout.BitMaskedArray.toIndexedOptionArray
=============================================

.. py:method:: ak.layout.BitMaskedArray.toIndexedOptionArray()
