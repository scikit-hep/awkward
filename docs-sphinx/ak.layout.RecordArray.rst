ak.layout.RecordArray
---------------------

The RecordArray class represents an array of tuples or records, all with the
same type. Its ``contents`` is an ordered list of arrays.

   * If a ``recordlookup`` is absent, the data are tuples, indexed only by
     their order.
   * If a ``recordlookup`` is present, it is an ordered list of names with
     the same length as the ``contents``, associating a field name to every
     content.

The length of the RecordArray, if not given, is the length of its shortest
content; all are aligned element-by-element. If a RecordArray has zero contents,
it may still represent a non-empty array. In that case, its length is specified
by a ``length`` parameter.

RecordArrays correspond to Apache Arrow's `struct type <https://arrow.apache.org/docs/format/Columnar.html#struct-layout>`__.

Below is a simplified implementation of a RecordArray class in pure Python
that exhaustively checks validity in its constructor (see
:doc:`_auto/ak.is_valid`) and can generate random valid arrays. The
``random_number()`` function returns a random float and the
``random_length(minlen)`` function returns a random int that is at least
``minlen``. The ``RawArray`` class represents simple, one-dimensional data.

.. code-block:: python

    class RecordArray(Content):
        def __init__(self, contents, recordlookup, length):
            assert isinstance(contents, list)
            assert isinstance(length, int)
            for x in contents:
                assert isinstance(x, Content)
                assert len(x) >= length
            assert recordlookup is None or isinstance(recordlookup, list)
            if isinstance(recordlookup, list):
                assert len(recordlookup) == len(contents)
                for x in recordlookup:
                    assert isinstance(x, str)
            self.contents = contents
            self.recordlookup = recordlookup
            self.length = length

        @staticmethod
        def random(minlen, choices):
            length = random_length(minlen)
            contents = []
            for i in range(random.randint(0, 2)):
                contents.append(random.choice(choices).random(length, choices))
            if random.randint(0, 1) == 0:
                recordlookup = None
            else:
                recordlookup = ["x" + str(i) for i in range(len(contents))]
            return RecordArray(contents, recordlookup, length)

        def __len__(self):
            return self.length

        def __getitem__(self, where):
            if isinstance(where, int):
                assert 0 <= where < len(self)
                record = [x[where] for x in self.contents]
                if self.recordlookup is None:
                    return tuple(record)
                else:
                    return dict(zip(self.recordlookup, record))
            elif isinstance(where, slice) and where.step is None:
                if len(self.contents) == 0:
                    start = min(max(where.start, 0), self.length)
                    stop = min(max(where.stop, 0), self.length)
                    if stop < start:
                        stop = start
                    return RecordArray([], self.recordlookup, stop - start)
                else:
                    return RecordArray([x[where] for x in self.contents], self.recordlookup,
                                       self.length)
            elif isinstance(where, str):
                if self.recordlookup is None:
                    try:
                        i = int(where)
                    except ValueError:
                        pass
                    else:
                        if i < len(self.contents):
                            return self.contents[i]
                else:
                    try:
                        i = self.recordlookup.index(where)
                    except ValueError:
                        pass
                    else:
                        return self.contents[i]
                raise ValueError("field " + repr(where) + " not found")
            else:
                raise AssertionError(where)

        def __repr__(self):
            return ("RecordArray([" + ", ".join(repr(x) for x in self.contents) + "], "
                    + repr(self.recordlookup) + ", " + repr(self.length) + ")")

        def xml(self, indent="", pre="", post=""):
            out = indent + pre + "<RecordArray>\n"
            if len(self.contents) == 0:
                out += indent + "    <istuple>" + str(self.recordlookup is None) + "</istuple>\n"
            out += indent + "    <length>" + str(self.length) + "</length>\n"
            if self.recordlookup is None:
                for i, content in enumerate(self.contents):
                    out += content.xml(indent + "    ", "<content i=\"" + str(i) + "\">",
                                       "</content>\n")
            else:
                for i, (key, content) in enumerate(zip(self.recordlookup, self.contents)):
                    out += content.xml(indent + "    ", "<content i=\"" + str(i) + "\" key=\""
                                       + repr(key) + "\">", "</content>\n")
            out += indent + "</RecordArray>" + post
            return out

Here is an example:

.. code-block:: python

    RecordArray([RawArray([1.8, 6.2, 2.3, 7.2, 8.6, 6.0, 0.1, 4.6, 7.4, 3.6, 8.6, 10.7]),
                 RawArray([2.9, -0.9, 2.6, 0.9, -0.8, 5.3, 4.7, 1.2, 3.3, 5.5])],
                ['x0', 'x1'],
                10)

.. code-block:: xml

    <RecordArray>
        <length>10</length>
        <content i="0" key="'x0'"><RawArray>
            <ptr>1.8 6.2 2.3 7.2 8.6 6.0 0.1 4.6 7.4 3.6 8.6 10.7</ptr>
        </RawArray></content>
        <content i="1" key="'x1'"><RawArray>
            <ptr>2.9 -0.9 2.6 0.9 -0.8 5.3 4.7 1.2 3.3 5.5</ptr>
        </RawArray></content>
    </RecordArray>

which represents the following logical data.

.. code-block:: python

    [{'x0': 1.8, 'x1': 2.9},
     {'x0': 6.2, 'x1': -0.9},
     {'x0': 2.3, 'x1': 2.6},
     {'x0': 7.2, 'x1': 0.9},
     {'x0': 8.6, 'x1': -0.8},
     {'x0': 6.0, 'x1': 5.3},
     {'x0': 0.1, 'x1': 4.7},
     {'x0': 4.6, 'x1': 1.2},
     {'x0': 7.4, 'x1': 3.3},
     {'x0': 3.6, 'x1': 5.5}]

Here is an example without field names.

.. code-block:: python

    RecordArray([RawArray([1.5, 1.7, 2.6, 5.4, 5.8, 2.6, 7.0, 3.5, 7.1, 6.9, 6.3, 5.3, 2.9, 3.6,
                           3.7, 3.6, 0.8, 2.1, 0.4, -0.6, 5.1, 4.2, 9.5, 1.9, 8.4, 7.4, 6.5, 9.6,
                           7.7, 4.0, 5.4, 2.5, 6.7, 3.6, 7.4, 1.5, 3.6, 2.3, 3.6, 2.4, 4.7, 4.0,
                           6.0, 10.2, 4.7, 0.6]),
                 RawArray([6.5, 8.8, 2.4, 2.2, 5.0, 4.4, 7.7, 5.1, 6.2, 3.7, 6.7, 1.2])],
                 None,
                 12)

.. code-block:: xml

    <RecordArray>
        <length>12</length>
        <content i="0"><RawArray>
            <ptr>1.5 1.7 2.6 5.4 5.8 2.6 7.0 3.5 7.1 6.9 6.3 5.3 2.9 3.6 3.7 3.6 0.8 2.1 0.4
                 -0.6 5.1 4.2 9.5 1.9 8.4 7.4 6.5 9.6 7.7 4.0 5.4 2.5 6.7 3.6 7.4 1.5 3.6 2.3
                 3.6 2.4 4.7 4.0 6.0 10.2 4.7 0.6</ptr>
        </RawArray></content>
        <content i="1"><RawArray>
            <ptr>6.5 8.8 2.4 2.2 5.0 4.4 7.7 5.1 6.2 3.7 6.7 1.2</ptr>
        </RawArray></content>
    </RecordArray>

which represents the following logical data.

.. code-block:: python

    [(1.5, 6.5),
     (1.7, 8.8),
     (2.6, 2.4),
     (5.4, 2.2),
     (5.8, 5.0),
     (2.6, 4.4),
     (7.0, 7.7),
     (3.5, 5.1),
     (7.1, 6.2),
     (6.9, 3.7),
     (6.3, 6.7),
     (5.3, 1.2)]

And here is an example of records with no ``contents``, but a non-zero ``length``.

.. code-block:: python

    RecordArray([], [], 12)

.. code-block:: xml

    <RecordArray>
        <istuple>False</istuple>
        <length>12</length>
    </RecordArray>

which represents the following block of logical data.

.. code-block:: python

    [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]

In addition to the properties and methods described in :doc:`ak.layout.Content`,
a RecordArray has the following.

ak.layout.RecordArray.__init__
==============================

.. py:method:: ak.layout.RecordArray.__init__(contents, keys=None, length=None, identities=None, parameters=None)

ak.layout.RecordArray.contents
==============================

.. py:attribute:: ak.layout.RecordArray.contents

ak.layout.RecordArray.recordlookup
==================================

.. py:attribute:: ak.layout.RecordArray.recordlookup

ak.layout.RecordArray.istuple
=============================

.. py:attribute:: ak.layout.RecordArray.istuple

Returns True if ``recordlookup`` does not exist; False if it does.

ak.layout.RecordArray.astuple
=============================

.. py:attribute:: ak.layout.RecordArray.astuple

Returns a RecordArray with the ``recordlookup`` removed.

ak.layout.RecordArray.setitem_field
===================================

.. py:method:: ak.layout.RecordArray.setitem_field(where, what)

Sets a field in-place.

**Do not use this function.** It is deprecated. Use :doc:`_auto/ak.with_field`
instead.

ak.layout.RecordArray.field
===========================

.. py:method:: ak.layout.RecordArray.field(fieldindex)

Gets a field by index number.

ak.layout.RecordArray.field
===========================

.. py:method:: ak.layout.RecordArray.field(key)

Gets a field by str name.

ak.layout.RecordArray.fields
============================

.. py:method:: ak.layout.RecordArray.fields()

Returns a list of the fields themselves; equivalent to ``contents``.

ak.layout.RecordArray.fielditems
================================

.. py:method:: ak.layout.RecordArray.fielditems()

Returns a list of key-value pairs, where the values are ``contents``.

ak.layout.RecordArray.simplify
==============================

.. py:method:: ak.layout.RecordArray.simplify()

Pass-through; returns the original array.
