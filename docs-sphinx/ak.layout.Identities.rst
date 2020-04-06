ak.layout.Identities
--------------------

The Identities concept is implemented in 2 specialized classes:

    * ``ak.layout.Identities64``: values are 64-bit signed integers.
    * ``ak.layout.Identities32``: values are 32-bit signed integers.

Identities are optional auxiliary arrays that can be attached to any
:doc:`ak.layout.Content` node. The Identities label every element of an
array (at every node of the tree) with the original ``__getitem__`` path
from the root of a data structure to that element, the original "identity"
of the element. As data are filtered with selections, shuffled with
sorting operations, or duplicated in combinatorial functions like
:doc:`_auto/ak.cartesian` and :doc:`_auto/ak.combinations`, the Identities
are selected, shuffled, and duplicated with them.

The primary purpose of these Identities is to act as an index key for
SQL-like join operations. At the time of this writing, such operations
have not been written, but Identities are included in the framework as a
placeholder to ensure that this will be possible later.

The identity of a single item is equivalent to a tuple of int and str, such
as

.. code-block:: python

    (999, "muons", 1, "pt")

This tuple corresponds to the original path from the root of the array structure
to that value:

.. code-block:: python

    array[999, "muons", 1, "pt"]

After filtering, reordering, and/or duplicating, the actual path of the
manipulated array might be very different from this, but two elements sharing
an identity means they originally came from the same value.

In addition to the tuple, an Identities has a globally unique (per process)
reference integer, to distinguish between Identities from originally
distinct arrays.

The integer values of each identity are collectively stored in a 2-dimensional
array. The width of that array would include ``999`` and ``1`` from the above
example and the length of that array includes all the distinct identities from
distinct objects.

The strings of the tuple are stored in a "fieldloc," a list of int-str pairs
that specify the index positions where fields are located. Whereas a large,
2-dimensional array is needed to express all row positions, only a small,
indexed set of strings are needed to express all column positions.

Here is an example of a properly formatted Identities object (with 64-bit
integers):

.. code-block:: python

    >>> identities = ak.layout.Identities64(0, [(0, "muons"), (1, "pt")],
    ...                  np.array([[0, 0], [0, 1], [0, 2], [2, 0], [2, 1]]))
    >>> for i in range(5):
    ...     print(identities.identity_at(i))
    ... 
    (0, 'muons', 0, 'pt')
    (0, 'muons', 1, 'pt')
    (0, 'muons', 2, 'pt')
    (2, 'muons', 0, 'pt')
    (2, 'muons', 1, 'pt')

(The ``"muons"`` field has index ``0`` because it comes *after* axis ``0``
in the NumPy array, and the ``"pt"`` field has index ``1`` because it comes
*after* index ``1``. It is possible for fields in the sequence to have the
same index if there are records nested directly inside records, with no
list dimensions in between.)

These identities would correspond with an array like

.. code-block:: python

    [{"muons": [{"pt": ...}, {"pt": ...}, {"pt": ...}]},
     {"muons": [{"pt": ...}]},
     {"muons": [{"pt": ...}, {"pt": ...}]}]

ak.layout.Identities.newref
===========================

.. py:method:: ak.layout.Identities.newref()

Static method to create a new reference (int). This reference is globally
unique in the process, as it is implemented with an atomic integer.

ak.layout.Identities.__init__
=============================

.. py:method:: ak.layout.Identities.__init__(ref, fieldloc, array)

Creates a new Identities from a reference (int), fieldloc (list of int-str
pairs), and a 2-dimensional array (np.ndarray) of integers.

.. py:method:: ak.layout.Identities.__init__(ref, fieldloc, width, length)

Allocates a new Identities from a reference (int), fieldloc (list of int-str
pairs), and a width and height for the 2-dimensional array.

The data in the newly allocated array are uninitialized, but the Identities
object is a buffer that may be cast as NumPy or the array can be accessed
from the ``array`` property to set its values.

ak.layout.Identities.array
==========================

.. py:attribute:: ak.layout.Identities.array

The 2-dimensional array containing all the numeric row data.

ak.layout.Identities.fieldloc
=============================

.. py:attribute:: ak.layout.Identities.fieldloc

The list of int-str pairs containing all the string field data.

ak.layout.Identities.__getitem__
================================

.. py:method:: ak.layout.Identities.__getitem__(at)

ak.layout.Identities.__getitem__
================================

.. py:method:: ak.layout.Identities.__getitem__(start, stop)

ak.layout.Identities.__len__
============================

.. py:method:: ak.layout.Identities.__len__()

ak.layout.Identities.__repr__
=============================

.. py:method:: ak.layout.Identities.__repr__()

ak.layout.Identities.identity_at
================================

.. py:method:: ak.layout.Identities.identity_at(at)

ak.layout.Identities.identity_at_str
====================================

.. py:method:: ak.layout.Identities.identity_at_str(at)
