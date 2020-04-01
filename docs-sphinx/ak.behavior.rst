ak.behavior
-----------

A data structure is defined both in terms of the information it encodes and
in how it can be used. For example, a hash-table is not just a buffer, it's
also the "get" and "set" operations that make the buffer usable as a key-value
store. Awkward Arrays have a suite of operations for transforming tree
structures into new tree structures, but an application of these structures to
a data analysis problem should be able to interpret them as objects in the
analysis domain, such as latitude-longitude coordinates in geographical
studies or Lorentz vectors in particle physics.

Object-oriented programming unites data with its operations. This is a
conceptual improvement for data analysts because functions like "distance
between this latitude-longitude point and another on a spherical globe" can
be bound to the objects that represent latitude-longitude points. It
matches the way that data analysts want to think about their data.

However, if these methods are saved in the data, or are written in a way
that will only work for one version of the data structures, then it becomes
difficult to work with large datasets. Old data that do not "fit" the new
methods would have to be converted, or the analysis would have to be broken
into different cases for each data generation. This problem is known as
schema evolution, and there are many solutions to it.

The approach taken by the Awkward Array library is to encode very little
interpretation into the data themselves and apply an interpretation as
late as possible. Thus, a latitude-longitude record might be stamped with
the name ``"latlon"``, but the operations on it are added immediately before
the user wants them. These operations can be written in such a way that
they only require the ``"latlon"`` to have ``lat`` and ``lon`` fields, so
different versions of the data can have additional fields or even be
embedded in different structures.

Metadata can be embedded in data using an array node's **parameters**, and
parameter-dependent operations can be defined using **behavior**. A global
mapping from parameters to behavior is in a dict called

.. code-block:: python

    ak.behavior

but behavior dicts can also be loaded into :doc:`_auto/ak.Array`,
:doc:`_auto/ak.Record`, and :doc:`_auto/ak.ArrayBuilder` objects as a
constructor argument. See
`ak.Array.behavior <_auto/ak.Array.html#ak-array-behavior>`_.

The general flow is

   * **parameters** link data objects to names;
   * **behavior** links names to code.

In large datasets, parameters may be hard to change (permanently, at least:
on-the-fly parameter changes are easier), but behavior is easy to change
(it is always assigned on-the-fly).

In the following example, we create two nested arrays of records with fields
``"x"`` and ``"y"`` and the records are named ``"point"``.

.. code-block:: python

    one = ak.Array([[{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}],
                    [],
                    [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}],
                    [{"x": 6, "y": 6.6}],
                    [{"x": 7, "y": 7.7}, {"x": 8, "y": 8.8}, {"x": 9, "y": 9.9}]],
                   with_name="point")
    two = ak.Array([[{"x": 0.9, "y": 1}, {"x": 1.9, "y": 2}, {"x": 2.9, "y": 3}],
                    [],
                    [{"x": 3.9, "y": 4}, {"x": 4.9, "y": 5}],
                    [{"x": 5.9, "y": 6}],
                    [{"x": 6.9, "y": 7}, {"x": 7.9, "y": 8}, {"x": 8.9, "y": 9}]],
                   with_name="point")

The name is assigned through a special ``"__record__"`` parameter, which we can
see in the array's type:

.. code-block:: python

    >>> ak.type(one)
    5 * var * point["x": int64, "y": float64]

