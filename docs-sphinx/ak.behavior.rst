ak.behavior
-----------

Motivation
==========

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
matches the way that data analysts usually think about their data.

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

Parameters and behaviors
========================

In Awkward Array, metadata are embedded in data using an array node's
**parameters**, and parameter-dependent operations can be defined using
**behavior**. A global mapping from parameters to behavior is in a dict called
``ak.behavior``:

.. code-block:: python

    >>> import awkward1 as ak
    >>> ak.behavior

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
    two = ak.Array([[{"x": 0.9, "y": 1}, {"x": 2, "y": 2.2}, {"x": 2.9, "y": 3}],
                    [],
                    [{"x": 3.9, "y": 4}, {"x": 5, "y": 5.5}],
                    [{"x": 5.9, "y": 6}],
                    [{"x": 6.9, "y": 7}, {"x": 8, "y": 8.8}, {"x": 8.9, "y": 9}]],
                   with_name="point")

The name appears in the way the type is presented as a string (a departure from
`Datashape notation <https://datashape.readthedocs.io/>`__):

.. code-block:: python

    >>> ak.type(one)
    5 * var * point["x": int64, "y": float64]

and it may be accessed as the ``"__record__"`` property, through the
`ak.Array.layout <_auto/ak.Array.html#ak-array-layout>`_:

.. code-block:: python

    >>> one.layout
    <ListOffsetArray64>
        <offsets><Index64 i="[0 3 3 5 6 9]" offset="0" length="6" at="0x55eb8da65e90"/></offsets>
        <content><RecordArray>
            <parameters>
                <param key="__record__">"point"</param>
            </parameters>
            <field index="0" key="x">
                <NumpyArray format="l" shape="9" data="1 2 3 4 5 6 7 8 9" at="0x55eb8da5c2e0"/>
            </field>
            <field index="1" key="y">
                <NumpyArray format="d" shape="9" data="1.1 2.2 3.3 4.4 5.5 6.6 7.7 8.8 9.9" at="0x55eb8da63370"/>
            </field>
        </RecordArray></content>
    </ListOffsetArray64>
    >>> one.layout.content.parameters
    {'__record__': 'point'}

We have to dig into the layout's content because the ``"__record__"`` parameter
is set on the :doc:`ak.layout.RecordArray`, which is buried inside of a
:doc:`ak.layout.ListOffsetArray`.

Alternatively, we can navigate to a single :doc:`_auto/ak.Record` first:

.. code-block:: python

    >>> one[0, 0]
    <Record {x: 1, y: 1.1} type='point["x": int64, "y": float64]'>
    >>> one[0, 0].layout.parameters
    {'__record__': 'point'}

Adding behavior to records
==========================

Suppose we want the points in the above example to be able to calculate
distances to other points. We can do this by creating a subclass of
:doc:`_auto/ak.Record` that has the new methods and associating it with
the ``"__record__"`` name.

.. code-block:: python

    class Point(ak.Record):
        def distance(self, other):
            return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    ak.behavior["point"] = Point

Now ``one[0, 0]`` is instantiated as a ``Point``, rather than a ``Record``,

.. code-block:: python

    >>> one[0, 0]
    <Point {x: 1, y: 1.1} type='point["x": int64, "y": float64]'>

and it has the ``distance`` method.

.. code-block:: python

    >>> for xs, ys in zip(one, two):
    ...     for x, y in zip(xs, ys):
    ...         print(x.distance(y))
    0.14142135623730953
    0.0
    0.31622776601683783
    0.4123105625617664
    0.0
    0.6082762530298216
    0.7071067811865477
    0.0
    0.905538513813742

Looping over data in Python is inconvenient and slow; we want to compute
quantities like this with array-at-a-time methods, but ``distance`` is
bound to a :doc:`_auto/ak.Record`, not an :doc:`_auto/ak.Array` of records.

.. code-block:: python

    >>> one.distance(two)
    AttributeError: no field named 'distance'

To add ``distance`` as a method on arrays of points, create a subclass of
:doc:`_auto/ak.Array` and attach that as ``ak.behavior[".", "point"]`` for
"array of points."

.. code-block:: python

    class PointArray(ak.Array):
        def distance(self, other):
            return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    ak.behavior[".", "point"] = PointArray

Now ``one[0]`` is a ``PointArray`` and can compute ``distance`` on arrays at a
time. Thanks to NumPy's
`universal function <https://docs.scipy.org/doc/numpy/reference/ufuncs.html>`__
(ufunc) syntax, the expression is the same (and could perhaps be implemented
once and used by both ``Point`` and ``PointArray``).

.. code-block:: python

    >>> one[0]
    <PointArray [{x: 1, y: 1.1}, ... {x: 3, y: 3.3}] type='3 * point["x": int64, "y"...'>
    >>> one[0].distance(two[0])
    <Array [0.141, 0, 0.316] type='3 * float64'>

But ``one`` itself is an ``Array`` of ``PointArrays``, and does not apply.

.. code-block:: python

    >>> one
    <Array [[{x: 1, y: 1.1}, ... x: 9, y: 9.9}]] type='5 * var * point["x": int64, "...'>
    >>> one.distance(two)
    AttributeError: no field named 'distance'

We can make the assignment work at all levels of list-depth by using a ``"*"``
instead of a ``"."``.

.. code-block:: python

    ak.behavior["*", "point"] = PointArray

One last caveat: our ``one`` array was created *before* this behavior was
assigned, so it needs to be recreated to be a member of the new class. The
normal :doc:`_auto/ak.Array` constructor is sufficient for this. This is only
an issue if you're working interactively (but something to think about when
debugging!).

.. code-block:: python

    >>> one = ak.Array(one)
    >>> two = ak.Array(two)

Now it works, and again we're taking advantage of the fact that the expression
for ``distance`` based on ufuncs works equally well on Awkward Arrays.

.. code-block:: python

    >>> one
    <PointArray [[{x: 1, y: 1.1}, ... x: 9, y: 9.9}]] type='5 * var * point["x": int...'>
    >>> one.distance(two)
    <Array [[0.141, 0, 0.316, ... 0.707, 0, 0.906]] type='5 * var * float64'>

**In most cases, you want to apply array-of-records for all levels of list-depth:** use ``ak.behavior["*", record_name]``.

Overriding NumPy ufuncs and binary operators
============================================

The :doc:`_auto/ak.Array` class overrides Python's binary operators with the
equivalent ufuncs, so ``__eq__`` actually calls ``np.equal``, for instance.
This is also true of other basic functions, like ``__abs__`` for overriding
``abs`` with ``np.absolute``. Each ufunc is then passed down to the leaves
(deepest sub-elements) of an Awkward data structure.

For example,

.. code-block:: python

    >>> ak.to_list(one == two)
    [[{'x': False, 'y': False}, {'x': True, 'y': True}, {'x': False, 'y': False}],
     [],
     [{'x': False, 'y': False}, {'x': True, 'y': True}],
     [{'x': False, 'y': False}],
     [{'x': False, 'y': False}, {'x': True, 'y': True}, {'x': False, 'y': False}]]

We might want to take an object-oriented view in which the ``==`` operation
applies to points, rather than their components. If we try to do it by adding
``__eq__`` as a method on ``PointArray``, it would work if the ``PointArray``
is the top of the data structure, but not if it's nested within another
structure.

Instead, we should override ``np.equal`` itself. Custom ufunc overrides are
checked at every step in broadcasting, so the override would be applied if
point objects are discovered at any level.

.. code-block:: python

    def point_equal(left, right):
        return np.logical_and(left.x == right.x, left.y == right.y)

    ak.behavior[np.equal, "point", "point"] = point_equal

The above should be read as "override ``np.equal`` for cases in which both
arguments are ``"point"``."

.. code-block:: python

    >>> ak.to_list(one == two)
    [[False, True, False], [], [False, True], [False], [False, True, False]]

Similarly for overriding ``abs``

.. code-block:: python

    >>> def point_abs(point):
    ...     return np.sqrt(point.x**2 + point.y**2)
    ... 
    >>> ak.behavior[np.absolute, "point"] = point_abs
    >>> ak.to_list(abs(one))
    [[1.4866068747318506, 2.973213749463701, 4.459820624195552],
     [],
     [5.946427498927402, 7.433034373659253],
     [8.919641248391104],
     [10.406248123122953, 11.892854997854805, 13.379461872586655]]

and all other ufuncs.

Adding behavior to arrays
=========================

Less often, you may want to add behavior to an array that does not contain
records. A good example of that is strings: strings are not a special data type
in Awkward Array as they are in many other libraries, they are a behavior
overlaid on arrays.

There are four predefined string behaviors:

   * :doc:`_auto/ak.behaviors.string.CharBehavior`: an array of UTF-8 encoded characters;
   * :doc:`_auto/ak.behaviors.string.ByteBehavior`: an array of unencoded characters;
   * :doc:`_auto/ak.behaviors.string.StringBehavior`: an array of variable-length UTF-8 encoded strings;
   * :doc:`_auto/ak.behaviors.string.ByteStringBehavior`: an array of variable-length unencoded bytestrings.

The character behaviors add a screen representation (``__str__`` and
``__repr__``) while the string behaviors additionally override equality
(``__eq__`` and ``__ne__``) to compare strings as units, rather than letting
the ``np.equal`` ufunc descend into bytes.

.. code-block:: python

    >>> ak.Array(["one", "two", "three"]) == ak.Array(["1", "TWO", "three"])
    <Array [False, False, True] type='3 * bool'>

HERE

Overriding behavior in Numba
============================
