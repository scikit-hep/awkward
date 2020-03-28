# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import re
import keyword
try:
    from collections.abc import Sequence
    from collections.abc import Iterable
except:
    from collections import Sequence
    from collections import Iterable

import numpy

import awkward1._connect._numpy
import awkward1._connect._pandas
import awkward1.layout
import awkward1.operations.convert

_dir_pattern = re.compile(r"^[a-zA-Z_]\w*$")

class Array(awkward1._connect._numpy.NDArrayOperatorsMixin,
            awkward1._connect._pandas.PandasMixin, Sequence):
    """
    Args:
        data (#ak.layout.Content, #ak.Array, np.ndarray, str, or iterable):
            Data to wrap or convert into an array.
            If a NumPy array, the regularity of its dimensions is preserved
            and the data are viewed, not copied.
            If a string, the data are assumed to be JSON.
            If an iterable, calls #ak.fromiter, which assumes all dimensions
            have irregular lengths.
        behavior (None or dict): Custom #ak.behavior for this Array only.
        withname (None or str): Gives tuples and records a name that can be
            used to override their behavior (see below).
        checkvalid (bool): If True, verify that the #layout is valid.

    High-level array that can contain data of any type.

    For most users, this is the only class in Awkward Array that matters: it
    is the entry point for data analysis with an emphasis on usability. It
    intentionally has a minimum of methods, preferring standalone functions
    like

        ak.cross([left, right])

    instead of bound methods like

        left.cross(right)

    because its namespace is valuable for domain-specific parameters and
    functionality. For example, with

        vectors = ak.Array([{"x": 0.1, "y": 1.0, "z": 10.0},
                            {"x": 0.2, "y": 2.0, "z": 20.0},
                            {"x": 0.3, "y": 3.0, "z": 30.0}])

    we want to access fields `x`, `y`, `z` as attributes

        >>> vectors.x
        <Array [0.1, 0.2, 0.3] type='3 * float64'>
        >>> vectors.y
        <Array [1, 2, 3] type='3 * float64'>
        >>> vectors.z
        <Array [10, 20, 30] type='3 * float64'>

    Additionally, we might want to add functionality,

        class Vec3Array(ak.Array):
            def cross(self, other):
                "Computes the cross-product of 3D vectors."
                x = self.y*other.z - self.z*other.y
                y = self.z*other.x - self.x*other.z
                z = self.x*other.y - self.y*other.x
                return ak.zip({"x": x, "y": y, "z": z}, withname="vec3")

        # Arrays of vec3 use subclass Vec3Array instead of ak.Array.
        ak.behavior["*", "vec3"] = Vec3Array

        # Records with name "vec3" are presented as having type "vec3".
        ak.behavior["__typestr__", "vec3"] = "vec3"

        vectors = ak.Array([{"x": 0.1, "y": 1.0, "z": 10.0},
                            {"x": 0.2, "y": 2.0, "z": 20.0},
                            {"x": 0.3, "y": 3.0, "z": 30.0}],
                           withname="vec3")
        more_vectors = ak.Array([{"x": 10.0, "y": 1.0, "z": 0.1},
                                 {"x": 20.0, "y": 2.0, "z": 0.2},
                                 {"x": 30.0, "y": 3.0, "z": 0.3}],
                                withname="vec3")

    Now the record types are presented as "vec3" and the Array has class
    `Vec3Array`, so it has a `cross` method.

        >>> vectors
        <Array [{x: 0.1, y: 1, z: 10, ... y: 3, z: 30}] type='3 * vec3'>
        >>> more_vectors
        <Array [{x: 10, y: 1, z: 0.1, ... z: 0.3}] type='3 * vec3'>
        >>> type(vectors)
        <class '__main__.Vec3Array'>
        >>> type(more_vectors)
        <class '__main__.Vec3Array'>

        >>> vectors.cross(more_vectors)
        <Array [{x: -9.9, y: 100, ... z: -89.1}] type='3 * vec3'>

    If the #ak.cross function were a method of this Array class, then it would
    conflict with applications where we might want `array.cross` to mean
    something else.

    Arrays can be used in [Numba](http://numba.pydata.org/): they can be
    passed as arguments to a Numba-compiled function or returned as return
    values. The only limitation is that Awkward Arrays cannot be *created*
    inside the Numba-compiled function; to make outputs, consider
    #ak.ArrayBuilder.

    See also #ak.Record.
    """

    def __init__(self, data, behavior=None, withname=None, checkvalid=False):
        if isinstance(data, awkward1.layout.Content):
            layout = data
        elif isinstance(data, Array):
            layout = data.layout
        elif isinstance(data, numpy.ndarray):
            layout = awkward1.operations.convert.fromnumpy(data,
                                                           highlevel=False)
        elif isinstance(data, str):
            layout = awkward1.operations.convert.fromjson(data,
                                                          highlevel=False)
        elif isinstance(data, dict):
            raise TypeError(
                    "could not convert dict into an awkward1.Array; "
                    "try awkward1.Record")
        else:
            layout = awkward1.operations.convert.fromiter(data,
                                                          highlevel=False,
                                                          allowrecord=False)
        if not isinstance(layout, awkward1.layout.Content):
            raise TypeError("could not convert data into an awkward1.Array")

        if withname is not None:
            layout = awkward1.operations.structure.withname(layout,
                                                            withname,
                                                            highlevel=False)

        if self.__class__ is Array:
            self.__class__ = awkward1._util.arrayclass(layout, behavior)

        self.layout = layout
        self.behavior = behavior
        if checkvalid:
            awkward1.operations.describe.validityerror(self, exception=True)

    @property
    def layout(self):
        """
        The composable #ak.layout.Content elements that determine how this
        Array is structured.

        This may be considered a "low-level" view, as it distinguishes between
        arrays that have the same logical meaning (i.e. same JSON output and
        high-level #type) but different

           * node types, such as #ak.layout.ListArray64 and
             #ak.layout.ListOffsetArray64,
           * integer type specialization, such as #ak.layout.ListArray64
             and #ak.layout.ListArray32,
           * or specific values, such as gaps in a #ak.layout.ListArray64.

        The #ak.layout.Content elements are fully composable, whereas an
        Array is not; the high-level Array is a single-layer "shell" around
        its layout.

        Layouts are rendered as XML instead of a nested list. For example,

            array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])

        is presented as

            <Array [[1.1, 2.2, 3.3], [], [4.4, 5.5]] type='3 * var * float64'>

        but `array.layout` is presented as

            <ListOffsetArray64>
                <offsets>
                    <Index64 i="[0 3 3 5]" offset="0" length="4" at="0x55a26df62590"/>
                </offsets>
                <content>
                    <NumpyArray format="d" shape="5" data="1.1 2.2 3.3 4.4 5.5" at="0x55a26e0c5f50"/>
                </content>
            </ListOffsetArray64>

        (with truncation for large arrays).
        """
        return self._layout

    @layout.setter
    def layout(self, layout):
        if isinstance(layout, awkward1.layout.Content):
            self._layout = layout
            self._numbaview = None
        else:
            raise TypeError(
                    "layout must be a subclass of awkward1.layout.Content")

    @property
    def behavior(self):
        """
        The `behavior` parameter passed into this Array's constructor.

           * If a dict, this `behavior` overrides the global #ak.behavior.
             Any keys in the global #ak.behavior but not this `behavior` are
             still valid, but any keys in both are overridden by this
             `behavior`. Keys with a None value are equivalent to missing keys,
             so this `behavior` can effectively remove keys from the
             global #ak.behavior.

           * If None, the Array defaults to the global #ak.behavior.

        See #ak.behavior for a list of recognized key patterns and their
        meanings.
        """
        return self._behavior

    @behavior.setter
    def behavior(self, behavior):
        if behavior is None or isinstance(behavior, dict):
            self._behavior = behavior
        else:
            raise TypeError("behavior must be None or a dict")

    @property
    def type(self):
        """
        The high-level type of this Array as #ak.types.Type objects.

        The high-level type ignores #layout differences like
        #ak.layout.ListArray64 versus #ak.layout.ListOffsetArray64, but
        not differences like "regular-sized lists" (i.e.
        #ak.layout.RegularArray) versus "variable-sized lists" (i.e.
        #ak.layout.ListArray64 and similar).

        Types are rendered as [Datashape](https://datashape.readthedocs.io/)
        strings, which makes the same distinctions.

        For example,

            ak.Array([[{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}],
                      [],
                      [{"x": 3.3, "y": [3, 3, 3]}]])

        has type

            3 * var * {"x": float64, "y": var * int64}

        but

            ak.Array(np.arange(2*3*5).reshape(2, 3, 5))

        has type

            2 * 3 * 5 * int64

        Some cases, like heterogeneous data, require [extensions beyond the
        Datashape specification](https://github.com/blaze/datashape/issues/237).
        For example,

            ak.Array([1, "two", [3, 3, 3]])

        has type

            3 * union[int64, string, var * int64]

        but "union" is not a Datashape type-constructor. (Its syntax is
        similar to existing type-constructors, so it's a plausible addition
        to the language.)
        """
        return awkward1.types.ArrayType(
                 self._layout.type(awkward1._util.typestrs(self._behavior)),
                 len(self._layout))

    def __len__(self):
        """
        The length of this Array, only counting the outermost structure.

        For example, the length of

            ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])

        is `3`, not `5`.
        """
        return len(self._layout)

    def __iter__(self):
        """
        Iterates over this Array in Python.

        Note that this is the *slowest* way to access data (even slower than
        native Python objects, like lists and dicts). Usually, you should
        express your problems in array-at-a-time operations.

        In other words, do this:

            >>> print(np.sqrt(ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])))
            [[1.05, 1.48, 1.82], [], [2.1, 2.35]]

        not this:

            >>> for outer in ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]):
            ...     for inner in outer:
            ...         print(np.sqrt(inner))
            ... 
            1.0488088481701516
            1.4832396974191326
            1.816590212458495
            2.0976176963403033
            2.345207879911715

        Iteration over Arrays exists so that they can be more easily inspected
        as Python objects.

        See also #ak.tolist.
        """
        for x in self._layout:
            yield awkward1._util.wrap(x, self._behavior)

    def __getitem__(self, where):
        """
        Args:
            where (many types supported; see below): Index of positions to
                select from this Array.

        Select items from the Array using an extension of NumPy's (already
        quite extensive) rules.

        All methods of selecting items described in
        [NumPy indexing](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html)
        are supported with one exception
        ([combining advanced and basic indexing](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#combining-advanced-and-basic-indexing)
        with basic indexes *between* two advanced indexes: the definition
        NumPy chose for the result does not have a generalization beyond
        rectilinear arrays).

        The `where` parameter can be any of the following or a tuple of
        the following.

           * **An integer** selects one element. Like Python/NumPy, it is
             zero-indexed: `0` is the first item, `1` is the second, etc.
             Negative indexes count from the end of the list: `-1` is the
             last, `-2` is the second-to-last, etc.
             Indexes beyond the size of the array, either because they're too
             large or because they're too negative, raise errors. In
             particular, some nested lists might contain a desired element
             while others don't; this would raise an error.
           * **A slice** (either a Python `slice` object or the
             `start:stop:step` syntax) selects a range of elements. The
             `start` and `stop` values are zero-indexed; `start` is inclusive
             and `stop` is exclusive, like Python/NumPy. Negative `step`
             values are allowed, but a `step` of `0` is an error. Slices
             beyond the size of the array are not errors but are truncated,
             like Python/NumPy.
           * **A string** selects a tuple or record field, even if its
             position in the tuple is to the left of the dimension where the
             tuple/record is defined. (See <<projection>> below.) This is
             similar to NumPy's
             [field access](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#field-access),
             except that strings are allowed in the same tuple with other
             slice types. While record fields have names, tuple fields are
             integer strings, such as `"0"`, `"1"`, `"2"` (always
             non-negative). Be careful to distinguish these from non-string
             integers.
           * **An iterable of strings** (not the top-level tuple) selects
             multiple tuple/record fields.
           * **An ellipsis** (either the Python `Ellipsis` object or the
             `...` syntax) skips as many dimensions as needed to put the
             rest of the slice items to the innermost dimensions.
           * **A np.newaxis** or its equivalent, None, does not select items
             but introduces a new regular dimension in the output with size
             `1`. This is a convenient way to explicitly choose a dimension
             for broadcasting.
           * **A boolean array** with the same length as the current dimension
             (or any iterable, other than the top-level tuple) selects elements
             corresponding to each True value in the array, dropping those
             that correspond to each False. The behavior is similar to
             NumPy's
             [compress](https://docs.scipy.org/doc/numpy/reference/generated/numpy.compress.html)
             function.
           * **An integer array** (or any iterable, other than the top-level
             tuple) selects elements like a single integer, but produces a
             regular dimension of as many as are desired. The array can have
             any length, any order, and it can have duplicates and incomplete
             coverage. The behavior is similar to NumPy's
             [take](https://docs.scipy.org/doc/numpy/reference/generated/numpy.take.html)
             function.
           * **An integer Array with missing (None) items** selects multiple
             values by index, as above, but None values are passed through
             to the output. This behavior matches pyarrow's
             [Array.take](https://arrow.apache.org/docs/python/generated/pyarrow.Array.html#pyarrow.Array.take)
             which also manages arrays with missing values. See
             <<option indexing>> below.
           * **An Array of nested lists**, ultimately containing booleans or
             integers and having the same lengths of lists at each level as
             the Array to which they're applied, selects by boolean or by
             integer at the deeply nested level. See <<nested indexing>> below.

        A tuple of the above applies each slice item to a dimension of the
        data, which can be very expressive. More than one flat boolean/integer
        array are "iterated as one" as described in the
        [NumPy documentation](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#integer-array-indexing).

        Projection
        **********

        The following `array`

            array = ak.Array([[{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}],
                              [{"x": 3.3, "y": [3, 3, 3]}],
                              [{"x": 0, "y": []}, {"x": 1.1, "y": [1, 1, 1]}]])

        has records inside of nested lists:

            >>> ak.typeof(array)
            3 * var * {"x": float64, "y": var * int64}

        In principle, one should select nested lists before record fields,

            >>> array[2, :, "x"]
            <Array [0, 1.1] type='2 * float64'>
            >>> array[::2, :, "x"]
            <Array [[1.1, 2.2], [0, 1.1]] type='2 * var * float64'>

        but it's also possible to select record fields first.

            >>> array["x"]
            <Array [[1.1, 2.2], [3.3], [0, 1.1]] type='3 * var * float64'>

        The string can "commute" to the left through integers and slices to
        get the same result as it would in its "natural" position.

            >>> array[2, :, "x"]
            <Array [0, 1.1] type='2 * float64'>
            >>> array[2, "x", :]
            <Array [0, 1.1] type='2 * float64'>
            >>> array["x", 2, :]
            <Array [0, 1.1] type='2 * float64'>

        The is analogous to selecting rows (integer indexes) before columns
        (string names) or columns before rows, except that the rows are
        more complex (like a Pandas
        [MultiIndex](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html)).
        This would be an expensive operation in a typical object-oriented
        environment, in which the records with fields `"x"` and `"y"` are
        akin to C structs, but for columnar Awkward Arrays, projecting
        through all records to produce an array of nested lists of `"x"`
        values just changes the metadata (no loop over data, and therefore
        fast).

        Thus, data analysts should think of records as fluid objects that
        can be easily projected apart and zipped back together with
        #ak.zip.

        Note, however, that while a column string can "commute" with row
        indexes to the left of its position in the tree, it can't commute
        to the right. For example, it's possible to use slices inside
        `"y"` because `"y"` is a list:

            >>> array[0, :, "y"]
            <Array [[1], [2, 2]] type='2 * var * int64'>
            >>> array[0, :, "y", 0]
            <Array [1, 2] type='2 * int64'>

        but it's not possible to move `"y"` to the right

            >>> array[0, :, 0, "y"]
            ValueError: in NumpyArray, too many dimensions in slice

        because the `array[0, :, 0, ...]` slice applies to both `"x"` and
        `"y"` before `"y"` is selected, and `"x"` is a one-dimensional
        NumpyArray that can't take more than its share of slices.

        Option indexing
        ***************

        NumPy arrays can be sliced by all of the above slice types except
        arrays with missing values and arrays with nested lists, both of
        which are inexpressible in NumPy. Missing values, represented by
        None in Python, are called option types (#ak.type.OptionType) in
        Awkward Array and can be used as a slice.

        For example, an `array` like

            array = ak.Array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])

        can be sliced with a boolean array

            >>> array[[False, False, False, False, True, False, True, False, True]]
            <Array [5.5, 7.7, 9.9] type='3 * float64'>

        or a boolean array containing None values:

            >>> array[[False, False, False, False, True, None, True, None, True]]
            <Array [5.5, None, 7.7, None, 9.9] type='5 * ?float64'>

        Similarly for arrays of integers and None:        

            >>> array[[0, 1, None, None, 7, 8]]
            <Array [1.1, 2.2, None, None, 8.8, 9.9] type='6 * ?float64'>

        This is the same behavior as pyarrow's
        [Array.take](https://arrow.apache.org/docs/python/generated/pyarrow.Array.html#pyarrow.Array.take),
        which establishes a convention for how to interpret slice arrays
        with option type:

            >>> import pyarrow as pa
            >>> array = pa.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
            >>> array.take(pa.array([0, 1, None, None, 7, 8]))
            <pyarrow.lib.DoubleArray object at 0x7efc7f060210>
            [
              1.1,
              2.2,
              null,
              null,
              8.8,
              9.9
            ]

        Nested indexing
        ***************

        Awkward Array's nested lists can be used as slices as well, as long
        as the type at the deepest level of nesting is boolean or integer.

        For example, the `array`

            array = ak.Array([[[0.0, 1.1, 2.2], [], [3.3, 4.4]], [], [[5.5]]])

        can be sliced at the top level with one-dimensional arrays:

            >>> array[[False, True, True]]
            <Array [[], [[5.5]]] type='2 * var * var * float64'>
            >>> array[[1, 2]]
            <Array [[], [[5.5]]] type='2 * var * var * float64'>

        with singly nested lists:

            >>> array[[[False, True, True], [], [True]]]
            <Array [[[], [3.3, 4.4]], [], [[5.5]]] type='3 * var * var * float64'>
            >>> array[[[1, 2], [], [0]]]
            <Array [[[], [3.3, 4.4]], [], [[5.5]]] type='3 * var * var * float64'>

        and with doubly nested lists:

            >>> array[[[[False, True, False], [], [True, False]], [], [[False]]]]
            <Array [[[1.1], [], [3.3]], [], [[]]] type='3 * var * var * float64'>
            >>> array[[[[1], [], [0]], [], [[]]]]
            <Array [[[1.1], [], [3.3]], [], [[]]] type='3 * var * var * float64'>

        The key thing is that the nested slice has the same number of elements
        as the array it's slicing at every level of nesting that it reproduces.
        This is similar to the requirement that boolean arrays have the same
        length as the array they're filtering.

        This kind of slicing is useful because NumPy's
        [universal functions](https://docs.scipy.org/doc/numpy/reference/ufuncs.html)
        produce arrays with the same structure as the original array, which
        can then be used as filters.

            >>> print((array * 10) % 2 == 1)
            [[[False, True, False], [], [True, False]], [], [[True]]]
            >>> print(array[(array * 10) % 2 == 1])
            [[[1.1], [], [3.3]], [], [[5.5]]]

        Functions whose names start with "arg" return index positions, which
        can be used with the integer form.

            >>> print(np.argmax(array, axis=-1))
            [[2, None, 1], [], [0]]
            >>> print(array[np.argmax(array, axis=-1)])
            [[[3.3, 4.4], None, []], [], [[5.5]]]

        Here, the `np.argmax` returns the integer position of the maximum
        element or None for empty arrays. It's a nice example of
        <<option indexing>> with <<nested indexing>>.
        """
        return awkward1._util.wrap(self._layout[where], self._behavior)

    def __setitem__(self, where, what):
        """
        Args:
            where (str): Field name to add to records in the array.
            what (#ak.Array): Array to add as the new field.

        Unlike #__getitem__, which allows a wide variety of slice types,
        only single field-slicing is supported for assignment.
        (#ak.layout.Content arrays are immutable; field assignment replaces
        the #layout with an array that has the new field using #ak.withfield.)

        If necessary, the new field will be broadcasted to fit the array.
        For example, given an `array` like

            ak.Array([[{"x": 1.1}, {"x": 2.2}, {"x": 3.3}], [], [{"x": 4.4}, {"x": 5.5}]])

        which has three elements with nested data in each, assigning

            >>> array["y"] = [100, 200, 300]

        will result in

            >>> ak.tolist(array)
            [[{'x': 1.1, 'y': 100}, {'x': 2.2, 'y': 100}, {'x': 3.3, 'y': 100}],
             [],
             [{'x': 4.4, 'y': 300}, {'x': 5.5, 'y': 300}]]

        because the `100` in `what[0]` is broadcasted to all three nested
        elements of `array[0]`, the `200` in `what[1]` is broadcasted to the
        empty list `array[1]`, and the `300` in `what[2]` is broadcasted to
        both elements of `array[2]`.
        """
        if not isinstance(where, str):
            raise ValueError(
                    "only fields may be assigned in-place (by field name)")
        self._layout = awkward1.operations.structure.withfield(self._layout,
                                                               what,
                                                               where).layout
        self._numbaview = None

    def __getattr__(self, where):
        """
        Whenever possible, fields can be accessed as attributes.

        For example, the fields of an `array` like

            ak.Array([[{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}, {"x": 3.3, "y": [3, 3, 3]}],
                      [],
                      [{"x": 4.4, "y": [4, 4, 4, 4]}, {"x": 5.5, "y": [5, 5, 5, 5, 5]}]])

        can be accessed as

            >>> array.x
            <Array [[1.1, 2.2, 3.3], [], [4.4, 5.5]] type='3 * var * float64'>
            >>> array.y
            <Array [[[1], [2, 2], ... [5, 5, 5, 5, 5]]] type='3 * var * var * int64'>

        which are equivalent to `array["x"]` and `array["y"]`. (See
        <<projection>>.)

        Fields can't be accessed as attributes when

           * #ak.Array methods or properties take precedence,
           * a domain-specific behavior has methods or properties that take
             precedence, or
           * the field name is not a valid Python identifier or is a Python
             keyword.
        """
        if where in dir(type(self)):
            return super(Array, self).__getattribute__(where)
        else:
            if where in self._layout.keys():
                try:
                    return self[where]
                except Exception as err:
                    raise AttributeError(
                            "while trying to get field {0}, an exception "
                            "occurred:\n{1}: {2}".format(repr(where),
                                                         type(err),
                                                         str(err)))
            else:
                raise AttributeError("no field named {0}".format(repr(where)))

    def __dir__(self):
        """
        Lists all methods, properties, and field names (see #__getattr__)
        that can be accessed as attributes.
        """
        return sorted(set(dir(super(Array, self))
                          + [x for x in self._layout.keys()
                               if _dir_pattern.match(x) and
                                  not keyword.iskeyword(x)]))

    @property
    def slot0(self):
        """
        Equivalent to #__getitem__ with `"0"`, which selects slot `0` from
        all tuples.

        Record fields can be accessed from #__getitem__ with strings (see
        <<projection>>), but tuples only have slot positions, which are
        0-indexed integers. However, they must also be quoted as strings
        to avoid confusion with integers as array indexes. Sometimes, though,
        interleaving integers in strings and integers outside of strings
        can be confusing in analysis code.

        Record fields can also be accessed as attributes (with limitations),
        and the distinction between attributes (#__getattr__) and subscripts
        (#__getitem__) shows up more clearly in dense code. But integers would
        not be valid attribute names, so they're named #slot0 through #slot9.

        (Tuples with more than 10 slots are rare and can defer to
        #__getitem__.)
        """
        return self["0"]
    @property
    def slot1(self):
        """
        Equivalent to #__getitem__ with `"1"`. See #slot0.
        """
        return self["1"]
    @property
    def slot2(self):
        """
        Equivalent to #__getitem__ with `"2"`. See #slot0.
        """
        return self["2"]
    @property
    def slot3(self):
        """
        Equivalent to #__getitem__ with `"3"`. See #slot0.
        """
        return self["3"]
    @property
    def slot4(self):
        """
        Equivalent to #__getitem__ with `"4"`. See #slot0.
        """
        return self["4"]
    @property
    def slot5(self):
        """
        Equivalent to #__getitem__ with `"5"`. See #slot0.
        """
        return self["5"]
    @property
    def slot6(self):
        """
        Equivalent to #__getitem__ with `"6"`. See #slot0.
        """
        return self["6"]
    @property
    def slot7(self):
        """
        Equivalent to #__getitem__ with `"7"`. See #slot0.
        """
        return self["7"]
    @property
    def slot8(self):
        """
        Equivalent to #__getitem__ with `"8"`. See #slot0.
        """
        return self["8"]
    @property
    def slot9(self):
        """
        Equivalent to #__getitem__ with `"9"`. See #slot0.
        """
        return self["9"]

    def __str__(self, limit_value=85):
        """
        Args:
            limit_value (int): Maximum number of characters to use when
                presenting the Array as a string.

        Presents this Array as a string without type or `"<Array ...>"`.

        Large Arrays are truncated to the first few elements and the last
        few elements to fit within `limit_value` characters, using ellipsis
        to indicate the break. For example, an `array` like

            ak.Array([[1.1, 2.2, 3.3],
                      [],
                      [4.4, 5.5, 6.6],
                      [7.7, 8.8, 9.9, 10.0],
                      [],
                      [],
                      [],
                      [11.1, 12.2]])

        is shown as

            [[1.1, 2.2, 3.3], [], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9, ... [], [], [], [11.1, 12.2]]

        The algorithm does not split tokens; it will not show half a number
        (which can be very misleading), but it can lose structural elements
        like the `]` that closes `[7.7, 8.8, 9.9, 10.0]`.

        The algorithm also avoids reading data unnecessarily: most of the data
        in the ellipsis are not even read. This can be particularly important
        for datasets that contain #ak.layout.VirtualArray nodes that might
        be expensive to read.

        Note that the string also does not quote field names. An `array` like

            ak.Array([[{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}, {"x": 3.3, "y": [3, 3, 3]}],
                      [],
                      [{"x": 4.4, "y": [4, 4, 4, 4]}]])

        is presented as

            [[{x: 1.1, y: [1]}, {x: 2.2, y: [2, 2]}, ... [], [{x: 4.4, y: [4, 4, 4, 4]}]]

        Floating point numbers are presented in `.3g` format (3 digits using
        exponential notation if necessary).

        The string representation cannot be read as JSON or as an #ak.Array
        constructor.

        See #ak.tolist and #ak.tojson to convert whole Arrays into Python data
        or JSON strings without loss (except for #type).
        """
        return awkward1._util.minimally_touching_string(limit_value,
                                                        self._layout,
                                                        self._behavior)

    def __repr__(self, limit_value=40, limit_total=85):
        """
        Args:
            limit_value (int): Maximum number of characters to use when
                presenting the data of the Array.
            limit_total (int): Maximum number of characters to use for
                the whole string (should be larger than `limit_value`).

        Presents this Array as a string with its type and `"<Array ...>"`.

        See #__str__ for details of the string truncation algorithm.

        The #type is truncated as well, but showing only the left side
        of its string (the outermost data structures).
        """
        value = awkward1._util.minimally_touching_string(limit_value,
                                                         self._layout,
                                                         self._behavior)

        limit_type = limit_total - len(value) - len("<Array  type=>")
        type = repr(str(self.type))
        if len(type) > limit_type:
            type = type[:(limit_type - 4)] + "..." + type[-1]

        return "<Array {0} type={1}>".format(value, type)

    def __array__(self, *args, **kwargs):
        """
        Intercepts attempts to convert this Array into a NumPy array and
        either performs a zero-copy conversion or raises an error.

        This function is also called by the
        [np.asarray](https://docs.scipy.org/doc/numpy/reference/generated/numpy.asarray.html)
        family of functions, which have `copy=False` by default.

            >>> np.asarray(ak.Array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]))
            array([[1.1, 2.2, 3.3],
                   [4.4, 5.5, 6.6]])

        If the data are numerical and regular (nested lists have equal lengths
        in each dimension, as described by the #type), they can be losslessly
        converted to a NumPy array and this function returns without an error.

        Otherwise, the function raises an error. It does not create a NumPy
        array with dtype `"O"` for `np.object_` (see the
        [note on object_ type](https://docs.scipy.org/doc/numpy/reference/arrays.scalars.html#arrays-scalars-built-in))
        since silent conversions to dtype `"O"` arrays would not only be a
        significant performance hit, but would also break functionality, since
        nested lists in a NumPy `"O"` array are severed from the array and
        cannot be sliced as dimensions.

        Only exception: Pandas can generate NumPy `"O"` arrays to print
        Array fragments to the screen.
        """
        if awkward1._util.called_by_module("pandas"):
            try:
                return awkward1._connect._numpy.convert_to_array(self._layout,
                                                                 args,
                                                                 kwargs)
            except:
                out = numpy.empty(len(self._layout), dtype="O")
                for i, x in enumerate(self._layout):
                    out[i] = awkward1._util.wrap(x, self._behavior)
                return out
        else:
            return awkward1._connect._numpy.convert_to_array(self._layout,
                                                             args,
                                                             kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Intercepts attempts to pass this Array to a NumPy
        [universal functions](https://docs.scipy.org/doc/numpy/reference/ufuncs.html)
        (ufuncs) and passes it through the Array's structure.

        This method conforms to NumPy's
        [NEP 13](https://numpy.org/neps/nep-0013-ufunc-overrides.html)
        for overriding ufuncs, which has been
        [available since NumPy 1.13](https://numpy.org/devdocs/release/1.13.0-notes.html#array-ufunc-added)
        (and thus NumPy 1.13 is the minimum allowed version).

        When any ufunc is applied to an Awkward Array, it applies to the
        innermost level of structure and preserves the structure through the
        operation.

        For example, with an `array` like

            ak.Array([[{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}], [], [{"x": 2.2, "y": [2, 2]}]])

        applying `np.sqrt` would yield

            >>> print(np.sqrt(array))
            [[{x: 0, y: []}, {x: 1.05, y: [1]}], [], [{x: 1.48, y: [1.41, 1.41]}]]

        In addition, many unary and binary operators implicitly call ufuncs,
        such as `np.power` in

            >>> print(array**2)
            [[{x: 0, y: []}, {x: 1.21, y: [1]}], [], [{x: 4.84, y: [4, 4]}]]

        Third party libraries can create ufuncs, not just NumPy, so any library
        that "plays well" with the NumPy ecosystem can be used with Awkward
        Arrays:

            >>> import numba as nb
            >>> @nb.vectorize([nb.float64(nb.float64)])
            ... def sqr(x):
            ...     return x * x
            ... 
            >>> print(sqr(array))
            [[{x: 0, y: []}, {x: 1.21, y: [1]}], [], [{x: 4.84, y: [4, 4]}]]

        See also #__array_function__.
        """
        return awkward1._connect._numpy.array_ufunc(ufunc,
                                                    method,
                                                    inputs,
                                                    kwargs,
                                                    self._behavior)

    def __array_function__(self, func, types, args, kwargs):
        """
        Intercepts attempts to pass this Array to those NumPy functions other
        than universal functions that have an Awkward equivalent.

        This method conforms to NumPy's
        [NEP 18](https://numpy.org/neps/nep-0018-array-function-protocol.html)
        for overriding functions, which has been
        [available since NumPy 1.17](https://numpy.org/devdocs/release/1.17.0-notes.html#numpy-functions-now-always-support-overrides-with-array-function)
        (and
        [NumPy 1.16 with an experimental flag set](https://numpy.org/devdocs/release/1.16.0-notes.html#numpy-functions-now-support-overrides-with-array-function)).
        This is not crucial for Awkward Array to work correctly, as NumPy
        functions like np.concatenate can be manually replaced with
        #ak.concatenate for early versions of NumPy.
        
        See also #__array_ufunc__.
        """
        return awkward1._connect._numpy.array_function(func,
                                                       types,
                                                       args,
                                                       kwargs)

    @property
    def numbatype(self):
        """
        The type of this Array when it is used in Numba. It contains enough
        information to generate low-level code for accessing any element,
        down to the leaves.

        See [Numba documentation](https://numba.pydata.org/numba-doc/dev/reference/types.html)
        on types and signatures.
        """
        import numba
        import awkward1._connect._numba
        awkward1._connect._numba.register()
        if self._numbaview is None:
            self._numbaview = \
              awkward1._connect._numba.arrayview.ArrayView.fromarray(self)
        return numba.typeof(self._numbaview)

class Record(awkward1._connect._numpy.NDArrayOperatorsMixin):
    """
    Args:
        data (#ak.layout.Record, #ak.Record, str, or dict):
            Data to wrap or convert into a record.
            If a string, the data are assumed to be JSON.
            If a dict, calls #ak.fromiter, which assumes all inner
            dimensions have irregular lengths.
        behavior (None or dict): Custom #ak.behavior for this Record only.
        withname (None or str): Gives the record type a name that can be
            used to override its behavior (see below).
        checkvalid (bool): If True, verify that the #layout is valid.

    High-level record that can contain fields of any type.

    Most users won't be creating Records manually. This class primarily exists
    to be overridden in the same way as #ak.Array.

    Following a very similar example,

        class Vec3(ak.Record):
            def cross(self, other):
                "Computes the cross-product of 3D vectors."
                x = self.y*other.z - self.z*other.y
                y = self.z*other.x - self.x*other.z
                z = self.x*other.y - self.y*other.x
                return ak.Record({"x": x, "y": y, "z": z}, withname="vec3")

        # Records of vec3 use subclass Vec3 instead of ak.Record.
        ak.behavior["vec3"] = Vec3

        # Records with name "vec3" are presented as having type "vec3".
        ak.behavior["__typestr__", "vec3"] = "vec3"

        vectors = ak.Array([{"x": 0.1, "y": 1.0, "z": 30.0},
                            {"x": 0.2, "y": 2.0, "z": 20.0},
                            {"x": 0.3, "y": 3.0, "z": 10.0}],
                           withname="vec3")

    Now the record types are presented as "vec3" and the Record has class
    `Vec3`, so it has a `cross` method.

        >>> vectors[0]
        <Record {x: 0.1, y: 1, z: 30} type='vec3'>
        >>> type(vectors[0])
        <class '__main__.Vec3'>

        >>> vectors[0].cross(vectors[1])
        <Record {x: -40, y: 4, z: 0} type='vec3'>

    Be sure to distinguish between records, which subclass #ak.Record, and
    arrays, which subclass #ak.Array, even though the method implementations
    can be very similar because NumPy's
    [universal functions](https://docs.scipy.org/doc/numpy/reference/ufuncs.html)
    are equally usable on scalars as they are on arrays.

    Records can be used in [Numba](http://numba.pydata.org/): they can be
    passed as arguments to a Numba-compiled function or returned as return
    values. The only limitation is that they cannot be *created*
    inside the Numba-compiled function; to make outputs, consider
    #ak.ArrayBuilder.

    See also #ak.Array.
    """
    def __init__(self, data, behavior=None, withname=None, checkvalid=False):
        if isinstance(data, awkward1.layout.Record):
            layout = data
        elif isinstance(data, Record):
            layout = data.layout
        elif isinstance(data, str):
            layout = awkward1.operations.convert.fromjson(data,
                                                          highlevel=False)
        elif isinstance(data, dict):
            layout = awkward1.operations.convert.fromiter([data],
                                                          highlevel=False)[0]
        elif isinstance(data, Iterable):
            raise TypeError("could not convert non-dict into an "
                            "awkward1.Record; try awkward1.Array")
        else:
            layout = None
        if not isinstance(layout, awkward1.layout.Record):
            raise TypeError("could not convert data into an awkward1.Record")

        if self.__class__ is Record:
            self.__class__ = awkward1._util.recordclass(layout, behavior)

        if withname is not None:
            layout = awkward1.operations.structure.withname(layout,
                                                            withname,
                                                            highlevel=False)

        self.layout = layout
        self.behavior = behavior
        if checkvalid:
            awkward1.operations.describe.validityerror(self, exception=True)

    @property
    def layout(self):
        """
        The #ak.layout.Record that contains composable #ak.layout.Content
        elements to determine how the array is structured.

        See #ak.Array.layout for a more complete description.

        The #ak.layout.Record is not a subclass of #ak.layout.Content in
        Python (note: [Record](../_static/classawkward_1_1Record.html) *is* a
        subclass of [Content](../_static/classawkward_1_1Content.html) in
        C++!) and it is not composable with them: #ak.layout.Record contains
        one #ak.layout.RecordArray (which is a #ak.layout.Content), but
        #ak.layout.Content nodes cannot contain a #ak.layout.Record.

        A #ak.layout.Record is not an independent entity from its
        #ak.layout.RecordArray; it's really just a marker indicating which
        element to select. The XML representation reflects that:

            >>> vectors = ak.Array([{"x": 0.1, "y": 1.0, "z": 30.0},
            ...                     {"x": 0.2, "y": 2.0, "z": 20.0},
            ...                     {"x": 0.3, "y": 3.0, "z": 10.0}])

            >>> vectors[1].layout
            <Record at="1">
                <RecordArray>
                    <field index="0" key="x">
                        <NumpyArray format="d" shape="3" data="0.1 0.2 0.3" at="0x555660dfe7d0"/>
                    </field>
                    <field index="1" key="y">
                        <NumpyArray format="d" shape="3" data="1 2 3" at="0x555660df4180"/>
                    </field>
                    <field index="2" key="z">
                        <NumpyArray format="d" shape="3" data="30 20 10" at="0x555660df6190"/>
                    </field>
                </RecordArray>
            </Record>
        """
        return self._layout

    @layout.setter
    def layout(self, layout):
        if isinstance(layout, awkward1.layout.Record):
            self._layout = layout
            self._numbaview = None
        else:
            raise TypeError(
                    "layout must be a subclass of awkward1.layout.Record")

    @property
    def behavior(self):
        """
        The `behavior` parameter passed into this Record's constructor.

           * If a dict, this `behavior` overrides the global #ak.behavior.
             Any keys in the global #ak.behavior but not this `behavior` are
             still valid, but any keys in both are overridden by this
             `behavior`. Keys with a None value are equivalent to missing keys,
             so this `behavior` can effectively remove keys from the
             global #ak.behavior.

           * If None, the Record defaults to the global #ak.behavior.

        See #ak.behavior for a list of recognized key patterns and their
        meanings.
        """
        return self._behavior

    @behavior.setter
    def behavior(self, behavior):
        if behavior is None or isinstance(behavior, dict):
            self._behavior = behavior
        else:
            raise TypeError("behavior must be None or a dict")

    @property
    def type(self):
        """
        The high-level type of this Record as #ak.types.Type objects.

        See #ak.Array.type for a more complete description.
        """
        return self._layout.type(awkward1._util.typestrs(self._behavior))

    def __getitem__(self, where):
        """
        Args:
            where (many types supported; see below): Index of positions to
                select from this Record.

        Select items from the Record using an extension of NumPy's (already
        quite extensive) rules.

        See #ak.Array.__getitem__ for a more complete description. Since
        this is a record, the first item in the slice tuple must be a
        string, selecting a field.

        For example, with a `record` like

            ak.Record({"x": 3.3, "y": [1, 2, 3]})

        we can select

            >>> record["x"]
            3.3
            >>> record["y"]
            <Array [1, 2, 3] type='3 * int64'>
            >>> record["y", 1]
            2
        """
        return awkward1._util.wrap(self._layout[where], self._behavior)

    def __setitem__(self, where, what):
        """
        Args:
            where (str): Field name to add data to the record.
            what: Data to add as the new field.

        For example:

            >>> record = ak.Record({"x": 3.3})
            >>> record["y"] = 4
            >>> record["z"] = {"another": "record"}
            >>> print(record)
            {x: 3.3, y: 4, z: {another: 'record'}}
        """
        if not isinstance(where, str):
            raise ValueError(
                    "only fields may be assigned in-place (by field name)")
        self._layout = awkward1.operations.structure.withfield(self._layout,
                                                               what,
                                                               where).layout
        self._numbaview = None

    def __getattr__(self, where):
        """
        Whenever possible, fields can be accessed as attributes.

        For example, the fields of an `record` like

            ak.Record({"x": 1.1, "y": [2, 2], "z": "three"})

        can be accessed as

            >>> record.x
            1.1
            >>> record.y
            <Array [2, 2] type='2 * int64'>
            >>> record.z
            'three'

        which are equivalent to `record["x"]`, `record["y"]`, and
        `record["z"]`.

        Fields can't be accessed as attributes when

           * #ak.Record methods or properties take precedence,
           * a domain-specific behavior has methods or properties that take
             precedence, or
           * the field name is not a valid Python identifier or is a Python
             keyword.
        """
        if where in dir(type(self)):
            return super(Record, self).__getattribute__(where)
        else:
            if where in self._layout.keys():
                try:
                    return self[where]
                except Exception as err:
                    raise AttributeError(
                            "while trying to get field {0}, an exception "
                            "occurred:\n{1}: {2}".format(repr(where),
                                                         type(err),
                                                         str(err)))
            else:
                raise AttributeError("no field named {0}".format(repr(where)))

    def __dir__(self):
        """
        Lists all methods, properties, and field names (see #__getattr__)
        that can be accessed as attributes.
        """
        return sorted(set(dir(super(Array, self))
                          + [x for x in self._layout.keys()
                               if _dir_pattern.match(x) and
                               not keyword.iskeyword(x)]))

    @property
    def slot0(self):
        """
        Equivalent to #__getitem__ with `"0"`, which selects slot `0` from
        the Record as a tuple.

        See #ak.Array.slot0 for a more complete description.
        """
        return self["0"]
    @property
    def slot1(self):
        """
        Equivalent to #__getitem__ with `"1"`, which selects slot `1` from
        the Record as a tuple.

        See #ak.Array.slot0 for a more complete description.
        """
        return self["1"]
    @property
    def slot2(self):
        """
        Equivalent to #__getitem__ with `"2"`, which selects slot `2` from
        the Record as a tuple.

        See #ak.Array.slot0 for a more complete description.
        """
        return self["2"]
    @property
    def slot3(self):
        """
        Equivalent to #__getitem__ with `"3"`, which selects slot `3` from
        the Record as a tuple.

        See #ak.Array.slot0 for a more complete description.
        """
        return self["3"]
    @property
    def slot4(self):
        """
        Equivalent to #__getitem__ with `"4"`, which selects slot `4` from
        the Record as a tuple.

        See #ak.Array.slot0 for a more complete description.
        """
        return self["4"]
    @property
    def slot5(self):
        """
        Equivalent to #__getitem__ with `"5"`, which selects slot `5` from
        the Record as a tuple.

        See #ak.Array.slot0 for a more complete description.
        """
        return self["5"]
    @property
    def slot6(self):
        """
        Equivalent to #__getitem__ with `"6"`, which selects slot `6` from
        the Record as a tuple.

        See #ak.Array.slot0 for a more complete description.
        """
        return self["6"]
    @property
    def slot7(self):
        """
        Equivalent to #__getitem__ with `"7"`, which selects slot `7` from
        the Record as a tuple.

        See #ak.Array.slot0 for a more complete description.
        """
        return self["7"]
    @property
    def slot8(self):
        """
        Equivalent to #__getitem__ with `"8"`, which selects slot `8` from
        the Record as a tuple.

        See #ak.Array.slot0 for a more complete description.
        """
        return self["8"]
    @property
    def slot9(self):
        """
        Equivalent to #__getitem__ with `"9"`, which selects slot `9` from
        the Record as a tuple.

        See #ak.Array.slot0 for a more complete description.
        """
        return self["9"]

    def __str__(self, limit_value=85):
        """
        Args:
            limit_value (int): Maximum number of characters to use when
                presenting the Record as a string.

        Presents this Record as a string without type or `"<Record ...>"`.

        See #ak.Array.__str__ for a more complete description.
        """
        return awkward1._util.minimally_touching_string(limit_value + 2,
                                                        self._layout,
                                                        self._behavior)[1:-1]

    def __repr__(self, limit_value=40, limit_total=85):
        """
        Args:
            limit_value (int): Maximum number of characters to use when
                presenting the data of the Record.
            limit_total (int): Maximum number of characters to use for
                the whole string (should be larger than `limit_value`).

        Presents this Record as a string with its type and `"<Record ...>"`.

        See #ak.Array.__repr__ for a more complete description.
        """
        value = awkward1._util.minimally_touching_string(limit_value + 2,
                                                         self._layout,
                                                         self._behavior)[1:-1]

        limit_type = limit_total - len(value) - len("<Record  type=>")
        type = repr(str(self.type))
        if len(type) > limit_type:
            type = type[:(limit_type - 4)] + "..." + type[-1]

        return "<Record {0} type={1}>".format(value, type)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Intercepts attempts to pass this Record to a NumPy
        [universal functions](https://docs.scipy.org/doc/numpy/reference/ufuncs.html)
        (ufuncs) and passes it through the Record's structure.

        This method conforms to NumPy's
        [NEP 13](https://numpy.org/neps/nep-0013-ufunc-overrides.html)
        for overriding ufuncs, which has been
        [available since NumPy 1.13](https://numpy.org/devdocs/release/1.13.0-notes.html#array-ufunc-added)
        (and thus NumPy 1.13 is the minimum allowed version).

        See #ak.Array.__array_ufunc__ for a more complete description.
        """
        return awkward1._connect._numpy.array_ufunc(ufunc,
                                                    method,
                                                    inputs,
                                                    kwargs,
                                                    self._behavior)

    @property
    def numbatype(self):
        """
        The type of this Record when it is used in Numba. It contains enough
        information to generate low-level code for accessing any element,
        down to the leaves.

        See [Numba documentation](https://numba.pydata.org/numba-doc/dev/reference/types.html)
        on types and signatures.
        """
        import numba
        import awkward1._connect._numba
        awkward1._connect._numba.register()
        if self._numbaview is None:
            self._numbaview = \
              awkward1._connect._numba.arrayview.RecordView.fromrecord(self)
        return numba.typeof(self._numbaview)

class ArrayBuilder(Sequence):
    """
    Args:
        behavior (None or dict): Custom #ak.behavior for arrays built by
            this ArrayBuilder.

    General tool for building arrays of nested data structures from a sequence
    of commands. Most data types can be constructed by calling commands in the
    right order, similar to printing tokens to construct JSON output.

    To illustrate how this works, consider the following example.

        b = ak.ArrayBuilder()

        # fill commands  # as JSON   # current array type
        ##########################################################################################
        b.beginlist()    # [         # 0 * var * unknown     (initially, the type is unknown)
        b.integer(1)     #   1,      # 0 * var * int64
        b.integer(2)     #   2,      # 0 * var * int64
        b.real(3)        #   3.0     # 0 * var * float64     (all the integers have become floats)
        b.endlist()      # ],        # 1 * var * float64
        b.beginlist()    # [         # 1 * var * float64
        b.endlist()      # ],        # 2 * var * float64
        b.beginlist()    # [         # 2 * var * float64
        b.integer(4)     #   4,      # 2 * var * float64
        b.null()         #   null,   # 2 * var * ?float64    (now the floats are nullable)
        b.integer(5)     #   5       # 2 * var * ?float64
        b.endlist()      # ],        # 3 * var * ?float64
        b.beginlist()    # [         # 3 * var * ?float64
        b.beginrecord()  #   {       # 3 * var * ?union[float64, {}]
        b.field("x")     #     "x":  # 3 * var * ?union[float64, {"x": unknown}]
        b.integer(1)     #      1,   # 3 * var * ?union[float64, {"x": int64}]
        b.field("y")     #      "y": # 3 * var * ?union[float64, {"x": int64, "y": unknown}]
        b.beginlist()    #      [    # 3 * var * ?union[float64, {"x": int64, "y": var * unknown}]
        b.integer(2)     #        2, # 3 * var * ?union[float64, {"x": int64, "y": var * int64}]
        b.integer(3)     #        3  # 3 * var * ?union[float64, {"x": int64, "y": var * int64}]
        b.endlist()      #      ]    # 3 * var * ?union[float64, {"x": int64, "y": var * int64}]
        b.endrecord()    #   }       # 3 * var * ?union[float64, {"x": int64, "y": var * int64}]
        b.endlist()      # ]         # 4 * var * ?union[float64, {"x": int64, "y": var * int64}]

    To get an array, we take a #snapshot of the ArrayBuilder's current state.

        >>> ak.tolist(b.snapshot())
        [[1.0, 2.0, 3.0], [], [4.0, None, 5.0], [{'x': 1, 'y': [2, 3]}]]

    The full set of filling commands is the following.

       * #null: appends a None value.
       * #boolean: appends True or False.
       * #integer: appends an integer.
       * #real: appends a floating-point value.
       * #bytestring: appends an unencoded string (raw bytes).
       * #string: appends a UTF-8 encoded string.
       * #beginlist: begins filling a list; must be closed with #endlist.
       * #endlist: ends a list.
       * #begintuple: begins filling a tuple; must be closed with #endtuple.
       * #index: selects a tuple slot to fill; must be followed by a command
         that actually fills that slot.
       * #endtuple: ends a tuple.
       * #beginrecord: begins filling a record; must be closed with #endrecord.
       * #field: selects a record field to fill; must be followed by a command
         that actually fills that field.
       * #endrecord: ends a record.
       * #append: generic method for filling #null, #boolean, #integer, #real,
         #bytestring, #string, #ak.Array, #ak.Record, or arbitrary Python data.
         When filling from #ak.Array or #ak.Record, the output holds references
         to the original data, rather than copying.
       * #extend: appends all the items from an #ak.Array (by reference).
       * #list: context manager for #beginlist and #endlist.
       * #tuple: context manager for #begintuple and #endtuple.
       * #record: context manager for #beginrecord and #endrecord.

    ArrayBuilders can be used in [Numba](http://numba.pydata.org/): they can
    be passed as arguments to a Numba-compiled function or returned as return
    values. (Since ArrayBuilder works by accumulating side-effects, it's not
    strictly necessary to return the object.)

    The primary limitation is that ArrayBuilders cannot be *created* and
    #snapshot cannot be called inside the Numba-compiled function. Awkward
    Array uses Numba as a transformer: #ak.Array and an empty #ak.ArrayBuilder
    go in and a filled #ak.ArrayBuilder is the result; #snapshot can be called
    outside of the compiled function.

    Also, context managers (Python's `with` statement) are not supported in
    Numba yet, so the #list, #tuple, and #record methods are not available
    in Numba-compiled functions.

    Here is an example of filling an ArrayBuilder in Numba, which makes a
    tree of dynamic depth.

        >>> import numba as nb
        >>> @nb.njit
        ... def deepnesting(builder, probability):
        ...     if np.random.uniform(0, 1) > probability:
        ...         builder.append(np.random.normal())
        ...     else:
        ...         builder.beginlist()
        ...         for i in range(np.random.poisson(3)):
        ...             deepnesting(builder, probability**2)
        ...         builder.endlist()
        ... 
        >>> builder = ak.ArrayBuilder()
        >>> deepnesting(builder, 0.9)
        >>> builder.snapshot()
        <Array [... 1.23, -0.498, 0.272], -0.0519]]]] type='1 * var * var * union[var * ...'>
        >>> ak.tolist(builder)
        [[[[2.052949634260401, 0.9522057655747124], [[[0.2560810133948006], 1.8668954120287653, 0.8933700720920406, 0.31709173110067773], 0.38515995466456676, -1.6259655150460695, [[0.18211022402412927], 0.46592679548320143, 0.39275072293709223], [-0.572569956850481, 1.3991748897028693, -0.15414122174138611, -0.20008742443379549]], [[[-0.7410750761192828, -0.34455689325781347], -0.8446675414135969], [-0.8139112572198548, -0.7250728258598154, -0.42851563653684244, [1.0498296931855706, 1.6969612860075955, -0.18093559189614564, 1.078608791657082]]], [[0.5172670690419124]]], [[-1.9731106633939228, 0.5778640337060391], [-1.2488533773832633, -2.1458066486349434, -0.5439318468515132, [[0.2419441207503176, -2.313974422156488, [-0.6811651539055098, 0.08323572953509818], 1.801261721511669, 0.16653718365329456], -0.6348811801078983, [0.016350096268563003, [-1.2867920376687112, 0.38205295881313484, 1.4093210810506318, -0.2698869943849985, -0.48804922126979045]]], -0.6297773736098737, -2.5333506573111424], [-1.6680144776019314, 0.5862818687707498]], [0.6266171347177766, [[-0.7660737060966999, -0.677432480564727, -1.1527197837522167], -0.5025371508398492, [0.3610998752041169, 0.4811870365139723, -0.8030689233086394, [1.1538103888031122, -1.0955905747145644], -1.3980944016010062, 1.2822990047991039]], 0.939566155023095, [1.3581048298505891, [0.36949478822799947, 1.096666130135532, -0.2769024331557954, -0.7993215902675834], [-0.4103823967097248], [0.6789480075462166, 0.8991579880810466, 0.7900472554969632]], [], [0.6772644918729233, [-0.48385354748861575, -0.39154812719778437], 1.069329510451712, 0.8057750827838897, -0.3440192823735095], [[1.5687828887524105, -1.6086288847970498, [-0.6907842744344904], -0.42627155869364414], 0.33605387861917574, -0.7329513818714791, 0.5040026160756554, -1.2529377572694538, -1.1566264096307166], [[0.6407540268295862], [-0.017540252205401917], -0.9530971110439417], [[0.41643810453893765, -0.682997865214066, 0.7930286671567052], 0.5142103949393788]], [[0.6271004836147108, [0.5895664560584991, -0.7563863809912544]], [1.6176958047983054, 0.5226854288884638, 0.24149248202497436], -1.0912185170716135, [-1.1122535648683918], 0.22727974012353094], [-0.4161362684360263, [[0.4234696267033054], 0.7866791657813567, [1.225201951430818, -0.49790730839958713, 0.2715010029532568], -0.051866117232298316]]]]
        >>> ak.typeof(builder.snapshot())
        1 * var * var * union[var * union[float64, var * union[var * union[float64, var * float64], float64]], float64]

    Note that this is a *general* method for building arrays; if the type is
    known in advance, more specialized procedures can be faster. This should
    be considered the "least effort" approach.
    """

    def __init__(self, behavior=None):
        self._layout = awkward1.layout.ArrayBuilder()
        self.behavior = behavior

    @classmethod
    def _wrap(cls, layout, behavior=None):
        """
        Args:
            layout (#ak.layout.ArrayBuilder): Low-level builder to wrap.
            behavior (None or dict): Custom #ak.behavior for arrays built by
                this ArrayBuilder.

        Wraps a low-level #ak.layout.ArrayBuilder as a high-level
        #ak.ArrayBulider.

        The #ak.ArrayBuilder constructor creates a new #ak.layout.ArrayBuilder
        with no accumulated data, but Numba needs to wrap existing data
        when returning from a lowered function.
        """
        assert isinstance(layout, awkward1.layout.ArrayBuilder)
        out = cls.__new__(cls)
        out._layout = layout
        out.behavior = behavior
        return out

    @property
    def behavior(self):
        """
        The `behavior` parameter passed into this ArrayBuilder's constructor.

           * If a dict, this `behavior` overrides the global #ak.behavior.
             Any keys in the global #ak.behavior but not this `behavior` are
             still valid, but any keys in both are overridden by this
             `behavior`. Keys with a None value are equivalent to missing keys,
             so this `behavior` can effectively remove keys from the
             global #ak.behavior.

           * If None, the Array defaults to the global #ak.behavior.

        See #ak.behavior for a list of recognized key patterns and their
        meanings.
        """
        return self._behavior

    @behavior.setter
    def behavior(self, behavior):
        if behavior is None or isinstance(behavior, dict):
            self._behavior = behavior
        else:
            raise TypeError("behavior must be None or a dict")

    @property
    def type(self):
        """
        The current high-level type of this ArrayBuilder as #ak.types.Type
        objects.

        Note that the type can change as data are accumulated.

        See #ak.Array.type for a more complete description.
        """
        return self.snapshot().type

    def __len__(self):
        """
        The current length of the accumulated array.
        """
        return len(self._layout)

    def __getitem__(self, where):
        """
        Args:
            where (many types supported; see below): Index of positions to
                select from the array.

        Takes a #snapshot and selects items from the array.

        See #ak.Array.__getitem__ for a more complete description.
        """
        return awkward1._util.wrap(self._layout[where], self._behavior)

    def __iter__(self):
        """
        Iterates over a #snapshot of the array in Python.

        See #ak.Array.__iter__ for performance considerations.
        """
        for x in self.snapshot():
            yield x

    def __str__(self, limit_value=85, snapshot=None):
        """
        Args:
            limit_value (int): Maximum number of characters to use when
                presenting the ArrayBuilder as a string.

        Presents this ArrayBuilder as a string without type or
        `"<ArrayBuilder ...>"`.

        See #ak.Array.__str__ for a more complete description.
        """
        if snapshot is None:
            snapshot = self.snapshot()
        return snapshot.__str__(limit_value=limit_value)

    def __repr__(self, limit_value=40, limit_total=85):
        """
        Args:
            limit_value (int): Maximum number of characters to use when
                presenting the data of the ArrayBuilder.
            limit_total (int): Maximum number of characters to use for
                the whole string (should be larger than `limit_value`).

        Presents this ArrayBuilder as a string with its type and
        `"<ArrayBuilder ...>"`.

        See #ak.Array.__repr__ for a more complete description.
        """
        snapshot = self.snapshot()
        value = self.__str__(limit_value=limit_value, snapshot=snapshot)

        limit_type = limit_total - len(value) - len("<ArrayBuilder  type=>")
        typestrs = awkward1._util.typestrs(self._behavior)
        type = repr(str(snapshot.type(typestrs)))
        if len(type) > limit_type:
            type = type[:(limit_type - 4)] + "..." + type[-1]

        return "<ArrayBuilder {0} type={1}>".format(value, type)

    def __array__(self, *args, **kwargs):
        """
        Intercepts attempts to convert a #snapshot of this array into a
        NumPy array and either performs a zero-copy conversion or raises
        an error.

        See #ak.Array.__array__ for a more complete description.
        """
        return awkward1._connect._numpy.convert_to_array(
                 self.snapshot(), args, kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Intercepts attempts to pass this ArrayBuilder to a NumPy
        [universal functions](https://docs.scipy.org/doc/numpy/reference/ufuncs.html)
        (ufuncs) and passes it through the structure of the array's #snapshot.

        See #ak.Array.__array_ufunc__ for a more complete description.
        """
        return awkward1._connect._numpy.array_ufunc(ufunc,
                                                    method,
                                                    inputs,
                                                    kwargs,
                                                    self._behavior)

    def __array_function__(self, func, types, args, kwargs):
        """
        Intercepts attempts to pass this ArrayBuilder to those NumPy functions
        other than universal functions that have an Awkward equivalent.

        See #ak.ArrayBuilder.__array_ufunc__ for a more complete description.
        """
        return awkward1._connect._numpy.array_function(func,
                                                       types,
                                                       args,
                                                       kwargs)

    @property
    def numbatype(self):
        """
        The type of this Array when it is used in Numba. It contains enough
        information to generate low-level code for accessing any element,
        down to the leaves.

        See [Numba documentation](https://numba.pydata.org/numba-doc/dev/reference/types.html)
        on types and signatures.
        """
        import numba
        import awkward1._connect._numba.builder
        awkward1._connect._numba.register()
        return awkward1._connect._numba.builder.ArrayBuilderType(
                 self._behavior)

    def snapshot(self):
        """
        Converts the currently accumulated data into an #ak.Array.

        This is almost always an *O(1)* operation (does not scale with the
        size of the accumulated data, and therefore safe to call relatively
        often).

        The resulting #ak.Array shares memory with the accumulated data (it
        is a zero-copy operation), but it is safe to continue filling the
        ArrayBuilder because its append-only operations only affect data
        outside the range viewed by old snapshots. If ArrayBuilder reallocates
        an internal buffer, the data are no longer shared, but they're
        reference-counted by the #ak.Array and the #ak.ArrayBuilder, so all
        buffers are deleted exactly once.
        """
        layout = self._layout.snapshot()
        return awkward1._util.wrap(layout, self._behavior)

    def null(self):
        """
        Appends a None value at the current position in the accumulated array.
        """
        self._layout.null()

    def boolean(self, x):
        """
        Appends a boolean value `x` at the current position in the accumulated
        array.
        """
        self._layout.boolean(x)

    def integer(self, x):
        """
        Appends an integer `x` at the current position in the accumulated
        array.
        """
        self._layout.integer(x)

    def real(self, x):
        """
        Appends a floating point number `x` at the current position in the
        accumulated array.
        """
        self._layout.real(x)

    def bytestring(self, x):
        """
        Appends an unencoded string (raw bytes) `x` at the current position
        in the accumulated array.
        """
        self._layout.bytestring(x)

    def string(self, x):
        """
        Appends a UTF-8 encoded string `x` at the current position in the
        accumulated array.
        """
        self._layout.string(x)

    def beginlist(self):
        """
        Begins filling a list; must be closed with #endlist.

        For example,

            builder.beginlist()
            builder.real(1.1)
            builder.real(2.2)
            builder.real(3.3)
            builder.endlist()
            builder.beginlist()
            builder.endlist()
            builder.beginlist()
            builder.real(4.4)
            builder.real(5.5)
            builder.endlist()

        produces

            [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
        """
        self._layout.beginlist()

    def endlist(self):
        """
        Ends a list.
        """
        self._layout.endlist()

    def begintuple(self, numfields):
        """
        Begins filling a tuple with `numfields` fields; must be closed with
        #endtuple.

        For example,

            builder.begintuple(3)
            builder.index(0).integer(1)
            builder.index(1).real(1.1)
            builder.index(2).string("one")
            builder.endtuple()
            builder.begintuple(3)
            builder.index(0).integer(2)
            builder.index(1).real(2.2)
            builder.index(2).string("two")
            builder.endtuple()

        produces

            [(1, 1.1, "one"), (2, 2.2, "two")]
        """
        self._layout.begintuple(numfields)

    def index(self, i):
        """
        Args:
            i (int): The tuple slot to fill.
        Returns:
            The #ak.ArrayBuilder, so that it can be chained with the value
                that fills the slot.

        Prepares to fill a tuple slot; see #begintuple for an example.
        """
        self._layout.index(i)
        return self

    def endtuple(self):
        """
        Ends a tuple.
        """
        self._layout.endtuple()

    def beginrecord(self, name=None):
        """
        Begins filling a record with an optional `name`; must be closed with
        #endrecord.

        For example,

            builder.beginrecord("points")
            builder.field("x").real(1)
            builder.field("y").real(1.1)
            builder.endrecord()
            builder.beginrecord("points")
            builder.field("x").real(2)
            builder.field("y").real(2.2)
            builder.endrecord()

        produces

            [{"x": 1.0, "y": 1.1}, {"x": 2.0, "y": 2.2}]

        with type

            2 * struct[["x", "y"], [float64, float64], parameters={"__record__": "points"}]

        The `"__record__"` parameter can be used to add behavior to the records
        in the array, as described in #ak.Array and #ak.Record.
        """
        self._layout.beginrecord(name)

    def field(self, key):
        """
        Args:
            key (str): The field key to fill.
        Returns:
            The #ak.ArrayBuilder, so that it can be chained with the value
                that fills the slot.

        Prepares to fill a field; see #beginrecord for an example.
        """
        self._layout.field(key)
        return self

    def endrecord(self):
        """
        Ends a record.
        """
        self._layout.endrecord()

    def append(self, obj, at=None):
        """
        Args:
            obj: The object to append.
            at (None or int): which value to select from `obj` if `obj` is
                an #ak.Array.

        Appends any type of object, which can be a shorthand for #null,
        #boolean, #integer, #real, #bytestring, or #string, but also
        an #ak.Array or #ak.Record to *reference* values from an existing
        dataset, or any Python object to *convert* to Awkward Array.

        If `obj` is an #ak.Array or #ak.Record, the output will be an
        #ak.layout.IndexedArray64 (or #ak.layout.IndexedOptionArray64 if
        there are any None values) that references the existing data. This
        can be a more time and memory-efficient way to put old data in a
        new structure, since it avoids copying and even walking over the
        old data structure (matters more when the structures are large).

        If `obj` is an arbitrary Python object, this is equivalent to
        #ak.fromiter except that it fills an existing #ak.ArrayBuilder,
        rather than creating a new one.

        If `obj` is an #ak.Array and `at` is an int, this method fills the
        ArrayBuilder with a reference to `obj[at]` instead of `obj`.
        """
        if at is None:
            if isinstance(obj, Record):
                self._layout.append(obj.layout.array, obj.layout.at)
            elif isinstance(obj, Array):
                self._layout.extend(obj.layout)
            else:
                self._layout.fromiter(obj)

        else:
            if isinstance(obj, Array):
                self._layout.append(obj.layout, at)
            else:
                raise TypeError(
                        "'append' method can only be used with 'at' when "
                        "'obj' is an ak.Array")

    def extend(self, obj):
        """
        Args:
            obj (#ak.Array): The Array to concatenate with the data in this
                ArrayBuilder.

        Appends every value from `obj`, by reference (see #append).
        """
        if isinstance(obj, Array):
            self._layout.extend(obj.layout)
        else:
            raise TypeError("'extend' method requires an ak.Array")

    class _Nested(object):
        def __init__(self, arraybuilder):
            self._arraybuilder = arraybuilder

        def __repr__(self):
            snapshot = self._arraybuilder.snapshot()
            value = self._arraybuilder.__str__(limit_value=limit_value,
                                               snapshot=snapshot)

            limit_type = (limit_total - len(value)
                          - len("<ArrayBuilder.  type=>") - len(self._name))
            typestrs = awkward1._util.typestrs(self._arraybuilder._behavior)
            type = repr(str(snapshot.type(typestrs)))
            if len(type) > limit_type:
                type = type[:(limit_type - 4)] + "..." + type[-1]

            return "<ArrayBuilder.{0} {1} type={2}>".format(self._name,
                                                            value,
                                                            type)

    class List(_Nested):
        _name = "List"

        def __enter__(self):
            self._arraybuilder.beginlist()

        def __exit__(self, type, value, traceback):
            self._arraybuilder.endlist()

    def list(self):
        """
        Context manager to prevent unpaired #beginlist and #endlist. The
        example in the #beginlist documentation can be rewritten as

            with builder.list():
                builder.real(1.1)
                builder.real(2.2)
                builder.real(3.3)
            with builder.list():
                pass
            with builder.list():
                builder.real(4.4)
                builder.real(5.5)

        to produce the same result.

            [[1.1, 2.2, 3.3], [], [4.4, 5.5]]

        Since context managers aren't yet suppored by Numba, this method
        can't be used in Numba.
        """
        return self.List(self)

    class Tuple(_Nested):
        _name = "Tuple"

        def __init__(self, arraybuilder, numfields):
            super(ArrayBuilder.Tuple, self).__init__(arraybuilder)
            self._numfields = numfields

        def __enter__(self):
            self._arraybuilder.begintuple(self._numfields)

        def __exit__(self, type, value, traceback):
            self._arraybuilder.endtuple()

    def tuple(self, numfields):
        """
        Context manager to prevent unpaired #begintuple and #endtuple. The
        example in the #begintuple documentation can be rewritten as

            with builder.tuple(3):
                builder.index(0).integer(1)
                builder.index(1).real(1.1)
                builder.index(2).string("one")
            with builder.tuple(3):
                builder.index(0).integer(2)
                builder.index(1).real(2.2)
                builder.index(2).string("two")

        to produce the same result.

            [(1, 1.1, "one"), (2, 2.2, "two")]

        Since context managers aren't yet suppored by Numba, this method
        can't be used in Numba.
        """
        return self.Tuple(self, numfields)

    class Record(_Nested):
        _name = "Record"

        def __init__(self, arraybuilder, name):
            super(ArrayBuilder.Record, self).__init__(arraybuilder)
            self._name = name

        def __enter__(self):
            self._arraybuilder.beginrecord(name=self._name)

        def __exit__(self, type, value, traceback):
            self._arraybuilder.endrecord()

    def record(self, name=None):
        """
        Context manager to prevent unpaired #beginrecord and #endrecord. The
        example in the #beginrecord documentation can be rewritten as

            with builder.record("points"):
                builder.field("x").real(1)
                builder.field("y").real(1.1)
            with builder.record("points"):
                builder.field("x").real(2)
                builder.field("y").real(2.2)

        to produce the same result.

            [{"x": 1.0, "y": 1.1}, {"x": 2.0, "y": 2.2}]

        Since context managers aren't yet suppored by Numba, this method
        can't be used in Numba.
        """
        return self.Record(self, name)
