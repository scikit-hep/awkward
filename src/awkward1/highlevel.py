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
        behavior (None or dict): Custom #ak.behavior for this array only.
        withname (None or str): Gives tuples and records a name that can be
            used to override their behavior (see below).
        checkvalid (bool): If True, verify that the #layout is valid.

    **High-level array that can contain data of any type.**

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

        vectors.x
        # <Array [0.1, 0.2, 0.3] type='3 * float64'>
        vectors.y
        # <Array [1, 2, 3] type='3 * float64'>
        vectors.z
        # <Array [10, 20, 30] type='3 * float64'>

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

        vectors
        # <Array [{x: 0.1, y: 1, z: 10, ... y: 3, z: 30}] type='3 * vec3'>
        more_vectors
        # <Array [{x: 10, y: 1, z: 0.1, ... z: 0.3}] type='3 * vec3'>
        type(vectors)
        # <class '__main__.Vec3Array'>
        type(more_vectors)
        # <class '__main__.Vec3Array'>

        vectors.cross(more_vectors)
        # <Array [{x: -9.9, y: 100, ... z: -89.1}] type='3 * vec3'>

    If the #ak.cross function were a method of this Array class, then it would
    conflict with applications where we might want `array.cross` to mean
    something else.
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
        The composable #ak.layout.Content elements that determine how the
        array is structured.

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
        The high-level type of an array as #ak.types.Type objects.

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
        The length of the array, only counting the outermost structure.

        For example, the length of

            ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])

        is `3`, not `5`.
        """
        return len(self._layout)

    def __iter__(self):
        """
        Iterates over the array in Python.

        Note that this is the _slowest_ way to access data (even slower than
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
        Select items from the Array using an extension of NumPy's (already
        quite extensive) rules.

        All methods of selecting items described in
        [NumPy indexing](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html)
        are supported with one exception
        ([combining advanced and basic indexing](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#combining-advanced-and-basic-indexing)
        with basic indexes _between_ two advanced indexes: the definition
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

        The following array

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

        For example, the array

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
        if not isinstance(where, str):
            raise ValueError(
                    "only fields may be assigned in-place (by field name)")
        self._layout = awkward1.operations.structure.withfield(self._layout,
                                                               what,
                                                               where).layout
        self._numbaview = None

    def __getattr__(self, where):
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
        return sorted(set(dir(super(Array, self))
                          + [x for x in self._layout.keys()
                               if _dir_pattern.match(x) and
                                  not keyword.iskeyword(x)]))

    @property
    def i0(self):
        return self["0"]
    @property
    def i1(self):
        return self["1"]
    @property
    def i2(self):
        return self["2"]
    @property
    def i3(self):
        return self["3"]
    @property
    def i4(self):
        return self["4"]
    @property
    def i5(self):
        return self["5"]
    @property
    def i6(self):
        return self["6"]
    @property
    def i7(self):
        return self["7"]
    @property
    def i8(self):
        return self["8"]
    @property
    def i9(self):
        return self["9"]

    def __str__(self, limit_value=85):
        return awkward1._util.minimally_touching_string(limit_value,
                                                        self._layout,
                                                        self._behavior)

    def __repr__(self, limit_value=40, limit_total=85):
        value = awkward1._util.minimally_touching_string(limit_value,
                                                         self._layout,
                                                         self._behavior)

        limit_type = limit_total - len(value) - len("<Array  type=>")
        type = repr(str(self.type))
        if len(type) > limit_type:
            type = type[:(limit_type - 4)] + "..." + type[-1]

        return "<Array {0} type={1}>".format(value, type)

    def __array__(self, *args, **kwargs):
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

    def __array_function__(self, func, types, args, kwargs):
        return awkward1._connect._numpy.array_function(func,
                                                       types,
                                                       args,
                                                       kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return awkward1._connect._numpy.array_ufunc(ufunc,
                                                    method,
                                                    inputs,
                                                    kwargs,
                                                    self._behavior)

    @property
    def numbatype(self):
        import numba
        import awkward1._connect._numba
        awkward1._connect._numba.register()
        if self._numbaview is None:
            self._numbaview = \
              awkward1._connect._numba.arrayview.ArrayView.fromarray(self)
        return numba.typeof(self._numbaview)

class Record(awkward1._connect._numpy.NDArrayOperatorsMixin):
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
        return self._behavior

    @behavior.setter
    def behavior(self, behavior):
        if behavior is None or isinstance(behavior, dict):
            self._behavior = behavior
        else:
            raise TypeError("behavior must be None or a dict")

    @property
    def type(self):
        return self._layout.type(awkward1._util.typestrs(self._behavior))

    def __getitem__(self, where):
        return awkward1._util.wrap(self._layout[where], self._behavior)

    def __setitem__(self, where, what):
        if not isinstance(where, str):
            raise ValueError(
                    "only fields may be assigned in-place (by field name)")
        self._layout = awkward1.operations.structure.withfield(self._layout,
                                                               what,
                                                               where).layout
        self._numbaview = None

    def __getattr__(self, where):
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
        return sorted(set(dir(super(Array, self))
                          + [x for x in self._layout.keys()
                               if _dir_pattern.match(x) and
                               not keyword.iskeyword(x)]))

    @property
    def i0(self):
        return self["0"]
    @property
    def i1(self):
        return self["1"]
    @property
    def i2(self):
        return self["2"]
    @property
    def i3(self):
        return self["3"]
    @property
    def i4(self):
        return self["4"]
    @property
    def i5(self):
        return self["5"]
    @property
    def i6(self):
        return self["6"]
    @property
    def i7(self):
        return self["7"]
    @property
    def i8(self):
        return self["8"]
    @property
    def i9(self):
        return self["9"]

    def __str__(self, limit_value=85):
        return awkward1._util.minimally_touching_string(limit_value + 2,
                                                        self._layout,
                                                        self._behavior)[1:-1]

    def __repr__(self, limit_value=40, limit_total=85):
        value = awkward1._util.minimally_touching_string(limit_value + 2,
                                                         self._layout,
                                                         self._behavior)[1:-1]

        limit_type = limit_total - len(value) - len("<Record  type=>")
        type = repr(str(self.type))
        if len(type) > limit_type:
            type = type[:(limit_type - 4)] + "..." + type[-1]

        return "<Record {0} type={1}>".format(value, type)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return awkward1._connect._numpy.array_ufunc(ufunc,
                                                    method,
                                                    inputs,
                                                    kwargs,
                                                    self._behavior)

    @property
    def numbatype(self):
        import numba
        import awkward1._connect._numba
        awkward1._connect._numba.register()
        if self._numbaview is None:
            self._numbaview = \
              awkward1._connect._numba.arrayview.RecordView.fromrecord(self)
        return numba.typeof(self._numbaview)

class ArrayBuilder(Sequence):
    @classmethod
    def _wrap(cls, layout, behavior=None, withname=None):
        assert isinstance(layout, awkward1.layout.ArrayBuilder)
        out = cls.__new__(cls)
        out._layout = layout
        out._withname = withname
        out.behavior = behavior
        return out

    def __init__(self, behavior=None, withname=None):
        self._layout = awkward1.layout.ArrayBuilder()
        self._withname = withname
        self.behavior = behavior

    @property
    def behavior(self):
        return self._behavior

    @behavior.setter
    def behavior(self, behavior):
        if behavior is None or isinstance(behavior, dict):
            self._behavior = behavior
        else:
            raise TypeError("behavior must be None or a dict")

    @property
    def type(self):
        return self.snapshot().type

    def __len__(self):
        return len(self._layout)

    def __getitem__(self, where):
        return awkward1._util.wrap(self._layout[where], self._behavior)

    def __iter__(self):
        for x in self.snapshot():
            yield x

    def __str__(self, limit_value=85, snapshot=None):
        if snapshot is None:
            snapshot = self.snapshot()
        return snapshot.__str__(limit_value=limit_value)

    def __repr__(self, limit_value=40, limit_total=85):
        snapshot = self.snapshot()
        value = self.__str__(limit_value=limit_value, snapshot=snapshot)

        limit_type = limit_total - len(value) - len("<ArrayBuilder  type=>")
        typestrs = awkward1._util.typestrs(self._behavior)
        type = repr(str(snapshot.type(typestrs)))
        if len(type) > limit_type:
            type = type[:(limit_type - 4)] + "..." + type[-1]

        return "<ArrayBuilder {0} type={1}>".format(value, type)

    def __array__(self, *args, **kwargs):
        return awkward1._connect._numpy.convert_to_array(
                 self.snapshot(), args, kwargs)

    def __array_function__(self, func, types, args, kwargs):
        return awkward1._connect._numpy.array_function(func,
                                                       types,
                                                       args,
                                                       kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return awkward1._connect._numpy.array_ufunc(ufunc,
                                                    method,
                                                    inputs,
                                                    kwargs,
                                                    self._behavior)

    @property
    def numbatype(self):
        import numba
        import awkward1._connect._numba.builder
        awkward1._connect._numba.register()
        return awkward1._connect._numba.builder.ArrayBuilderType(
                 self._behavior)

    def snapshot(self):
        layout = self._layout.snapshot()
        if self._withname is not None:
            layout = awkward1.operations.structure.withname(layout,
                                                            self._withname,
                                                            highlevel=False)
        return awkward1._util.wrap(layout, self._behavior)

    def null(self):
        self._layout.null()

    def boolean(self, x):
        self._layout.boolean(x)

    def integer(self, x):
        self._layout.integer(x)

    def real(self, x):
        self._layout.real(x)

    def bytestring(self, x):
        self._layout.bytestring(x)

    def string(self, x):
        self._layout.string(x)

    def beginlist(self):
        self._layout.beginlist()

    def endlist(self):
        self._layout.endlist()

    def begintuple(self):
        self._layout.begintuple()

    def index(self, i):
        self._layout.index(i)

    def endtuple(self):
        self._layout.endtuple()

    def beginrecord(self, name=None):
        self._layout.beginrecord(name)

    def field(self, key):
        self._layout.field(key)

    def endrecord(self):
        self._layout.endrecord()

    def append(self, obj, at=None):
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
        if isinstance(obj, Array):
            self._layout.extend(obj.layout)
        else:
            raise TypeError("'extend' method requires an ak.Array")
