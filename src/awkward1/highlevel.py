# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import re
import keyword

try:
    from collections.abc import Iterable
    from collections.abc import Sized
    from collections.abc import MutableMapping
except ImportError:
    from collections import Iterable
    from collections import Sized
    from collections import MutableMapping

import awkward1._connect._numpy
import awkward1._connect._pandas
import awkward1.nplike
import awkward1.layout
import awkward1.operations.convert
import awkward1.operations.structure


np = awkward1.nplike.NumpyMetadata.instance()
numpy = awkward1.nplike.Numpy.instance()

_dir_pattern = re.compile(r"^[a-zA-Z_]\w*$")


class Array(
    awkward1._connect._numpy.NDArrayOperatorsMixin,
    awkward1._connect._pandas.PandasMixin,
    Iterable,
    Sized,
):
    u"""
    Args:
        data (#ak.layout.Content, #ak.partition.PartitionedArray, #ak.Array,
              np.ndarray, pyarrow.*, str, dict, or iterable):
            Data to wrap or convert into an array.
            If a NumPy array, the regularity of its dimensions is preserved
            and the data are viewed, not copied.
            If a pyarrow object, calls #ak.from_arrow, preserving as much
            metadata as possible, usually zero-copy.
            If a dict of str \u2192 columns, combines the columns into an
            array of records (like Pandas's DataFrame constructor).
            If a string, the data are assumed to be JSON.
            If an iterable, calls #ak.from_iter, which assumes all dimensions
            have irregular lengths.
        behavior (None or dict): Custom #ak.behavior for this Array only.
        with_name (None or str): Gives tuples and records a name that can be
            used to override their behavior (see below).
        check_valid (bool): If True, verify that the #layout is valid.
        cache (None or MutableMapping): Stores data for any
            #ak.layout.VirtualArray nodes that this Array might contain.
            Persists through `__getitem__` but not any other operations.

    High-level array that can contain data of any type.

    For most users, this is the only class in Awkward Array that matters: it
    is the entry point for data analysis with an emphasis on usability. It
    intentionally has a minimum of methods, preferring standalone functions
    like

        ak.num(array1)
        ak.combinations(array1)
        ak.cartesian([array1, array2])
        ak.zip({"x": array1, "y": array2, "z": array3})

    instead of bound methods like

        array1.num()
        array1.combinations()
        array1.cartesian([array2, array3])
        array1.zip(...)   # ?

    because its namespace is valuable for domain-specific parameters and
    functionality. For example, if records contain a field named `"num"`,
    they can be accessed as

        array1.num

    instead of

        array1["num"]

    without any confusion or interference from #ak.num. The same is true
    for domain-specific methods that have been attached to the data. For
    instance, an analysis of mailing addresses might have a function that
    computes zip codes, which can be attached to the data with a method
    like

        latlon.zip()

    without any confusion or interference from #ak.zip. Custom methods like
    this can be added with #ak.behavior, and so the namespace of Array
    attributes must be kept clear for such applications.

    See also #ak.Record.

    Interfaces to other libraries
    =============================

    NumPy
    *****

    When NumPy
    [universal functions](https://docs.scipy.org/doc/numpy/reference/ufuncs.html)
    (ufuncs) are applied to an ak.Array, they are passed through the Awkward
    data structure, applied to the numerical data at its leaves, and the output
    maintains the original structure.

    For example,

        >>> array = ak.Array([[1, 4, 9], [], [16, 25]])
        >>> np.sqrt(array)
        <Array [[1, 2, 3], [], [4, 5]] type='3 * var * float64'>

    See also #ak.Array.__array_ufunc__.

    Some NumPy functions other than ufuncs are also handled properly in
    NumPy >= 1.17 (see
    [NEP 18](https://numpy.org/neps/nep-0018-array-function-protocol.html))
    and if an Awkward override exists. That is,

        np.concatenate

    can be used on an Awkward Array because

        ak.concatenate

    exists. If your NumPy is older than 1.17, use `ak.concatenate` directly.

    Pandas
    ******

    Ragged arrays (list type) can be converted into Pandas
    [MultiIndex](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html)
    rows and nested records can be converted into MultiIndex columns. If the
    Awkward Array has only one "branch" of nested lists (i.e. different record
    fields do not have different-length lists, but a single chain of lists-of-lists
    is okay), then it can be losslessly converted into a single DataFrame.
    Otherwise, multiple DataFrames are needed, though they can be merged (with a
    loss of information).

    The #ak.to_pandas function performs this conversion; if `how=None`, it
    returns a list of DataFrames; otherwise, `how` is passed to `pd.merge` when
    merging the resultant DataFrames.

    Numba
    *****

    Arrays can be used in [Numba](http://numba.pydata.org/): they can be
    passed as arguments to a Numba-compiled function or returned as return
    values. The only limitation is that Awkward Arrays cannot be *created*
    inside the Numba-compiled function; to make outputs, consider
    #ak.ArrayBuilder.

    Arrow
    *****

    Arrays are convertible to and from [Apache Arrow](https://arrow.apache.org/),
    a standard for representing nested data structures in columnar arrays.
    See #ak.to_arrow and #ak.from_arrow.

    NumExpr
    *******

    [NumExpr](https://numexpr.readthedocs.io/en/latest/user_guide.html) can
    calculate expressions on a set of ak.Arrays, but only if the functions in
    `ak.numexpr` are used, not the functions in the `numexpr` library directly.

    Like NumPy ufuncs, the expression is evaluated on the numeric leaves of the
    data structure, maintaining structure in the output.

    See #ak.numexpr.evaluate to calculate an expression.

    See #ak.numexpr.re_evaluate to recalculate an expression without
    rebuilding its virtual machine.

    Autograd
    ********

    Derivatives of a calculation on a set of ak.Arrays can be calculated with
    [Autograd](https://github.com/HIPS/autograd#readme), but only if the
    function in `ak.autograd` is used, not the functions in the `autograd`
    library directly.

    Like NumPy ufuncs, the function and its derivatives are evaluated on the
    numeric leaves of the data structure, maintaining structure in the output.

    See #ak.autograd.elementwise_grad to calculate a function and its
    derivatives elementwise on each numeric value in an ak.Array.
    """

    def __init__(
        self, data, behavior=None, with_name=None, check_valid=False, cache=None
    ):
        if isinstance(
            data, (awkward1.layout.Content, awkward1.partition.PartitionedArray)
        ):
            layout = data

        elif isinstance(data, Array):
            layout = data.layout

        elif isinstance(data, np.ndarray) and data.dtype != np.dtype("O"):
            layout = awkward1.operations.convert.from_numpy(data, highlevel=False)

        elif type(data).__module__ == "pyarrow" or type(data).__module__.startswith("pyarrow."):
            layout = awkward1.operations.convert.from_arrow(data, highlevel=False)

        elif isinstance(data, dict):
            keys = []
            contents = []
            for k, v in data.items():
                keys.append(k)
                contents.append(Array(v).layout)
            parameters = None
            if with_name is not None:
                parameters = {"__record__": with_name}
            layout = awkward1.layout.RecordArray(
                contents, keys, parameters=parameters
            )

        elif isinstance(data, str):
            layout = awkward1.operations.convert.from_json(data, highlevel=False)

        else:
            layout = awkward1.operations.convert.from_iter(
                data, highlevel=False, allow_record=False
            )

        if not isinstance(
            layout, (awkward1.layout.Content, awkward1.partition.PartitionedArray)
        ):
            raise TypeError(
                "could not convert data into an awkward1.Array"
                + awkward1._util.exception_suffix(__file__)
            )

        if with_name is not None:
            layout = awkward1.operations.structure.with_name(
                layout, with_name, highlevel=False
            )
        if self.__class__ is Array:
            self.__class__ = awkward1._util.arrayclass(layout, behavior)

        self.layout = layout
        self.behavior = behavior
        docstr = self.layout.purelist_parameter("__doc__")
        if isinstance(docstr, str):
            self.__doc__ = docstr
        if check_valid:
            awkward1.operations.describe.validity_error(self, exception=True)

        self.cache = cache

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
        the following `array`

            ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])

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
        if isinstance(
            layout, (awkward1.layout.Content, awkward1.partition.PartitionedArray)
        ):
            self._layout = layout
            self._numbaview = None
        else:
            raise TypeError(
                "layout must be a subclass of awkward1.layout.Content"
                + awkward1._util.exception_suffix(__file__)
            )

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
            raise TypeError(
                "behavior must be None or a dict"
                + awkward1._util.exception_suffix(__file__)
            )

    @property
    def cache(self):
        return self._cache

    @cache.setter
    def cache(self, value):
        if value is None or isinstance(value, MutableMapping):
            self._cache = value
        else:
            raise TypeError(
                "cache must be None or a MutableMapping"
                + awkward1._util.exception_suffix(__file__)
            )

    class Mask(object):
        def __init__(self, array, valid_when):
            self._array = array
            self._valid_when = valid_when

        def __str__(self):
            return self._str()

        def __repr__(self):
            return self._repr()

        def _str(self, limit_value=85):
            return self._array._str(limit_value=limit_value)

        def _repr(self, limit_value=40, limit_total=85):
            import awkward1.operations.structure

            layout = awkward1.operations.structure.with_cache(
                self._layout, {}, chain="last", highlevel=False
            )
            value = awkward1._util.minimally_touching_string(
                limit_value, layout, self._behavior
            )

            try:
                name = super(Array, self._array).__getattribute__("__name__")
            except AttributeError:
                name = type(self._array).__name__
            limit_type = limit_total - (len(value) + len(name) + len("<.mask  type=>"))
            typestr = repr(
                str(awkward1._util.highlevel_type(layout, self._array._behavior, True))
            )
            if len(typestr) > limit_type:
                typestr = typestr[: (limit_type - 4)] + "..." + typestr[-1]

            return "<{0}.mask {1} type={2}>".format(name, value, typestr)

        def __getitem__(self, where):
            return awkward1.operations.structure.mask(
                self._array, where, self._valid_when
            )

    @property
    def mask(self, valid_when=True):
        """
        Whereas

            array[array_of_booleans]

        removes elements from `array` in which `array_of_booleans` is False,

            array.mask[array_of_booleans]

        returns data with the same length as the original `array` but False
        values in `array_of_booleans` are mapped to None. Such an output
        can be used in mathematical expressions with the original `array`
        because they are still aligned.

        See <<filtering>> and #ak.mask.
        """
        return self.Mask(self, valid_when)

    def tolist(self):
        """
        Converts this Array into Python objects; same as #ak.to_list
        (but without the underscore, like NumPy's
        [tolist](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.tolist.html)).

        Awkward Array types have the following Pythonic translations.

           * #ak.types.PrimitiveType: converted into bool, int, float.
           * #ak.types.OptionType: missing values are converted into None.
           * #ak.types.ListType: converted into list.
           * #ak.types.RegularType: also converted into list. Python (and JSON)
             forms lose information about the regularity of list lengths.
           * #ak.types.ListType with parameter `"__array__"` equal to
             `"__bytestring__"`: converted into bytes.
           * #ak.types.ListType with parameter `"__array__"` equal to
             `"__string__"`: converted into str.
           * #ak.types.RecordArray without field names: converted into tuple.
           * #ak.types.RecordArray with field names: converted into dict.
           * #ak.types.UnionArray: Python data are naturally heterogeneous.

        See also #ak.to_list and #ak.from_iter.
        """
        return awkward1.operations.convert.to_list(self)

    def tojson(
        self, destination=None, pretty=False, maxdecimals=None, buffersize=65536
    ):
        """
        Args:
            destination (None or str): If None, this method returns a JSON str;
                if a str, it uses that as a file name and writes (overwrites)
                that file (returning None).
            pretty (bool): If True, indent the output for human readability; if
                False, output compact JSON without spaces.
            maxdecimals (None or int): If an int, limit the number of
                floating-point decimals to this number; if None, write all
                digits.
            buffersize (int): Size (in bytes) of the buffer used by the JSON
                parser.

        Converts this Array into a JSON string or file; same as #ak.to_json
        (but without the underscore, like #ak.Array.tolist).

        Awkward Array types have the following JSON translations.

           * #ak.types.PrimitiveType: converted into JSON booleans and numbers.
           * #ak.types.OptionType: missing values are converted into None.
           * #ak.types.ListType: converted into JSON lists.
           * #ak.types.RegularType: also converted into JSON lists. JSON (and
             Python) forms lose information about the regularity of list
             lengths.
           * #ak.types.ListType with parameter `"__array__"` equal to
             `"__bytestring__"` or `"__string__"`: converted into JSON strings.
           * #ak.types.RecordArray without field names: converted into JSON
             objects with numbers as strings for keys.
           * #ak.types.RecordArray with field names: converted into JSON
             objects.
           * #ak.types.UnionArray: JSON data are naturally heterogeneous.

        See also #ak.to_json and #ak.from_json.
        """
        return awkward1.operations.convert.to_json(
            self, destination, pretty, maxdecimals, buffersize
        )

    @property
    def nbytes(self):
        """
        The total number of bytes in all the #ak.layout.Index,
        #ak.layout.Identities, and #ak.layout.NumpyArray buffers in this
        array tree.

        Note: this calculation takes overlapping buffers into account, to the
        extent that overlaps are not double-counted, but overlaps are currently
        assumed to be complete subsets of one another, and so it is
        theoretically possible (though unlikely) that this number is an
        underestimate of the true usage.

        It also does not count buffers that must be kept in memory because
        of ownership, but are not directly used in the array. Nor does it count
        the (small) C++ nodes or Python objects that reference the (large)
        array buffers.
        """
        return self._layout.nbytes

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

        See also #ak.to_list.
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
             integer at the deeply nested level. Missing items at any level
             above the deepest level must broadcast. See <<nested indexing>> below.

        A tuple of the above applies each slice item to a dimension of the
        data, which can be very expressive. More than one flat boolean/integer
        array are "iterated as one" as described in the
        [NumPy documentation](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#integer-array-indexing).

        Filtering
        *********

        A common use of selection by boolean arrays is to filter a dataset by
        some property. For instance, to get the odd values of the `array`

            ak.Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        one can put an array expression with True for each odd value inside
        square brackets:

            >>> array[array % 2 == 1]
            <Array [1, 3, 5, 7, 9] type='5 * int64'>

        This technique is so common in NumPy and Pandas data analysis that it
        is often read as a syntax, rather than a consequence of array slicing.

        The extension to nested arrays like

            ak.Array([[[0, 1, 2], [], [3, 4], [5]], [[6, 7, 8], [9]]])

        allows us to use the same syntax more generally.

            >>> array[array % 2 == 1]
            <Array [[[1], [], [3], [5]], [[7], [9]]] type='2 * var * var * int64'>

        In this example, the boolean array is itself nested (see
        <<nested indexing>> below).

            >>> array % 2 == 1
            <Array [[[False, True, False], ... [True]]] type='2 * var * var * bool'>

        This also applies to data with record structures.

        For nested data, we often need to select the first or first two
        elements from variable-length lists. That can be a problem if some
        lists are empty. A function like #ak.num can be useful for first
        selecting by the lengths of lists.

            >>> array = ak.Array([[1.1, 2.2, 3.3],
            ...                   [],
            ...                   [4.4, 5.5],
            ...                   [6.6],
            ...                   [],
            ...                   [7.7, 8.8, 9.9]])
            ...
            >>> array[ak.num(array) > 0, 0]
            <Array [1.1, 4.4, 6.6, 7.7] type='4 * float64'>
            >>> array[ak.num(array) > 1, 1]
            <Array [2.2, 5.5, 8.8] type='3 * float64'>

        It's sometimes also a problem that "cleaning" the dataset by dropping
        empty lists changes its alignment, so that it can no longer be used
        in calculations with "uncleaned" data. For this, #ak.mask can be
        useful because it inserts None in positions that fail the filter,
        rather than removing them.

            >>> print(ak.mask(array, ak.num(array) > 1))
            [[1.1, 2.2, 3.3], None, [4.4, 5.5], None, None, [7.7, 8.8, 9.9]]

        Note, however, that the `0` or `1` to pick the first or second
        item of each nested list is in the second dimension, so the first
        dimension of the slice must be a `:`.

            >>> ak.mask(array, ak.num(array) > 1)[:, 0]
            <Array [1.1, None, 4.4, None, None, 7.7] type='6 * ?float64'>
            >>> ak.mask(array, ak.num(array) > 1)[:, 1]
            <Array [2.2, None, 5.5, None, None, 8.8] type='6 * ?float64'>

        Another syntax for

            ak.mask(array, array_of_booleans)

        is

            array.mask[array_of_booleans]

        (which is 5 characters away from simply filtering the `array`).

        Projection
        **********

        The following `array`

            ak.Array([[{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}],
                      [{"x": 3.3, "y": [3, 3, 3]}],
                      [{"x": 0, "y": []}, {"x": 1.1, "y": [1, 1, 1]}]])

        has records inside of nested lists:

            >>> ak.type(array)
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
        None in Python, are called option types (#ak.types.OptionType) in
        Awkward Array and can be used as a slice.

        For example, an `array` like

            ak.Array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])

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

            ak.Array([[[0.0, 1.1, 2.2], [], [3.3, 4.4]], [], [[5.5]]])

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

        When applying a nested index with missing (None) entries at levels
        higher than the last level, the indexer must have the same dimension
        as the array being indexed, and the resulting output will have missing
        entries at the corresponding locations, e.g. for

            >>> print(array[ [[[0, None, 2, None, None], None, [1]], None, [[0]]] ])
            [[[0, None, 2.2, None, None], None, [4.4]], None, [[5.5]]]

        the sub-list at entry 0,0 is extended as the masked entries are
        acting at the last level, while the higher levels of the indexer all
        have the same dimension as the array being indexed.
        """
        return awkward1._util.wrap(
            self._layout[where], self._behavior, cache=self._cache
        )

    def __setitem__(self, where, what):
        """
        Args:
            where (str): Field name to add to records in the array.
            what (#ak.Array): Array to add as the new field.

        Unlike #__getitem__, which allows a wide variety of slice types,
        only single field-slicing is supported for assignment.
        (#ak.layout.Content arrays are immutable; field assignment replaces
        the #layout with an array that has the new field using #ak.with_field.)

        However, a field can be assigned deeply into a nested record e.g.

            >>> nested = ak.zip({"a" : ak.zip({"x" : [1, 2, 3]})})
            >>> nested["a", "y"] = 2 * nested.a.x
            >>> ak.to_list(nested)
            [{'a': {'x': 1, 'y': 2}}, {'a': {'x': 2, 'y': 4}}, {'a': {'x': 3, 'y': 6}}]

        Note that the following does **not** work:

            >>> nested["a"]["y"] = 2 * nested.a.x # does not work, nested["a"] is a copy!

        Always assign by passing the whole path to the top level

            >>> nested["a", "y"] = 2 * nested.a.x

        If necessary, the new field will be broadcasted to fit the array.
        For example, given an `array` like

            ak.Array([[{"x": 1.1}, {"x": 2.2}, {"x": 3.3}], [], [{"x": 4.4}, {"x": 5.5}]])

        which has three elements with nested data in each, assigning

            >>> array["y"] = [100, 200, 300]

        will result in

            >>> ak.to_list(array)
            [[{'x': 1.1, 'y': 100}, {'x': 2.2, 'y': 100}, {'x': 3.3, 'y': 100}],
             [],
             [{'x': 4.4, 'y': 300}, {'x': 5.5, 'y': 300}]]

        because the `100` in `what[0]` is broadcasted to all three nested
        elements of `array[0]`, the `200` in `what[1]` is broadcasted to the
        empty list `array[1]`, and the `300` in `what[2]` is broadcasted to
        both elements of `array[2]`.

        See #ak.with_field for a variant that does not change the #ak.Array
        in-place. (Internally, this method uses #ak.with_field, so performance
        is not a factor in choosing one over the other.)
        """
        if not (
            isinstance(where, str)
            or (isinstance(where, tuple) and all(isinstance(x, str) for x in where))
        ):
            raise TypeError(
                "only fields may be assigned in-place (by field name)"
                + awkward1._util.exception_suffix(__file__)
            )
        self._layout = awkward1.operations.structure.with_field(
            self._layout, what, where
        ).layout
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

        Note that while fields can be accessed as attributes, they cannot be
        *assigned* as attributes: the following doesn't work.

            array.z = new_field

        Always use

            array["z"] = new_field

        to add a field.
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
                        "occurred:\n{1}: {2}".format(repr(where), type(err), str(err))
                        + awkward1._util.exception_suffix(__file__)
                    )
            else:
                raise AttributeError(
                    "no field named {0}".format(repr(where))
                    + awkward1._util.exception_suffix(__file__)
                )

    def __dir__(self):
        """
        Lists all methods, properties, and field names (see #__getattr__)
        that can be accessed as attributes.
        """
        return sorted(
            set(
                dir(super(Array, self))
                + [
                    x
                    for x in self._layout.keys()
                    if _dir_pattern.match(x) and not keyword.iskeyword(x)
                ]
            )
        )

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

    def __str__(self):
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

        See #ak.to_list and #ak.to_json to convert whole Arrays into Python
        data or JSON strings without loss (except for #type).
        """
        return self._str()

    def __repr__(self):
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
        return self._repr()

    def _str(self, limit_value=85):
        import awkward1.operations.structure

        layout = awkward1.operations.structure.with_cache(
            self._layout, {}, chain="last", highlevel=False
        )
        return awkward1._util.minimally_touching_string(
            limit_value, layout, self._behavior
        )

    def _repr(self, limit_value=40, limit_total=85):
        import awkward1.operations.structure

        layout = awkward1.operations.structure.with_cache(
            self._layout, {}, chain="last", highlevel=False
        )
        value = awkward1._util.minimally_touching_string(
            limit_value, layout, self._behavior
        )

        try:
            name = super(Array, self).__getattribute__("__name__")
        except AttributeError:
            name = type(self).__name__
        limit_type = limit_total - (len(value) + len(name) + len("<  type=>"))
        typestr = repr(str(awkward1._util.highlevel_type(layout, self._behavior, True)))
        if len(typestr) > limit_type:
            typestr = typestr[: (limit_type - 4)] + "..." + typestr[-1]

        return "<{0} {1} type={2}>".format(name, value, typestr)

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
        """
        if awkward1._util.called_by_module(
            "pandas.io.formats.format"
        ) or awkward1._util.called_by_module("pandas.core.generic"):
            out = numpy.empty(len(self._layout), dtype="O")
            for i, x in enumerate(self._layout):
                out[i] = awkward1._util.wrap(x, self._behavior)
            return out
        elif awkward1._util.called_by_module("pandas"):
            try:
                return awkward1._connect._numpy.convert_to_array(
                    self._layout, args, kwargs
                )
            except Exception:
                out = numpy.empty(len(self._layout), dtype="O")
                for i, x in enumerate(self._layout):
                    out[i] = awkward1._util.wrap(x, self._behavior)
                return out
        else:
            return awkward1._connect._numpy.convert_to_array(self._layout, args, kwargs)

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

        In the above example, `array` is a nested list of records and `2` is
        a scalar. Awkward Array applies the same broadcasting rules as NumPy
        plus a few more to deal with nested structures. In addition to
        broadcasting a scalar, as above, it is possible to broadcast
        arrays with less depth into arrays with more depth, such as

            >>> print(array + ak.Array([10, 20, 30]))
            [[{x: 10, y: []}, {x: 11.1, y: [11]}], [], [{x: 32.2, y: [32, 32]}]]

        See #ak.broadcast_arrays for details about broadcasting and the
        generalized set of broadcasting rules.

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
        return awkward1._connect._numpy.array_ufunc(ufunc, method, inputs, kwargs)

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
        return awkward1._connect._numpy.array_function(func, types, args, kwargs)

    @property
    def numba_type(self):
        """
        The type of this Array when it is used in Numba. It contains enough
        information to generate low-level code for accessing any element,
        down to the leaves.

        See [Numba documentation](https://numba.pydata.org/numba-doc/dev/reference/types.html)
        on types and signatures.
        """
        import awkward1._connect._numba

        awkward1._connect._numba.register_and_check()
        if self._numbaview is None:
            self._numbaview = awkward1._connect._numba.arrayview.ArrayView.fromarray(
                self
            )
        import numba

        return numba.typeof(self._numbaview)

    def __getstate__(self):
        form, container, num_partitions = awkward1.to_arrayset(self)
        if self._behavior is awkward1.behavior:
            behavior = None
        else:
            behavior = self._behavior
        return form, container, num_partitions, behavior

    def __setstate__(self, state):
        form, container, num_partitions, behavior = state
        layout = awkward1.from_arrayset(
            form, container, num_partitions, highlevel=False
        )
        if self.__class__ is Array:
            self.__class__ = awkward1._util.arrayclass(layout, behavior)
        self.layout = layout
        self.behavior = behavior
        self.cache = None


class Record(awkward1._connect._numpy.NDArrayOperatorsMixin):
    """
    Args:
        data (#ak.layout.Record, #ak.Record, str, or dict):
            Data to wrap or convert into a record.
            If a string, the data are assumed to be JSON.
            If a dict, calls #ak.from_iter, which assumes all inner
            dimensions have irregular lengths.
        behavior (None or dict): Custom #ak.behavior for this Record only.
        with_name (None or str): Gives the record type a name that can be
            used to override its behavior (see below).
        check_valid (bool): If True, verify that the #layout is valid.
        cache (None or MutableMapping): Stores data for any
            #ak.layout.VirtualArray nodes that this Array might contain.
            Persists through `__getitem__` but not any other operations.

    High-level record that can contain fields of any type.

    Most users won't be creating Records manually. This class primarily exists
    to be overridden in the same way as #ak.Array.

    Records can be used in [Numba](http://numba.pydata.org/): they can be
    passed as arguments to a Numba-compiled function or returned as return
    values. The only limitation is that they cannot be *created*
    inside the Numba-compiled function; to make outputs, consider
    #ak.ArrayBuilder.

    See also #ak.Array and #ak.behavior.
    """

    def __init__(
        self, data, behavior=None, with_name=None, check_valid=False, cache=None
    ):
        if isinstance(data, awkward1.layout.Record):
            layout = data

        elif isinstance(data, Record):
            layout = data.layout

        elif isinstance(data, str):
            layout = awkward1.operations.convert.from_json(data, highlevel=False)

        elif isinstance(data, dict):
            layout = awkward1.operations.convert.from_iter([data], highlevel=False)[0]

        elif isinstance(data, Iterable):
            raise TypeError(
                "could not convert non-dict into an awkward1.Record; try awkward1.Array"
                + awkward1._util.exception_suffix(__file__)
            )

        else:
            layout = None

        if not isinstance(layout, awkward1.layout.Record):
            raise TypeError(
                "could not convert data into an awkward1.Record"
                + awkward1._util.exception_suffix(__file__)
            )

        if self.__class__ is Record:
            self.__class__ = awkward1._util.recordclass(layout, behavior)

        if with_name is not None:
            layout = awkward1.operations.structure.with_name(
                layout, with_name, highlevel=False
            )

        self.layout = layout
        self.behavior = behavior
        docstr = self.layout.purelist_parameter("__doc__")
        if isinstance(docstr, str):
            self.__doc__ = docstr
        if check_valid:
            awkward1.operations.describe.validity_error(self, exception=True)

        self.cache = cache

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
                "layout must be a subclass of awkward1.layout.Record"
                + awkward1._util.exception_suffix(__file__)
            )

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
            raise TypeError(
                "behavior must be None or a dict"
                + awkward1._util.exception_suffix(__file__)
            )

    @property
    def cache(self):
        return self._cache

    @cache.setter
    def cache(self, value):
        if value is None or isinstance(value, MutableMapping):
            self._cache = value
        else:
            raise TypeError(
                "cache must be None or a MutableMapping"
                + awkward1._util.exception_suffix(__file__)
            )

    def tolist(self):
        """
        Converts this Record into Python objects.

        Awkward Array types have the following Pythonic translations.

           * #ak.types.PrimitiveType: converted into bool, int, float.
           * #ak.types.OptionType: missing values are converted into None.
           * #ak.types.ListType: converted into list.
           * #ak.types.RegularType: also converted into list. Python (and JSON)
             forms lose information about the regularity of list lengths.
           * #ak.types.ListType with parameter `"__array__"` equal to
             `"__bytestring__"`: converted into bytes.
           * #ak.types.ListType with parameter `"__array__"` equal to
             `"__string__"`: converted into str.
           * #ak.types.RecordArray without field names: converted into tuple.
           * #ak.types.RecordArray with field names: converted into dict.
           * #ak.types.UnionArray: Python data are naturally heterogeneous.

        See also #ak.to_list and #ak.from_iter.
        """
        return awkward1.operations.convert.to_list(self)

    def tojson(
        self, destination=None, pretty=False, maxdecimals=None, buffersize=65536
    ):
        """
        Args:
            destination (None or str): If None, this method returns a JSON str;
                if a str, it uses that as a file name and writes (overwrites)
                that file (returning None).
            pretty (bool): If True, indent the output for human readability; if
                False, output compact JSON without spaces.
            maxdecimals (None or int): If an int, limit the number of
                floating-point decimals to this number; if None, write all
                digits.
            buffersize (int): Size (in bytes) of the buffer used by the JSON
                parser.

        Converts this Record into a JSON string or file.

        Awkward Array types have the following JSON translations.

           * #ak.types.PrimitiveType: converted into JSON booleans and numbers.
           * #ak.types.OptionType: missing values are converted into None.
           * #ak.types.ListType: converted into JSON lists.
           * #ak.types.RegularType: also converted into JSON lists. JSON (and
             Python) forms lose information about the regularity of list
             lengths.
           * #ak.types.ListType with parameter `"__array__"` equal to
             `"__bytestring__"` or `"__string__"`: converted into JSON strings.
           * #ak.types.RecordArray without field names: converted into JSON
             objects with numbers as strings for keys.
           * #ak.types.RecordArray with field names: converted into JSON
             objects.
           * #ak.types.UnionArray: JSON data are naturally heterogeneous.

        See also #ak.to_json and #ak.from_json.
        """
        return awkward1.operations.convert.to_json(
            self, destination, pretty, maxdecimals, buffersize
        )

    @property
    def nbytes(self):
        """
        The total number of bytes in all the #ak.layout.Index,
        #ak.layout.Identities, and #ak.layout.NumpyArray buffers in this
        array tree.

        Note: this calculation takes overlapping buffers into account, to the
        extent that overlaps are not double-counted, but overlaps are currently
        assumed to be complete subsets of one another, and so it is
        theoretically possible (though unlikely) that this number is an
        underestimate of the true usage.

        It also does not count buffers that must be kept in memory because
        of ownership, but are not directly used in the array. Nor does it count
        the (small) C++ nodes or Python objects that reference the (large)
        array buffers.
        """
        return self._layout.nbytes

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
        return awkward1._util.wrap(
            self._layout[where], self._behavior, cache=self._cache
        )

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

        See #ak.with_field for a variant that does not change the #ak.Record
        in-place. (Internally, this method uses #ak.with_field, so performance
        is not a factor in choosing one over the other.)
        """
        if not (
            isinstance(where, str)
            or (isinstance(where, tuple) and all(isinstance(x, str) for x in where))
        ):
            raise TypeError(
                "only fields may be assigned in-place (by field name)"
                + awkward1._util.exception_suffix(__file__)
            )
        self._layout = awkward1.operations.structure.with_field(
            self._layout, what, where
        ).layout
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
                        "occurred:\n{1}: {2}".format(repr(where), type(err), str(err))
                        + awkward1._util.exception_suffix(__file__)
                    )
            else:
                raise AttributeError(
                    "no field named {0}".format(repr(where))
                    + awkward1._util.exception_suffix(__file__)
                )

    def __dir__(self):
        """
        Lists all methods, properties, and field names (see #__getattr__)
        that can be accessed as attributes.
        """
        return sorted(
            set(
                dir(super(Record, self))
                + [
                    x
                    for x in self._layout.keys()
                    if _dir_pattern.match(x) and not keyword.iskeyword(x)
                ]
            )
        )

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

    def __str__(self):
        """
        Args:
            limit_value (int): Maximum number of characters to use when
                presenting the Record as a string.

        Presents this Record as a string without type or `"<Record ...>"`.

        See #ak.Array.__str__ for a more complete description.
        """
        return self._str()

    def __repr__(self):
        """
        Args:
            limit_value (int): Maximum number of characters to use when
                presenting the data of the Record.
            limit_total (int): Maximum number of characters to use for
                the whole string (should be larger than `limit_value`).

        Presents this Record as a string with its type and `"<Record ...>"`.

        See #ak.Array.__repr__ for a more complete description.
        """
        return self._repr()

    def _str(self, limit_value=85):
        import awkward1.operations.structure

        layout = awkward1.operations.structure.with_cache(
            self._layout, {}, chain="last", highlevel=False
        )
        return awkward1._util.minimally_touching_string(
            limit_value + 2, layout, self._behavior
        )[1:-1]

    def _repr(self, limit_value=40, limit_total=85):
        import awkward1.operations.structure

        layout = awkward1.operations.structure.with_cache(
            self._layout, {}, chain="last", highlevel=False
        )
        value = awkward1._util.minimally_touching_string(
            limit_value + 2, layout, self._behavior
        )[1:-1]

        try:
            name = super(Record, self).__getattribute__("__name__")
        except AttributeError:
            name = type(self).__name__
        limit_type = limit_total - (len(value) + len(name) + len("<  type=>"))
        typestr = repr(
            str(awkward1._util.highlevel_type(layout, self._behavior, False))
        )
        if len(typestr) > limit_type:
            typestr = typestr[: (limit_type - 4)] + "..." + typestr[-1]

        return "<{0} {1} type={2}>".format(name, value, typestr)

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
        return awkward1._connect._numpy.array_ufunc(ufunc, method, inputs, kwargs)

    @property
    def numba_type(self):
        """
        The type of this Record when it is used in Numba. It contains enough
        information to generate low-level code for accessing any element,
        down to the leaves.

        See [Numba documentation](https://numba.pydata.org/numba-doc/dev/reference/types.html)
        on types and signatures.
        """
        import awkward1._connect._numba

        awkward1._connect._numba.register_and_check()
        if self._numbaview is None:
            self._numbaview = awkward1._connect._numba.arrayview.RecordView.fromrecord(
                self
            )
        import numba

        return numba.typeof(self._numbaview)

    def __getstate__(self):
        form, container, num_partitions = awkward1.to_arrayset(self._layout.array)
        if self._behavior is awkward1.behavior:
            behavior = None
        else:
            behavior = self._behavior
        return form, container, num_partitions, behavior, self._layout.at

    def __setstate__(self, state):
        form, container, num_partitions, behavior, at = state
        array = awkward1.from_arrayset(
            form, container, num_partitions, highlevel=False
        )
        layout = awkward1.layout.Record(array, at)
        if self.__class__ is Record:
            self.__class__ = awkward1._util.recordclass(layout, behavior)
        self.layout = layout
        self.behavior = behavior
        self.cache = None


class ArrayBuilder(Iterable, Sized):
    """
    Args:
        behavior (None or dict): Custom #ak.behavior for arrays built by
            this ArrayBuilder.
        initial (int): Initial size (in bytes) of buffers used by
            #ak.layout.ArrayBuilder (see #ak.layout.ArrayBuilderOptions).
        resize (float): Resize multiplier for buffers used by
            #ak.layout.ArrayBuilder (see #ak.layout.ArrayBuilderOptions);
            should be strictly greater than 1.

    General tool for building arrays of nested data structures from a sequence
    of commands. Most data types can be constructed by calling commands in the
    right order, similar to printing tokens to construct JSON output.

    To illustrate how this works, consider the following example.

        b = ak.ArrayBuilder()

        # fill commands   # as JSON   # current array type
        ##########################################################################################
        b.begin_list()    # [         # 0 * var * unknown     (initially, the type is unknown)
        b.integer(1)      #   1,      # 0 * var * int64
        b.integer(2)      #   2,      # 0 * var * int64
        b.real(3)         #   3.0     # 0 * var * float64     (all the integers have become floats)
        b.end_list()      # ],        # 1 * var * float64
        b.begin_list()    # [         # 1 * var * float64
        b.end_list()      # ],        # 2 * var * float64
        b.begin_list()    # [         # 2 * var * float64
        b.integer(4)      #   4,      # 2 * var * float64
        b.null()          #   null,   # 2 * var * ?float64    (now the floats are nullable)
        b.integer(5)      #   5       # 2 * var * ?float64
        b.end_list()      # ],        # 3 * var * ?float64
        b.begin_list()    # [         # 3 * var * ?float64
        b.begin_record()  #   {       # 3 * var * ?union[float64, {}]
        b.field("x")      #     "x":  # 3 * var * ?union[float64, {"x": unknown}]
        b.integer(1)      #      1,   # 3 * var * ?union[float64, {"x": int64}]
        b.field("y")      #      "y": # 3 * var * ?union[float64, {"x": int64, "y": unknown}]
        b.begin_list()    #      [    # 3 * var * ?union[float64, {"x": int64, "y": var * unknown}]
        b.integer(2)      #        2, # 3 * var * ?union[float64, {"x": int64, "y": var * int64}]
        b.integer(3)      #        3  # 3 * var * ?union[float64, {"x": int64, "y": var * int64}]
        b.end_list()      #      ]    # 3 * var * ?union[float64, {"x": int64, "y": var * int64}]
        b.end_record()    #   }       # 3 * var * ?union[float64, {"x": int64, "y": var * int64}]
        b.end_list()      # ]         # 4 * var * ?union[float64, {"x": int64, "y": var * int64}]

    To get an array, we take a #snapshot of the ArrayBuilder's current state.

        >>> ak.to_list(b.snapshot())
        [[1.0, 2.0, 3.0], [], [4.0, None, 5.0], [{'x': 1, 'y': [2, 3]}]]

    The full set of filling commands is the following.

       * #null: appends a None value.
       * #boolean: appends True or False.
       * #integer: appends an integer.
       * #real: appends a floating-point value.
       * #bytestring: appends an unencoded string (raw bytes).
       * #string: appends a UTF-8 encoded string.
       * #begin_list: begins filling a list; must be closed with #end_list.
       * #end_list: ends a list.
       * #begin_tuple: begins filling a tuple; must be closed with #end_tuple.
       * #index: selects a tuple slot to fill; must be followed by a command
         that actually fills that slot.
       * #end_tuple: ends a tuple.
       * #begin_record: begins filling a record; must be closed with
         #end_record.
       * #field: selects a record field to fill; must be followed by a command
         that actually fills that field.
       * #end_record: ends a record.
       * #append: generic method for filling #null, #boolean, #integer, #real,
         #bytestring, #string, #ak.Array, #ak.Record, or arbitrary Python data.
         When filling from #ak.Array or #ak.Record, the output holds references
         to the original data, rather than copying.
       * #extend: appends all the items from an #ak.Array (by reference).
       * #list: context manager for #begin_list and #end_list.
       * #tuple: context manager for #begin_tuple and #end_tuple.
       * #record: context manager for #begin_record and #end_record.

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
        ...         builder.begin_list()
        ...         for i in range(np.random.poisson(3)):
        ...             deepnesting(builder, probability**2)
        ...         builder.end_list()
        ...
        >>> builder = ak.ArrayBuilder()
        >>> deepnesting(builder, 0.9)
        >>> builder.snapshot()
        <Array [... 1.23, -0.498, 0.272], -0.0519]]]] type='1 * var * var * union[var * ...'>
        >>> ak.to_list(builder)
        [[[[2.052949634260401, 0.9522057655747124], [[[0.2560810133948006], 1.8668954120287653, 0.8933700720920406, 0.31709173110067773], 0.38515995466456676, -1.6259655150460695, [[0.18211022402412927], 0.46592679548320143, 0.39275072293709223], [-0.572569956850481, 1.3991748897028693, -0.15414122174138611, -0.20008742443379549]], [[[-0.7410750761192828, -0.34455689325781347], -0.8446675414135969], [-0.8139112572198548, -0.7250728258598154, -0.42851563653684244, [1.0498296931855706, 1.6969612860075955, -0.18093559189614564, 1.078608791657082]]], [[0.5172670690419124]]], [[-1.9731106633939228, 0.5778640337060391], [-1.2488533773832633, -2.1458066486349434, -0.5439318468515132, [[0.2419441207503176, -2.313974422156488, [-0.6811651539055098, 0.08323572953509818], 1.801261721511669, 0.16653718365329456], -0.6348811801078983, [0.016350096268563003, [-1.2867920376687112, 0.38205295881313484, 1.4093210810506318, -0.2698869943849985, -0.48804922126979045]]], -0.6297773736098737, -2.5333506573111424], [-1.6680144776019314, 0.5862818687707498]], [0.6266171347177766, [[-0.7660737060966999, -0.677432480564727, -1.1527197837522167], -0.5025371508398492, [0.3610998752041169, 0.4811870365139723, -0.8030689233086394, [1.1538103888031122, -1.0955905747145644], -1.3980944016010062, 1.2822990047991039]], 0.939566155023095, [1.3581048298505891, [0.36949478822799947, 1.096666130135532, -0.2769024331557954, -0.7993215902675834], [-0.4103823967097248], [0.6789480075462166, 0.8991579880810466, 0.7900472554969632]], [], [0.6772644918729233, [-0.48385354748861575, -0.39154812719778437], 1.069329510451712, 0.8057750827838897, -0.3440192823735095], [[1.5687828887524105, -1.6086288847970498, [-0.6907842744344904], -0.42627155869364414], 0.33605387861917574, -0.7329513818714791, 0.5040026160756554, -1.2529377572694538, -1.1566264096307166], [[0.6407540268295862], [-0.017540252205401917], -0.9530971110439417], [[0.41643810453893765, -0.682997865214066, 0.7930286671567052], 0.5142103949393788]], [[0.6271004836147108, [0.5895664560584991, -0.7563863809912544]], [1.6176958047983054, 0.5226854288884638, 0.24149248202497436], -1.0912185170716135, [-1.1122535648683918], 0.22727974012353094], [-0.4161362684360263, [[0.4234696267033054], 0.7866791657813567, [1.225201951430818, -0.49790730839958713, 0.2715010029532568], -0.051866117232298316]]]]
        >>> ak.type(builder.snapshot())
        1 * var * var * union[var * union[float64, var * union[var * union[float64, var * float64], float64]], float64]

    Note that this is a *general* method for building arrays; if the type is
    known in advance, more specialized procedures can be faster. This should
    be considered the "least effort" approach.
    """

    def __init__(self, behavior=None, initial=1024, resize=1.5):
        self._layout = awkward1.layout.ArrayBuilder(initial=initial, resize=resize)
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
            raise TypeError(
                "behavior must be None or a dict"
                + awkward1._util.exception_suffix(__file__)
            )

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

    def __str__(self):
        """
        Args:
            limit_value (int): Maximum number of characters to use when
                presenting the ArrayBuilder as a string.

        Presents this ArrayBuilder as a string without type or
        `"<ArrayBuilder ...>"`.

        See #ak.Array.__str__ for a more complete description.
        """
        return self._str()

    def __repr__(self):
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
        return self._repr()

    def _str(self, limit_value=85, snapshot=None):
        if snapshot is None:
            snapshot = self.snapshot()
        return snapshot._str(limit_value=limit_value)

    def _repr(self, limit_value=40, limit_total=85):
        snapshot = self.snapshot()
        value = self._str(limit_value=limit_value, snapshot=snapshot)

        limit_type = limit_total - len(value) - len("<ArrayBuilder  type=>")
        typestrs = awkward1._util.typestrs(self._behavior)
        typestr = repr(str(snapshot.layout.type(typestrs)))
        if len(typestr) > limit_type:
            typestr = typestr[: (limit_type - 4)] + "..." + typestr[-1]

        return "<ArrayBuilder {0} type={1}>".format(value, typestr)

    def __array__(self, *args, **kwargs):
        """
        Intercepts attempts to convert a #snapshot of this array into a
        NumPy array and either performs a zero-copy conversion or raises
        an error.

        See #ak.Array.__array__ for a more complete description.
        """
        return awkward1._connect._numpy.convert_to_array(self.snapshot(), args, kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Intercepts attempts to pass this ArrayBuilder to a NumPy
        [universal functions](https://docs.scipy.org/doc/numpy/reference/ufuncs.html)
        (ufuncs) and passes it through the structure of the array's #snapshot.

        See #ak.Array.__array_ufunc__ for a more complete description.
        """
        return awkward1._connect._numpy.array_ufunc(ufunc, method, inputs, kwargs)

    def __array_function__(self, func, types, args, kwargs):
        """
        Intercepts attempts to pass this ArrayBuilder to those NumPy functions
        other than universal functions that have an Awkward equivalent.

        See #ak.ArrayBuilder.__array_ufunc__ for a more complete description.
        """
        return awkward1._connect._numpy.array_function(func, types, args, kwargs)

    @property
    def numba_type(self):
        """
        The type of this Array when it is used in Numba. It contains enough
        information to generate low-level code for accessing any element,
        down to the leaves.

        See [Numba documentation](https://numba.pydata.org/numba-doc/dev/reference/types.html)
        on types and signatures.
        """
        import awkward1._connect._numba

        awkward1._connect._numba.register_and_check()
        import awkward1._connect._numba.builder

        return awkward1._connect._numba.builder.ArrayBuilderType(self._behavior)

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

    def begin_list(self):
        """
        Begins filling a list; must be closed with #end_list.

        For example,

            builder.begin_list()
            builder.real(1.1)
            builder.real(2.2)
            builder.real(3.3)
            builder.end_list()
            builder.begin_list()
            builder.end_list()
            builder.begin_list()
            builder.real(4.4)
            builder.real(5.5)
            builder.end_list()

        produces

            [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
        """
        self._layout.beginlist()

    def end_list(self):
        """
        Ends a list.
        """
        self._layout.endlist()

    def begin_tuple(self, numfields):
        """
        Begins filling a tuple with `numfields` fields; must be closed with
        #end_tuple.

        For example,

            builder.begin_tuple(3)
            builder.index(0).integer(1)
            builder.index(1).real(1.1)
            builder.index(2).string("one")
            builder.end_tuple()
            builder.begin_tuple(3)
            builder.index(0).integer(2)
            builder.index(1).real(2.2)
            builder.index(2).string("two")
            builder.end_tuple()

        produces

            [(1, 1.1, "one"), (2, 2.2, "two")]
        """
        self._layout.begintuple(numfields)

    def index(self, i):
        """
        Args:
            i (int): The tuple slot to fill.

        This method also returns the #ak.ArrayBuilder, so that it can be
        chained with the value that fills the slot.

        Prepares to fill a tuple slot; see #begin_tuple for an example.
        """
        self._layout.index(i)
        return self

    def end_tuple(self):
        """
        Ends a tuple.
        """
        self._layout.endtuple()

    def begin_record(self, name=None):
        """
        Begins filling a record with an optional `name`; must be closed with
        #end_record.

        For example,

            >>> builder = ak.ArrayBuilder()
            >>> builder.begin_record("points")
            >>> builder.field("x").real(1)
            >>> builder.field("y").real(1.1)
            >>> builder.end_record()
            >>> builder.begin_record("points")
            >>> builder.field("x").real(2)
            >>> builder.field("y").real(2.2)
            >>> builder.end_record()

        produces

            >>> ak.to_list(builder.snapshot())
            [{"x": 1.0, "y": 1.1}, {"x": 2.0, "y": 2.2}]

        with type

            >>> ak.type(builder.snapshot())
            2 * points["x": float64, "y": float64]

        The record type is named `"points"` because its `"__record__"`
        parameter is set to that value:

            >>> builder.snapshot().layout.parameters
            {'__record__': 'points'}

        The `"__record__"` parameter can be used to add behavior to the records
        in the array, as described in #ak.Array, #ak.Record, and #ak.behavior.
        """
        self._layout.beginrecord(name)

    def field(self, key):
        """
        Args:
            key (str): The field key to fill.

        This method also returns the #ak.ArrayBuilder, so that it can be
        chained with the value that fills the slot.

        Prepares to fill a field; see #begin_record for an example.
        """
        self._layout.field(key)
        return self

    def end_record(self):
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
        #ak.from_iter except that it fills an existing #ak.ArrayBuilder,
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
                    "'obj' is an ak.Array"
                    + awkward1._util.exception_suffix(__file__)
                )

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
            raise TypeError(
                "'extend' method requires an ak.Array"
                + awkward1._util.exception_suffix(__file__)
            )

    class _Nested(object):
        def __init__(self, arraybuilder):
            self._arraybuilder = arraybuilder

        def __repr__(self, limit_value=40, limit_total=85):
            snapshot = self._arraybuilder.snapshot()
            value = self._arraybuilder._str(
                limit_value=limit_value, snapshot=snapshot
            )

            limit_type = (
                limit_total
                - len(value)
                - len("<ArrayBuilder.  type=>")
                - len(self._name)
            )
            typestrs = awkward1._util.typestrs(self._arraybuilder._behavior)
            typestr = repr(str(snapshot.layout.type(typestrs)))
            if len(typestr) > limit_type:
                typestr = typestr[: (limit_type - 4)] + "..." + typestr[-1]

            return "<ArrayBuilder.{0} {1} type={2}>".format(self._name, value, typestr)

    class List(_Nested):
        _name = "list"

        def __enter__(self):
            self._arraybuilder.begin_list()

        def __exit__(self, type, value, traceback):
            self._arraybuilder.end_list()

    def list(self):
        """
        Context manager to prevent unpaired #begin_list and #end_list. The
        example in the #begin_list documentation can be rewritten as

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
        _name = "tuple"

        def __init__(self, arraybuilder, numfields):
            super(ArrayBuilder.Tuple, self).__init__(arraybuilder)
            self._numfields = numfields

        def __enter__(self):
            self._arraybuilder.begin_tuple(self._numfields)

        def __exit__(self, type, value, traceback):
            self._arraybuilder.end_tuple()

    def tuple(self, numfields):
        """
        Context manager to prevent unpaired #begin_tuple and #end_tuple. The
        example in the #begin_tuple documentation can be rewritten as

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
        _name = "record"

        def __init__(self, arraybuilder, name):
            super(ArrayBuilder.Record, self).__init__(arraybuilder)
            self._name = name

        def __enter__(self):
            self._arraybuilder.begin_record(name=self._name)

        def __exit__(self, type, value, traceback):
            self._arraybuilder.end_record()

    def record(self, name=None):
        """
        Context manager to prevent unpaired #begin_record and #end_record. The
        example in the #begin_record documentation can be rewritten as

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
