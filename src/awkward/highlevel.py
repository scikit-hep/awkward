# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import copy
import html
import io
import itertools
import keyword
import re
import sys
from collections.abc import Iterable, Mapping, Sized

from awkward_cpp.lib import _ext

import awkward as ak
from awkward._connect.numpy import NDArrayOperatorsMixin

np = ak._nplikes.NumpyMetadata.instance()
numpy = ak._nplikes.Numpy.instance()

_dir_pattern = re.compile(r"^[a-zA-Z_]\w*$")


class Array(NDArrayOperatorsMixin, Iterable, Sized):
    """
    Args:
        data (#ak.contents.Content, #ak.Array, `np.ndarray`, `cp.ndarray`, `pyarrow.*`, str, dict, or iterable):
            Data to wrap or convert into an array.
               - If a NumPy array, the regularity of its dimensions is preserved
                 and the data are viewed, not copied.
               - CuPy arrays are treated the same way as NumPy arrays except that
                 they default to `backend="cuda"`, rather than `backend="cpu"`.
               - If a pyarrow object, calls #ak.from_arrow, preserving as much
                 metadata as possible, usually zero-copy.
               - If a dict of str \u2192 columns, combines the columns into an
                 array of records (like Pandas's DataFrame constructor).
               - If a string, the data are assumed to be JSON.
               - If an iterable, calls #ak.from_iter, which assumes all dimensions
                 have irregular lengths.
        behavior (None or dict): Custom #ak.behavior for this Array only.
        with_name (None or str): Gives tuples and records a name that can be
            used to override their behavior (see below).
        check_valid (bool): If True, verify that the #layout is valid.
        backend (None, `"cpu"`, `"jax"`, `"cuda"`): If `"cpu"`, the Array will be placed in
            main memory for use with other `"cpu"` Arrays and Records; if `"cuda"`,
            the Array will be placed in GPU global memory using CUDA; if `"jax"`, the structure
            is copied to the CPU for use with JAX. if None, the `data` are left untouched.

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

    The #ak.to_dataframe function performs this conversion; if `how=None`, it
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

    JAX
    ********

    Derivatives of a calculation on an #ak.Array (s) can be calculated with
    [JAX](https://github.com/google/jax#readme), but only if the array
    functions in `ak` / `numpy` are used, not the functions in the `jax`
    library directly (apart from e.g. `jax.grad`).

    Like NumPy ufuncs, the function and its derivatives are evaluated on the
    numeric leaves of the data structure, maintaining structure in the output.
    """

    def __init__(
        self,
        data,
        *,
        behavior=None,
        with_name=None,
        check_valid=False,
        backend=None,
    ):
        if isinstance(data, ak.contents.Content):
            layout = data

        elif isinstance(data, Array):
            layout = data._layout
            behavior = ak._util.behavior_of(data, behavior=behavior)

        elif numpy.is_own_array(data) and data.dtype != np.dtype("O"):
            layout = ak.operations.from_numpy(data, highlevel=False)

        elif ak._nplikes.Cupy.is_own_array(data):
            layout = ak.operations.from_cupy(data, highlevel=False)

        elif ak._nplikes.Jax.is_own_array(data):
            layout = ak.operations.from_jax(data, highlevel=False)

        elif ak._util.in_module(data, "pyarrow"):
            layout = ak.operations.from_arrow(data, highlevel=False)

        elif isinstance(data, dict):
            fields = []
            contents = []
            length = None
            for k, v in data.items():
                fields.append(k)
                contents.append(Array(v).layout)
                if length is None:
                    length = len(contents[-1])
                elif length != len(contents[-1]):
                    raise ak._errors.wrap_error(
                        ValueError(
                            "dict of arrays in ak.Array constructor must have arrays "
                            "of equal length ({} vs {})".format(
                                length, len(contents[-1])
                            )
                        )
                    )
            layout = ak.contents.RecordArray(contents, fields)

        elif isinstance(data, str):
            layout = ak.operations.from_json(data, highlevel=False)

        else:
            layout = ak.operations.from_iter(data, highlevel=False, allow_record=False)

        if not isinstance(layout, ak.contents.Content):
            raise ak._errors.wrap_error(
                TypeError("could not convert data into an ak.Array")
            )

        if with_name is not None:
            layout = ak.operations.with_name(
                layout, with_name, highlevel=False, behavior=behavior
            )

        if backend is not None and backend != ak.operations.backend(layout):
            layout = ak.operations.to_backend(layout, backend, highlevel=False)

        self.layout = layout
        self.behavior = behavior

        docstr = layout.purelist_parameter("__doc__")
        if isinstance(docstr, str):
            self.__doc__ = docstr

        if check_valid:
            ak.operations.validity_error(self, exception=True)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        ak.jax.register_behavior_class(cls)

    @property
    def layout(self):
        """
        The composable #ak.contents.Content elements that determine how this
        Array is structured.

        This may be considered a "low-level" view, as it distinguishes between
        arrays that have the same logical meaning (i.e. same JSON output and
        high-level #type) but different

        * node types, such as #ak.contents.ListArray and
             #ak.contents.ListOffsetArray,
        * integer type specialization, such as `int64` vs `int32`
        * or specific values, such as gaps in a #ak.contents.ListArray.

        The #ak.contents.Content elements are fully composable, whereas an
        Array is not; the high-level Array is a single-layer "shell" around
        its layout.

        Layouts are rendered as XML instead of a nested list. For example,
        the following `array`

            ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])

        is presented as

            <Array [[1.1, 2.2, 3.3], [], [4.4, 5.5]] type='3 * var * float64'>

        but `array.layout` is presented as

            <ListOffsetArray len='3'>
                <offsets><Index dtype='int64' len='4'>
                    [0 3 3 5]
                </Index></offsets>
                <content>
                    <NumpyArray dtype='float64' len='5'>[1.1 2.2 3.3 4.4 5.5]</NumpyArray>
                </content>
            </ListOffsetArray>

        (with truncation for large arrays).
        """
        return self._layout

    @layout.setter
    def layout(self, layout):
        if isinstance(layout, ak.contents.Content):
            self._layout = layout
            self._numbaview = None
        else:
            raise ak._errors.wrap_error(
                TypeError("layout must be a subclass of ak.contents.Content")
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
        if behavior is None or isinstance(behavior, Mapping):
            self.__class__ = ak._util.arrayclass(self._layout, behavior)
            self._behavior = behavior
        else:
            raise ak._errors.wrap_error(TypeError("behavior must be None or a dict"))

    class Mask:
        def __init__(self, array):
            self._array = array

        def __getitem__(self, where):
            with ak._errors.OperationErrorContext(
                "ak.Array.mask", {0: self._array, 1: where}
            ):
                return ak.operations.mask(self._array, where, valid_when=True)

    @property
    def mask(self):
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
        return self.Mask(self)

    def tolist(self):
        """
        Converts this Array into Python objects; same as #ak.to_list
        (but without the underscore, like NumPy's
        [tolist](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.tolist.html)).
        """
        return self.to_list()

    def to_list(self):
        """
        Converts this Array into Python objects; same as #ak.to_list.
        """
        return self._layout.to_list(self._behavior)

    def to_numpy(self, allow_missing=True):
        """
        Converts this Array into a NumPy array, if possible; same as #ak.to_numpy.
        """
        return ak.operations.to_numpy(self, allow_missing=allow_missing)

    @property
    def nbytes(self):
        """
        The total number of bytes in all the #ak.index.Index,
        and #ak.contents.NumpyArray buffers in this array tree.

        It does not count buffers that must be kept in memory because
        of ownership, but are not directly used in the array. Nor does it count
        the (small) Python objects that reference the (large) array buffers.
        """
        return self._layout.nbytes

    @property
    def ndim(self):
        """
        Number of dimensions (nested variable-length lists and/or regular arrays)
        before reaching a numeric type or a record.

        There may be nested lists within the record, as field values, but this
        number of dimensions does not count those.

        (Some fields may have different depths than others, which is why they
        are not counted.)
        """
        return self._layout.purelist_depth

    @property
    def fields(self):
        """
        List of field names or tuple slot numbers (as strings) of the outermost
        record or tuple in this array.

        If the array contains nested records, only the fields of the outermost
        record are shown. If it contains tuples instead of records, its fields
        are string representations of integers, such as `"0"`, `"1"`, `"2"`, etc.
        The records or tuples may be within multiple layers of nested lists.

        If the array contains neither tuples nor records, it is an empty list.

        See also #ak.fields.
        """
        return self._layout.fields

    @property
    def is_tuple(self):
        """
        If True, the top-most record structure has no named fields, i.e. it's a tuple.
        """
        return self._layout.is_tuple

    def _ipython_key_completions_(self):
        return self._layout.fields

    @property
    def type(self):
        """
        The high-level type of this Array; same as #ak.type.

        Note that the outermost element of an Array's type is always an
        #ak.types.ArrayType, which specifies the number of elements in the array.

        The type of a #ak.contents.Content (from #ak.Array.layout) is not
        wrapped by an #ak.types.ArrayType.
        """
        return ak.types.ArrayType(
            self._layout.form.type_from_behavior(self._behavior), self._layout.length
        )

    @property
    def typestr(self):
        """
        The high-level type of this Array, presented as a string.

        Note that the outermost element of an Array's type is always an
        #ak.types.ArrayType, which specifies the number of elements in the array.

        The type of a #ak.contents.Content (from #ak.Array.layout) is not
        wrapped by an #ak.types.ArrayType.
        """
        return str(self.type)

    def __len__(self):
        """
        The length of this Array, only counting the outermost structure.

        For example, the length of

            ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])

        is `3`, not `5`.
        """
        return self._layout.length

    def __iter__(self):
        """
        Iterates over this Array in Python.

        Note that this is the *slowest* way to access data (even slower than
        native Python objects, like lists and dicts). Usually, you should
        express your problems in array-at-a-time operations.

        In other words, do this:

            >>> np.sqrt(ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]))
            <Array [[1.05, 1.48, 1.82], [], [2.1, 2.35]] type='3 * var * float64'>

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
        if isinstance(self._layout, ak.contents.NumpyArray):
            array = self._layout._raw(numpy)
            array_param = self._layout.parameter("__array__")
            if array_param == "byte":
                for x in ak._util.tobytes(array):
                    yield x
            elif array_param == "char":
                for x in ak._util.tobytes(array).decode(errors="surrogateescape"):
                    yield x
            else:
                for x in array:
                    yield x
        else:
            for x in self._layout:
                if isinstance(x, ak.contents.NumpyArray):
                    array_param = x.parameter("__array__")
                    if array_param == "byte":
                        yield ak._util.tobytes(x._raw(numpy))
                    elif array_param == "char":
                        yield ak._util.tobytes(x._raw(numpy)).decode(
                            errors="surrogateescape"
                        )
                    else:
                        yield ak._util.wrap(x, self._behavior)
                elif isinstance(x, (ak.contents.Content, ak.record.Record)):
                    yield ak._util.wrap(x, self._behavior)
                else:
                    yield x

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
        ([combining advanced and basic indexing](https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing)
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
          [field access](https://numpy.org/doc/stable/user/basics.indexing.html#field-access),
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
        [NumPy documentation](https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing).

        Filtering
        *********

        A common use of selection by boolean arrays is to filter a dataset by
        some property. For instance, to get the odd values of

            >>> array = ak.Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        one can put an array expression with True for each odd value inside
        square brackets:

            >>> array[array % 2 == 1]
            <Array [1, 3, 5, 7, 9] type='5 * int64'>

        This technique is so common in NumPy and Pandas data analysis that it
        is often read as a syntax, rather than a consequence of array slicing.

        The extension to nested arrays like

            >>> array = ak.Array([[[0, 1, 2], [], [3, 4], [5]], [[6, 7, 8], [9]]])

        allows us to use the same syntax more generally.

            >>> array[array % 2 == 1]
            <Array [[[1], [], [3], [5]], [[7], [9]]] type='2 * var * var * int64'>

        In this example, the boolean array is itself nested (see
        <<nested indexing>> below).

            >>> array % 2 == 1
            <Array [[[False, True, False], ..., [True]], ...] type='2 * var * var * bool'>

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

            >>> ak.mask(array, ak.num(array) > 1)
            <Array [[1.1, 2.2, 3.3], ..., [7.7, ..., 9.9]] type='6 * option[var * float64]'>

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

        The following

            >>> array = ak.Array([[{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}],
            ...                   [{"x": 3.3, "y": [3, 3, 3]}],
            ...                   [{"x": 0, "y": []}, {"x": 1.1, "y": [1, 1, 1]}]])

        has records inside of nested lists:

            >>> array.type.show()
            3 * var * {
                x: float64,
                y: var * int64
            }

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
            IndexError: while attempting to slice
                <Array [[{x: 1.1, y: [1]}, {...}], ...] type='3 * var * {x: float64, y:...'>
            with
                (0, :, 0, 'y')
            at inner NumpyArray of length 2, using sub-slice (0).

        because the `array[0, :, 0, ...]` slice applies to both `"x"` and
        `"y"` before `"y"` is selected, and `"x"` is a one-dimensional
        NumpyArray that can't take more than its share of slices.

        Finally, note that the dot (`__getattr__`) syntax is equivalent to a single
        string in a slice (`__getitem__`) if the field name is a valid Python
        identifier and doesn't conflict with #ak.Array methods or properties.

            >>> array.x
            <Array [[1.1, 2.2], [3.3], [0, 1.1]] type='3 * var * float64'>
            >>> array.y
            <Array [[[1], [2, 2]], ..., [[], [1, ...]]] type='3 * var * var * int64'>

        Nested Projection
        *****************

        If records are nested within records, you can use a series of strings in
        the selector to drill down. For instance, with the following

            >>> array = ak.Array([
            ...     {"a": {"x": 1, "y": 2}, "b": {"x": 10, "y": 20}, "c": {"x": 1.1, "y": 2.2}},
            ...     {"a": {"x": 1, "y": 2}, "b": {"x": 10, "y": 20}, "c": {"x": 1.1, "y": 2.2}},
            ...     {"a": {"x": 1, "y": 2}, "b": {"x": 10, "y": 20}, "c": {"x": 1.1, "y": 2.2}}])

        we can go directly to the numerical data by specifying a string for the
        outer field and a string for the inner field.

            >>> array["a", "x"]
            <Array [1, 1, 1] type='3 * int64'>
            >>> array["a", "y"]
            <Array [2, 2, 2] type='3 * int64'>
            >>> array["b", "y"]
            <Array [20, 20, 20] type='3 * int64'>
            >>> array["c", "y"]
            <Array [2.2, 2.2, 2.2] type='3 * float64'>

        As with single projections, the dot (`__getattr__`) syntax is equivalent
        to a single string in a slice (`__getitem__`) if the field name is a valid
        Python identifier and doesn't conflict with #ak.Array methods or properties.

            >>> array.a.x
            <Array [1, 1, 1] type='3 * int64'>

        You can even get every field of the same name within an outer record using
        a list of field names for the outer record. The following selects the `"x"`
        field of `"a"`, `"b"`, and `"c"` records:

            >>> array[["a", "b", "c"], "x"].show()
            [{a: 1, b: 10, c: 1.1},
             {a: 1, b: 10, c: 1.1},
             {a: 1, b: 10, c: 1.1}]

        You don't need to get all fields:

            >>> array[["a", "b"], "x"].show()
            [{a: 1, b: 10},
             {a: 1, b: 10},
             {a: 1, b: 10}]

        And you can select lists of field names at all levels:

            >>> array[["a", "b"], ["x", "y"]].show()
            [{a: {x: 1, y: 2}, b: {x: 10, y: 20}},
             {a: {x: 1, y: 2}, b: {x: 10, y: 20}},
             {a: {x: 1, y: 2}, b: {x: 10, y: 20}}]

        Option indexing
        ***************

        NumPy arrays can be sliced by all of the above slice types except
        arrays with missing values and arrays with nested lists, both of
        which are inexpressible in NumPy. Missing values, represented by
        None in Python, are called option types (#ak.types.OptionType) in
        Awkward Array and can be used as a slice.

        For example,

            >>> array = ak.Array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])

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

        For example,

            >>> array = ak.Array([[[0.0, 1.1, 2.2], [], [3.3, 4.4]], [], [[5.5]]])

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

            >>> ((array * 10) % 2 == 1).show()
            [[[False, True, False], [], [True, False]],
             [],
             [[True]]]
            >>> (array[(array * 10) % 2 == 1]).show()
            [[[1.1], [], [3.3]],
             [],
             [[5.5]]]

        Functions whose names start with "arg" return index positions, which
        can be used with the integer form.

            >>> np.argmax(array, axis=-1).show()
            [[2, None, 1],
             [],
             [0]]
            >>> array[np.argmax(array, axis=-1)].show()
            [[[3.3, 4.4], None, []],
             [],
             [[5.5]]]

        Here, the `np.argmax` returns the integer position of the maximum
        element or None for empty arrays. It's a nice example of
        <<option indexing>> with <<nested indexing>>.

        When applying a nested index with missing (None) entries at levels
        higher than the last level, the indexer must have the same dimension
        as the array being indexed, and the resulting output will have missing
        entries at the corresponding locations, e.g. for

            >>> array[ [[[0, None, 2, None, None], None, [1]], None, [[0]]] ].show()
            [[[0, None, 2.2, None, None], None, [4.4]],
             None,
             [[5.5]]]

        the sub-list at entry 0,0 is extended as the masked entries are
        acting at the last level, while the higher levels of the indexer all
        have the same dimension as the array being indexed.
        """
        with ak._errors.SlicingErrorContext(self, where):
            out = self._layout[where]
            if isinstance(out, ak.contents.NumpyArray):
                array_param = out.parameter("__array__")
                if array_param == "byte":
                    return ak._util.tobytes(out._raw(numpy))
                elif array_param == "char":
                    return ak._util.tobytes(out._raw(numpy)).decode(
                        errors="surrogateescape"
                    )
            if isinstance(out, (ak.contents.Content, ak.record.Record)):
                return ak._util.wrap(out, self._behavior)
            else:
                return out

    def __setitem__(self, where, what):
        """
        Args:
            where (str or tuple of str): Field name to add to records in the array.
            what (#ak.Array): Array to add as the new field.

        Unlike #__getitem__, which allows a wide variety of slice types,
        only single field-slicing is supported for assignment.
        (#ak.contents.Content arrays are immutable; field assignment replaces
        the #layout with an array that has the new field using #ak.with_field.)

        However, a field can be assigned deeply into a nested record e.g.

            >>> nested = ak.zip({"a" : ak.zip({"x" : [1, 2, 3]})})
            >>> nested["a", "y"] = 2 * nested.a.x
            >>> nested.show()
            [{a: {x: 1, y: 2}},
             {a: {x: 2, y: 4}},
             {a: {x: 3, y: 6}}]

        Note that the following does **not** work:

            >>> nested["a"]["y"] = 2 * nested.a.x # does not work, nested["a"] is a copy!

        Always assign by passing the whole path to the top level

            >>> nested["a", "y"] = 2 * nested.a.x

        If necessary, the new field will be broadcasted to fit the array.
        For example, given

            >>> array = ak.Array([
            ...     [{"x": 1.1}, {"x": 2.2}, {"x": 3.3}], [], [{"x": 4.4}, {"x": 5.5}]
            ... ])

        which has three elements with nested data in each, assigning

            >>> array["y"] = [100, 200, 300]

        will result in

            >>> array.show()
            [[{x: 1.1, y: 100}, {x: 2.2, y: 100}, {x: 3.3, y: 100}],
             [],
             [{x: 4.4, y: 300}, {x: 5.5, y: 300}]]

        because the `100` in `what[0]` is broadcasted to all three nested
        elements of `array[0]`, the `200` in `what[1]` is broadcasted to the
        empty list `array[1]`, and the `300` in `what[2]` is broadcasted to
        both elements of `array[2]`.

        See #ak.with_field for a variant that does not change the #ak.Array
        in-place. (Internally, this method uses #ak.with_field, so performance
        is not a factor in choosing one over the other.)
        """
        with ak._errors.OperationErrorContext(
            "ak.Array.__setitem__",
            dict(self=self, field_name=where, field_value=what),
        ):
            if not (
                isinstance(where, str)
                or (isinstance(where, tuple) and all(isinstance(x, str) for x in where))
            ):
                raise ak._errors.wrap_error(
                    TypeError("only fields may be assigned in-place (by field name)")
                )

            self._layout = ak.operations.with_field(
                self._layout, what, where, highlevel=False
            )
            self._numbaview = None

    def __delitem__(self, where):
        """
        Args:
            where (str or tuple of str): Field name to remove from the array.

        For example:

            >>> array = ak.Array([{"x": 3.3, "y": {"this": 10, "that": 20}}])
            >>> del array["y", "that"]
            >>> array.show()
            [{x: 3.3, y: {this: 10}}]

        See #ak.without_field for a variant that does not change the #ak.Array
        in-place. (Internally, this method uses #ak.without_field, so performance
        is not a factor in choosing one over the other.)
        """
        with ak._errors.OperationErrorContext(
            "ak.Array.__delitem__",
            dict(self=self, field_name=where),
        ):
            if not (
                isinstance(where, str)
                or (isinstance(where, tuple) and all(isinstance(x, str) for x in where))
            ):
                raise ak._errors.wrap_error(
                    TypeError("only fields may be removed in-place (by field name)")
                )

            self._layout = ak.operations.ak_without_field._impl(
                self._layout, where, highlevel=False, behavior=self._behavior
            )
            self._numbaview = None

    def __getattr__(self, where):
        """
        Args:
            where (str): Attribute name to lookup

        Whenever possible, fields can be accessed as attributes.

        For example, the fields of

            >>> array = ak.Array([
            ...     [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}, {"x": 3.3, "y": [3, 3, 3]}],
            ...     [],
            ...     [{"x": 4.4, "y": [4, 4, 4, 4]}, {"x": 5.5, "y": [5, 5, 5, 5, 5]}]
            ... ])

        can be accessed as

            >>> array.x
            <Array [[1.1, 2.2, 3.3], [], [4.4, 5.5]] type='3 * var * float64'>
            >>> array.y
            <Array [[[1], [2, 2], [3, 3, 3]], [], [...]] type='3 * var * var * int64'>

        which are equivalent to `array["x"]` and `array["y"]`. (See
        <<projection>>.)

        Fields can't be accessed as attributes when

        * #ak.Array methods or properties take precedence,
        * a domain-specific behavior has methods or properties that take
             precedence, or
        * the field name is not a valid Python identifier or is a Python
             keyword.

        Note that while fields can be accessed as attributes, they cannot be
        *assigned* as attributes. See #ak.Array.__setitem__ for more.
        """
        if hasattr(type(self), where):
            return super().__getattribute__(where)
        else:
            if where in self._layout.fields:
                try:
                    return self[where]
                except Exception as err:
                    raise ak._errors.wrap_error(
                        AttributeError(
                            "while trying to get field {}, an exception "
                            "occurred:\n{}: {}".format(repr(where), type(err), str(err))
                        )
                    ) from err
            else:
                raise ak._errors.wrap_error(AttributeError(f"no field named {where!r}"))

    def __setattr__(self, name, value):
        """
        Args:
            where (str): Attribute name to set

        Set an attribute on the array.

        Only existing public attributes e.g. #ak.Array.layout, or private
        attributes (with leading underscores), can be set.

        Fields are not assignable to as attributes, i.e. the following doesn't work:

            array.z = new_field

        Instead, always use #ak.Array.__setitem__:

            array["z"] = new_field

        or #ak.with_field:

            array = ak.with_field(array, new_field, "z")

        to add or modify a field.
        """
        if name.startswith("_") or hasattr(type(self), name):
            super().__setattr__(name, value)
        elif name in self._layout.fields:
            raise ak._errors.wrap_error(
                AttributeError(
                    "fields cannot be set as attributes. use #__setitem__ or #ak.with_field"
                )
            )
        else:
            raise ak._errors.wrap_error(
                AttributeError(
                    "only private attributes (started with an underscore) can be set on arrays"
                )
            )

    def __dir__(self):
        """
        Lists all methods, properties, and field names (see #__getattr__)
        that can be accessed as attributes.
        """
        return sorted(
            set(
                [x for x in dir(type(self)) if not x.startswith("_")]
                + dir(super())
                + [
                    x
                    for x in self._layout.fields
                    if _dir_pattern.match(x) and not keyword.iskeyword(x)
                ]
            )
        )

    def __str__(self):
        import awkward._prettyprint

        return awkward._prettyprint.valuestr(self, 1, 80)

    def __repr__(self):
        return self._repr(80)

    def _repr(self, limit_cols):
        import awkward._prettyprint

        try:
            pytype = super().__getattribute__("__name__")
        except AttributeError:
            pytype = type(self).__name__

        if (
            self._layout.backend.nplike.known_shape
            and self._layout.backend.nplike.known_data
        ):
            typestr = repr(str(self.type))[1:-1]
            if len(typestr) + len(pytype) + len(" type=''") + 3 < limit_cols // 2:
                strwidth = limit_cols - (
                    len(typestr) + len(pytype) + len(" type=''") + 3
                )
            else:
                strwidth = max(
                    0,
                    min(
                        limit_cols // 2,
                        limit_cols - len(pytype) - len(" type='...'") - 3,
                    ),
                )
            valuestr = " " + awkward._prettyprint.valuestr(self, 1, strwidth)

        else:
            self._layout._touch_data(recursive=True)
            typestr = repr(
                "?? * " + str(self._layout.form.type_from_behavior(self._behavior))
            )[1:-1]
            valuestr = "-typetracer"

        length = max(3, limit_cols - len(pytype) - len("type='...'") - len(valuestr))
        if len(typestr) > length:
            typestr = "'" + typestr[: length - 3] + "...'"
        else:
            typestr = "'" + typestr + "'"

        return f"<{pytype}{valuestr} type={typestr}>"

    def show(self, limit_rows=20, limit_cols=80, type=False, stream=sys.stdout):
        """
        Args:
            limit_rows (int): Maximum number of rows (lines) to use in the output.
            limit_cols (int): Maximum number of columns (characters wide).
            type (bool): If True, print the type as well. (Doesn't count toward number
                of rows/lines limit.)
            stream (object with a ``write(str)`` method or None): Stream to write the
                output to. If None, return a string instead of writing to a stream.

        Display the contents of the array within `limit_rows` and `limit_cols`, using
        ellipsis (`...`) for hidden nested data.
        """
        import awkward._prettyprint

        valuestr = awkward._prettyprint.valuestr(self, limit_rows, limit_cols)
        if type:
            tmp = io.StringIO()
            self.type.show(stream=tmp)
            out = "type: " + tmp.getvalue() + valuestr
        else:
            out = valuestr

        if stream is None:
            return out
        else:
            stream.write(out + "\n")

    def _repr_mimebundle_(self, include=None, exclude=None):
        value_buff = io.StringIO()
        self.show(type=False, stream=value_buff)
        header_lines = value_buff.getvalue().splitlines()

        type_buff = io.StringIO()
        self.type.show(stream=type_buff)
        footer_lines = type_buff.getvalue().splitlines()
        # Prepend a `type: ` prefix to the type information
        footer_lines[0] = f"type: {footer_lines[0]}"

        if header_lines[-1] == "":
            del header_lines[-1]

        n_cols = max(len(line) for line in itertools.chain(header_lines, footer_lines))
        body = "\n".join([*header_lines, "-" * n_cols, *footer_lines])

        return {
            "text/html": f"<pre>{html.escape(body)}</pre>",
            "text/plain": repr(self),
        }

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
        arguments = {0: self}
        for i, arg in enumerate(args):
            arguments[i + 1] = arg
        arguments.update(kwargs)
        with ak._errors.OperationErrorContext("numpy.asarray", arguments):
            return ak._connect.numpy.convert_to_array(self._layout, args, kwargs)

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

        For example, with

            >>> array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])

        applying `np.sqrt` would yield

            >>> np.sqrt(array).show()
            [[1.05, 1.48, 1.82],
             [],
             [2.1, 2.35]]

        In addition, many unary and binary operators implicitly call ufuncs,
        such as `np.power` in

            >>> (array**2).show()
            [[1.21, 4.84, 10.9],
             [],
             [19.4, 30.2]]

        In the above example, `array` is a nested list of records and `2` is
        a scalar. Awkward Array applies the same broadcasting rules as NumPy
        plus a few more to deal with nested structures. In addition to
        broadcasting a scalar, as above, it is possible to broadcast
        arrays with less depth into arrays with more depth, such as

            >>> (array + ak.Array([10, 20, 30])).show()
            [[11.1, 12.2, 13.3],
             [],
             [34.4, 35.5]]

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
            >>> sqr(array).show()
            [[1.21, 4.84, 10.9],
             [],
             [19.4, 30.2]]

        See also #__array_function__.
        """
        name = f"{type(ufunc).__module__}.{ufunc.__name__}.{str(method)}"
        arguments = {}
        for i, arg in enumerate(inputs):
            arguments[i] = arg
        arguments.update(kwargs)
        with ak._errors.OperationErrorContext(name, arguments):
            return ak._connect.numpy.array_ufunc(ufunc, method, inputs, kwargs)

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
        return ak._connect.numpy.array_function(
            func, types, args, kwargs, behavior=self._behavior
        )

    @property
    def numba_type(self):
        """
        The type of this Array when it is used in Numba. It contains enough
        information to generate low-level code for accessing any element,
        down to the leaves.

        See [Numba documentation](https://numba.pydata.org/numba-doc/dev/reference/types.html)
        on types and signatures.
        """
        ak.numba.register_and_check()

        if self._numbaview is None:
            self._numbaview = ak._connect.numba.arrayview.ArrayView.fromarray(self)
        import numba

        return numba.typeof(self._numbaview)

    def __getstate__(self):
        packed = ak.operations.to_packed(self._layout, highlevel=False)
        form, length, container = ak.operations.to_buffers(
            packed, buffer_key="{form_key}-{attribute}", form_key="node{id}"
        )
        if self._behavior is ak.behavior:
            behavior = None
        else:
            behavior = self._behavior
        return form.to_dict(), length, container, behavior

    def __setstate__(self, state):
        form, length, container, behavior = state
        layout = ak.operations.from_buffers(
            form,
            length,
            container,
            highlevel=False,
            buffer_key="{form_key}-{attribute}",
        )
        self.layout = layout
        self.behavior = behavior

    def __copy__(self):
        return Array(self._layout, behavior=self._behavior)

    def __deepcopy__(self, memo):
        return Array(
            copy.deepcopy(self._layout, memo),
            behavior=copy.deepcopy(self._behavior, memo),
        )

    def __bool__(self):
        if len(self) == 1:
            return bool(self[0])
        else:
            raise ak._errors.wrap_error(
                ValueError(
                    "the truth value of an array whose length is not 1 is ambiguous; "
                    "use ak.any() or ak.all()"
                )
            )


class Record(NDArrayOperatorsMixin):
    """
    Args:
        data (#ak.record.Record, #ak.Record, str, or dict):
            Data to wrap or convert into a record.
            If a string, the data are assumed to be JSON.
            If a dict, calls #ak.from_iter, which assumes all inner
            dimensions have irregular lengths.
        behavior (None or dict): Custom #ak.behavior for this Record only.
        with_name (None or str): Gives the record type a name that can be
            used to override its behavior (see below).
        check_valid (bool): If True, verify that the #layout is valid.
        backend (None, `"cpu"`, `"jax"`, `"cuda"`): If `"cpu"`, the Array will be placed in
            main memory for use with other `"cpu"` Arrays and Records; if `"cuda"`,
            the Array will be placed in GPU global memory using CUDA; if `"jax"`, the structure
            is copied to the CPU for use with JAX. if None, the `data` are left untouched.

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
        self,
        data,
        *,
        behavior=None,
        with_name=None,
        check_valid=False,
        library=None,
    ):
        if isinstance(data, ak.record.Record):
            layout = data

        elif isinstance(data, Record):
            layout = data._layout

        elif isinstance(data, str):
            layout = ak.operations.from_json(data, highlevel=False)

        elif isinstance(data, dict):
            fields = []
            contents = []
            for k, v in data.items():
                fields.append(k)
                if ak._util.is_non_string_iterable(v):
                    contents.append(Array(v).layout[np.newaxis])
                else:
                    contents.append(Array([v]).layout)

            layout = ak.record.Record(ak.contents.RecordArray(contents, fields), at=0)

        elif isinstance(data, Iterable):
            raise ak._errors.wrap_error(
                TypeError("could not convert non-dict into an ak.Record; try ak.Array")
            )

        else:
            layout = None

        if not isinstance(layout, ak.record.Record):
            raise ak._errors.wrap_error(
                TypeError("could not convert data into an ak.Record")
            )

        if with_name is not None:
            layout = ak.operations.with_name(layout, with_name, highlevel=False)

        if library is not None and library != ak.operations.library(layout):
            layout = ak.operations.to_library(layout, library, highlevel=False)

        self.layout = layout
        self.behavior = behavior

        docstr = layout.purelist_parameter("__doc__")
        if isinstance(docstr, str):
            self.__doc__ = docstr

        if check_valid:
            ak.operations.validity_error(self, exception=True)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        ak.jax.register_behavior_class(cls)

    @property
    def layout(self):
        """
        The #ak.record.Record that contains composable #ak.contents.Content
        elements to determine how the array is structured.

        See #ak.Array.layout for a more complete description.

        The #ak.record.Record is not a subclass of #ak.contents.Content in
        Python and it is not composable with them: #ak.record.Record contains
        one #ak.contents.RecordArray (which is a #ak.contents.Content), but
        #ak.contents.Content nodes cannot contain a #ak.record.Record.

        A #ak.record.Record is not an independent entity from its
        #ak.contents.RecordArray; it's really just a marker indicating which
        element to select. The XML representation reflects that:

            >>> vectors = ak.Array([{"x": 0.1, "y": 1.0, "z": 30.0},
            ...                     {"x": 0.2, "y": 2.0, "z": 20.0},
            ...                     {"x": 0.3, "y": 3.0, "z": 10.0}])
            >>> vectors[1].layout
            <Record at='1'>
                <array><RecordArray is_tuple='false' len='3'>
                    <content index='0' field='x'>
                        <NumpyArray dtype='float64' len='3'>[0.1 0.2 0.3]</NumpyArray>
                    </content>
                    <content index='1' field='y'>
                        <NumpyArray dtype='float64' len='3'>[1. 2. 3.]</NumpyArray>
                    </content>
                    <content index='2' field='z'>
                        <NumpyArray dtype='float64' len='3'>[30. 20. 10.]</NumpyArray>
                    </content>
                </RecordArray></array>
            </Record>
        """
        return self._layout

    @layout.setter
    def layout(self, layout):
        if isinstance(layout, ak.record.Record):
            self._layout = layout
            self._numbaview = None
        else:
            raise ak._errors.wrap_error(
                TypeError("layout must be a subclass of ak.record.Record")
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
        if behavior is None or isinstance(behavior, Mapping):
            self.__class__ = ak._util.recordclass(self._layout, behavior)
            self._behavior = behavior
        else:
            raise ak._errors.wrap_error(TypeError("behavior must be None or a dict"))

    def tolist(self):
        """
        Converts this Record into Python objects; same as #ak.to_list
        (but without the underscore, like NumPy's
        [tolist](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.tolist.html)).
        """
        return self.to_list()

    def to_list(self):
        """
        Converts this Record into Python objects; same as #ak.to_list.
        """
        return self._layout.to_list(self._behavior)

    @property
    def nbytes(self):
        """
        The total number of bytes in all the #ak.index.Index,
        and #ak.contents.NumpyArray buffers in this array tree.

        It does not count buffers that must be kept in memory because
        of ownership, but are not directly used in the array. Nor does it count
        the (small) Python objects that reference the (large)
        array buffers.
        """
        return self._layout.nbytes

    @property
    def fields(self):
        """
        List of field names or tuple slot numbers (as strings) of this record.

        If this is actually a tuple its fields are string representations of
        integers, such as `"0"`, `"1"`, `"2"`, etc.

        See also #ak.fields.
        """
        return self._layout.array.fields

    @property
    def is_tuple(self):
        """
        If True, the top-most record structure has no named fields, i.e. it's a tuple.
        """
        return self._layout.array.is_tuple

    def _ipython_key_completions_(self):
        return self._layout.array.fields

    # Disable iteration (and prevent old-style iteration from being used)
    __iter__ = None

    @property
    def type(self):
        """
        The high-level type of this Record; same as #ak.type.

        Note that the outermost element of a Record's type is always a
        #ak.types.RecordType.
        """
        return self._layout.array.form.type_from_behavior(self._behavior)

    @property
    def typestr(self):
        """
        The high-level type of this Record, presented as a string.

        Note that the outermost element of a Record's type is always a
        #ak.types.RecordType.
        """
        return str(self.type)

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

        For example, with

            >>> record = ak.Record({"x": 3.3, "y": [1, 2, 3]})

        we can select

            >>> record["x"]
            3.3
            >>> record["y"]
            <Array [1, 2, 3] type='3 * int64'>
            >>> record["y", 1]
            2
        """
        with ak._errors.SlicingErrorContext(self, where):
            out = self._layout[where]
            if isinstance(out, ak.contents.NumpyArray):
                array_param = out.parameter("__array__")
                if array_param == "byte":
                    return ak._util.tobytes(out._raw(numpy))
                elif array_param == "char":
                    return ak._util.tobytes(out._raw(numpy)).decode(
                        errors="surrogateescape"
                    )
            if isinstance(out, (ak.contents.Content, ak.record.Record)):
                return ak._util.wrap(out, self._behavior)
            else:
                return out

    def __setitem__(self, where, what):
        """
        Args:
            where (str or tuple of str): Field name to add data to the record.
            what: Data to add as the new field.

        For example:

            >>> record = ak.Record({"x": 3.3})
            >>> record["y"] = 4
            >>> record["z"] = {"another": "record"}
            >>> record.show()
            {x: 3.3,
             y: 4,
             z: {another: 'record'}}

        See #ak.with_field for a variant that does not change the #ak.Record
        in-place. (Internally, this method uses #ak.with_field, so performance
        is not a factor in choosing one over the other.)
        """
        with ak._errors.OperationErrorContext(
            "ak.Record.__setitem__",
            dict(self=self, field_name=where, field_value=what),
        ):
            if not (
                isinstance(where, str)
                or (isinstance(where, tuple) and all(isinstance(x, str) for x in where))
            ):
                raise ak._errors.wrap_error(
                    TypeError("only fields may be assigned in-place (by field name)")
                )

            self._layout = ak.operations.ak_with_field._impl(
                self._layout, what, where, highlevel=False, behavior=self._behavior
            )
            self._numbaview = None

    def __delitem__(self, where):
        """
        Args:
            where (str or tuple of str): Field name to remove from the record.

        For example:

            >>> record = ak.Record({"x": 3.3, "y": {"this": 10, "that": 20}})
            >>> del record["y", "that"]
            >>> record.show()
            {x: 3.3,
             y: {this: 10}}

        See #ak.without_field for a variant that does not change the #ak.Record
        in-place. (Internally, this method uses #ak.without_field, so performance
        is not a factor in choosing one over the other.)
        """
        with ak._errors.OperationErrorContext(
            "ak.Record.__delitem__",
            dict(self=self, field_name=where),
        ):
            if not (
                isinstance(where, str)
                or (isinstance(where, tuple) and all(isinstance(x, str) for x in where))
            ):
                raise ak._errors.wrap_error(
                    TypeError("only fields may be removed in-place (by field name)")
                )

            self._layout = ak.operations.ak_without_field._impl(
                self._layout, where, highlevel=False, behavior=self._behavior
            )
            self._numbaview = None

    def __getattr__(self, where):
        """
        Whenever possible, fields can be accessed as attributes.

        For example, the fields of

            >>> record = ak.Record({"x": 1.1, "y": [2, 2], "z": "three"})

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
        if hasattr(type(self), where):
            return super().__getattribute__(where)
        else:
            if where in self._layout.fields:
                try:
                    return self[where]
                except Exception as err:
                    raise ak._errors.wrap_error(
                        AttributeError(
                            "while trying to get field {}, an exception "
                            "occurred:\n{}: {}".format(repr(where), type(err), str(err))
                        )
                    ) from err
            else:
                raise ak._errors.wrap_error(AttributeError(f"no field named {where!r}"))

    def __setattr__(self, name, value):
        """
        Args:
            where (str): Attribute name to set

        Set an attribute on the record.

        Only existing public attributes e.g. #ak.Record.layout, or private
        attributes (with leading underscores), can be set.

        Fields are not assignable to as attributes, i.e. the following doesn't work:

            record.z = new_field

        Instead, always use #ak.Record.__setitem__:

            record["z"] = new_field

        or #ak.with_field:

            record = ak.with_field(record, new_field, "z")

        to add or modify a field.
        """
        if name.startswith("_") or hasattr(type(self), name):
            super().__setattr__(name, value)
        elif name in self._layout.fields:
            raise ak._errors.wrap_error(
                AttributeError(
                    "fields cannot be set as attributes. use #__setitem__ or #ak.with_field"
                )
            )
        else:
            raise ak._errors.wrap_error(
                AttributeError(
                    "only private attributes (started with an underscore) can be set on records"
                )
            )

    def __dir__(self):
        """
        Lists all methods, properties, and field names (see #__getattr__)
        that can be accessed as attributes.
        """
        return sorted(
            set(
                [x for x in dir(type(self)) if not x.startswith("_")]
                + dir(super())
                + [
                    x
                    for x in self._layout.fields
                    if _dir_pattern.match(x) and not keyword.iskeyword(x)
                ]
            )
        )

    def __str__(self):
        import awkward._prettyprint

        return awkward._prettyprint.valuestr(self, 1, 80)

    def __repr__(self):
        return self._repr(80)

    def _repr(self, limit_cols):
        import awkward._prettyprint

        pytype = type(self).__name__

        if (
            self._layout.array.backend.nplike.known_shape
            and self._layout.array.backend.nplike.known_data
        ):
            typestr = repr(str(self.type))[1:-1]
            if len(typestr) + len(pytype) + len(" type=''") + 3 < limit_cols // 2:
                strwidth = limit_cols - (
                    len(typestr) + len(pytype) + len(" type=''") + 3
                )
            else:
                strwidth = max(
                    0,
                    min(
                        limit_cols // 2,
                        limit_cols - len(pytype) - len(" type='...'") - 3,
                    ),
                )
            valuestr = " " + awkward._prettyprint.valuestr(self, 1, strwidth)

        else:
            self._layout._touch_data(recursive=True)
            typestr = repr(str(self._layout.form.type_from_behavior(self._behavior)))[
                1:-1
            ]
            valuestr = "-typetracer"

        length = max(3, limit_cols - len(pytype) - len("type='...'") - len(valuestr))
        if len(typestr) > length:
            typestr = "'" + typestr[: length - 3] + "...'"
        else:
            typestr = "'" + typestr + "'"

        return f"<{pytype}{valuestr} type={typestr}>"

    def show(self, limit_rows=20, limit_cols=80, type=False, stream=sys.stdout):
        """
        Args:
            limit_rows (int): Maximum number of rows (lines) to use in the output.
            limit_cols (int): Maximum number of columns (characters wide).
            type (bool): If True, print the type as well. (Doesn't count toward number
                of rows/lines limit.)
            stream (object with a ``write(str)`` method or None): Stream to write the
                output to. If None, return a string instead of writing to a stream.

        Display the contents of the record within `limit_rows` and `limit_cols`, using
        ellipsis (`...`) for hidden nested data.
        """
        import awkward._prettyprint

        valuestr = awkward._prettyprint.valuestr(self, limit_rows, limit_cols)
        if type:
            tmp = io.StringIO()
            self.type.show(stream=tmp)
            out = "type: " + tmp.getvalue() + valuestr
        else:
            out = valuestr

        if stream is None:
            return out
        else:
            stream.write(out + "\n")

    def _repr_mimebundle_(self, include=None, exclude=None):
        value_buff = io.StringIO()
        self.show(type=False, stream=value_buff)
        header_lines = value_buff.getvalue().splitlines()

        type_buff = io.StringIO()
        self.type.show(stream=type_buff)
        footer_lines = type_buff.getvalue().splitlines()
        # Prepend a `type: ` prefix to the type information
        footer_lines[0] = f"type: {footer_lines[0]}"

        if header_lines[-1] == "":
            del header_lines[-1]

        n_cols = max(len(line) for line in itertools.chain(header_lines, footer_lines))
        body = "\n".join([*header_lines, "-" * n_cols, *footer_lines])

        return {
            "text/html": f"<pre>{html.escape(body)}</pre>",
            "text/plain": repr(self),
        }

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
        name = f"{type(ufunc).__module__}.{ufunc.__name__}.{str(method)}"
        arguments = {}
        for i, arg in enumerate(inputs):
            arguments[i] = arg
        arguments.update(kwargs)
        with ak._errors.OperationErrorContext(name, arguments):
            return ak._connect.numpy.array_ufunc(ufunc, method, inputs, kwargs)

    @property
    def numba_type(self):
        """
        The type of this Record when it is used in Numba. It contains enough
        information to generate low-level code for accessing any element,
        down to the leaves.

        See [Numba documentation](https://numba.pydata.org/numba-doc/dev/reference/types.html)
        on types and signatures.
        """
        ak.numba.register_and_check()

        if self._numbaview is None:
            self._numbaview = ak._connect.numba.arrayview.RecordView.fromrecord(self)
        import numba

        return numba.typeof(self._numbaview)

    def __getstate__(self):
        packed = ak.operations.to_packed(self._layout, highlevel=False)
        form, length, container = ak.operations.to_buffers(
            packed.array, buffer_key="{form_key}-{attribute}", form_key="node{id}"
        )
        if self._behavior is ak.behavior:
            behavior = None
        else:
            behavior = self._behavior
        return form.to_dict(), length, container, behavior, packed.at

    def __setstate__(self, state):
        form, length, container, behavior, at = state
        layout = ak.operations.from_buffers(
            form,
            length,
            container,
            highlevel=False,
            buffer_key="{form_key}-{attribute}",
        )
        layout = ak.record.Record(layout, at)
        self.layout = layout
        self.behavior = behavior

    def __copy__(self):
        return Record(self._layout, behavior=self._behavior)

    def __deepcopy__(self, memo):
        return Record(
            copy.deepcopy(self._layout, memo),
            behavior=copy.deepcopy(self._behavior, memo),
        )

    def __bool__(self):
        raise ak._errors.wrap_error(
            ValueError(
                "the truth value of a record is ambiguous; "
                "use ak.any() or ak.all() or pick a field"
            )
        )


class ArrayBuilder(Sized):
    """
    Args:
        behavior (None or dict): Custom #ak.behavior for arrays built by
            this ArrayBuilder.
        initial (int): Initial size (in bytes) of buffers used by the `ak::ArrayBuilder`.
        resize (float): Resize multiplier for buffers used by the `ak::ArrayBuilder`;
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
        b.end_list()      # ],        # 1 * var * float64     (closed first list; array length is 1)
        b.begin_list()    # [         # 1 * var * float64
        b.end_list()      # ],        # 2 * var * float64     (closed empty list; array length is 2)
        b.begin_list()    # [         # 2 * var * float64
        b.integer(4)      #   4,      # 2 * var * float64
        b.null()          #   null,   # 2 * var * ?float64    (now the floats are nullable)
        b.integer(5)      #   5       # 2 * var * ?float64
        b.end_list()      # ],        # 3 * var * ?float64
        b.begin_list()    # [         # 3 * var * ?float64
        b.begin_record()  #   {       # 3 * var * union[?float64, ?{}]
        b.field("x")      #     "x":  # 3 * var * union[?float64, ?{x: unknown}]
        b.integer(1)      #      1,   # 3 * var * union[?float64, ?{x: int64}]
        b.field("y")      #      "y": # 3 * var * union[?float64, ?{x: int64, y: unknown}]
        b.begin_list()    #      [    # 3 * var * union[?float64, ?{x: int64, y: var * unknown}]
        b.integer(2)      #        2, # 3 * var * union[?float64, ?{x: int64, y: var * int64}]
        b.integer(3)      #        3  # 3 * var * union[?float64, ?{x: int64, y: var * int64}]
        b.end_list()      #      ]    # 3 * var * union[?float64, ?{x: int64, y: var * int64}]
        b.end_record()    #   }       # 3 * var * union[?float64, ?{x: int64, y: var * int64}]
        b.end_list()      # ]         # 4 * var * union[?float64, ?{x: int64, y: var * int64}]

    To get an array, we take a #snapshot of the ArrayBuilder's current state.

        >>> b.snapshot()
        <Array [[1, 2, 3], ..., [{x: 1, y: ..., ...}]] type='4 * var * union[?float...'>
        >>> b.snapshot().show()
        [[1, 2, 3],
         [],
         [4, None, 5],
         [{x: 1, y: [2, 3]}]]

    The full set of filling commands is the following.

    * #null: appends a None value.
    * #boolean: appends True or False.
    * #integer: appends an integer.
    * #real: appends a floating-point value.
    * #complex: appends a complex value.
    * #datetime: appends a datetime value.
    * #timedelta: appends a timedelta value.
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
    * #extend: appends all the items from an iterable.
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
        <Array [[[-0.523, ..., [[2.16, ...], ...]]]] type='1 * var * var * union[fl...'>
        >>> builder.type.show()
        1 * var * var * union[
            float64,
            var * union[
                var * union[
                    float64,
                    var * unknown
                ],
                float64
            ]
        ]

    Note that this is a *general* method for building arrays; if the type is
    known in advance, more specialized procedures can be faster. This should
    be considered the "least effort" approach.
    """

    def __init__(self, *, behavior=None, initial=1024, resize=1.5):
        self._layout = _ext.ArrayBuilder(initial=initial, resize=resize)
        self.behavior = behavior

    @classmethod
    def _wrap(cls, layout, behavior=None):
        """
        Args:
            layout (`ak._ext.ArrayBuilder`): Low-level builder to wrap.
            behavior (None or dict): Custom #ak.behavior for arrays built by
                this ArrayBuilder.

        Wraps a low-level `ak._ext.ArrayBuilder` as a high-level
        #ak.ArrayBulider.

        The #ak.ArrayBuilder constructor creates a new `ak._ext.ArrayBuilder`
        with no accumulated data, but Numba needs to wrap existing data
        when returning from a lowered function.
        """
        assert isinstance(layout, _ext.ArrayBuilder)
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
            raise ak._errors.wrap_error(TypeError("behavior must be None or a dict"))

    def tolist(self):
        """
        Converts this Array into Python objects; same as #ak.to_list
        (but without the underscore, like NumPy's
        [tolist](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.tolist.html)).
        """
        return self.to_list()

    def to_list(self):
        """
        Converts this Array into Python objects; same as #ak.to_list.
        """
        return self.snapshot().to_list()

    def to_numpy(self, allow_missing=True):
        """
        Converts this Array into a NumPy array, if possible; same as #ak.to_numpy.
        """
        return self.snapshot().to_numpy(allow_missing=allow_missing)

    @property
    def type(self):
        """
        The high-level type of the accumulated array; same as #ak.type.

        Note that the outermost element of an Array's type is always an
        #ak.types.ArrayType, which specifies the number of elements in the array.

        The type of a #ak.contents.Content (from #ak.Array.layout) is not
        wrapped by an #ak.types.ArrayType.
        """
        form = ak.forms.from_json(self._layout.form())
        return ak.types.ArrayType(
            form.type_from_behavior(self._behavior), len(self._layout)
        )

    @property
    def typestr(self):
        """
        The high-level type of this accumulated array, presented as a string.

        Note that the outermost element of an Array's type is always an
        #ak.types.ArrayType, which specifies the number of elements in the array.

        The type of a #ak.contents.Content (from #ak.Array.layout) is not
        wrapped by an #ak.types.ArrayType.
        """
        return str(self.type)

    def __len__(self):
        """
        The current length of the accumulated array.
        """
        return len(self._layout)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self._repr(80)

    def _repr(self, limit_cols):
        typestr = repr(self.typestr)

        limit_type = limit_cols - len("<ArrayBuilder type=>")
        if len(typestr) > limit_type:
            typestr = typestr[: (limit_type - 4)] + "..." + typestr[-1]

        return f"<ArrayBuilder type={typestr}>"

    def show(self, limit_rows=20, limit_cols=80, type=False, stream=sys.stdout):
        """
        Args:
            limit_rows (int): Maximum number of rows (lines) to use in the output.
            limit_cols (int): Maximum number of columns (characters wide).
            type (bool): If True, print the type as well. (Doesn't count toward number
                of rows/lines limit.)
            stream (object with a ``write(str)`` method or None): Stream to write the
                output to. If None, return a string instead of writing to a stream.

        Display the contents of the array within `limit_rows` and `limit_cols`, using
        ellipsis (`...`) for hidden nested data.

        This method takes a snapshot of the data and calls show on it, and a snapshot
        copies data.
        """
        return self.snapshot().show(
            limit_rows=limit_rows, limit_cols=limit_cols, type=type, stream=stream
        )

    def __array__(self, *args, **kwargs):
        """
        Intercepts attempts to convert a #snapshot of this array into a
        NumPy array and either performs a zero-copy conversion or raises
        an error.

        See #ak.Array.__array__ for a more complete description.
        """
        arguments = {0: self}
        for i, arg in enumerate(args):
            arguments[i + 1] = arg
        arguments.update(kwargs)
        with ak._errors.OperationErrorContext("numpy.asarray", arguments):
            return ak._connect.numpy.convert_to_array(self.snapshot(), args, kwargs)

    @property
    def numba_type(self):
        """
        The type of this Array when it is used in Numba. It contains enough
        information to generate low-level code for accessing any element,
        down to the leaves.

        See [Numba documentation](https://numba.pydata.org/numba-doc/dev/reference/types.html)
        on types and signatures.
        """
        ak.numba.register_and_check()

        return ak._connect.numba.builder.ArrayBuilderType(self._behavior)

    def __bool__(self):
        if len(self) == 1:
            return bool(self[0])
        else:
            raise ak._errors.wrap_error(
                ValueError(
                    "the truth value of an array whose length is not 1 is ambiguous; "
                    "use ak.any() or ak.all()"
                )
            )

    def snapshot(self):
        """
        Converts the currently accumulated data into an #ak.Array.

        The currently accumulated data are *copied* into the new array.
        """
        formstr, length, container = self._layout.to_buffers()
        form = ak.forms.from_json(formstr)

        with ak._errors.OperationErrorContext("ak.ArrayBuilder.snapshot", {}):
            return ak.operations.ak_from_buffers._impl(
                form,
                length,
                container,
                buffer_key="{form_key}-{attribute}",
                backend="cpu",
                highlevel=True,
                behavior=self._behavior,
                simplify=True,
            )

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

    def complex(self, x):
        """
        Appends a floating point number `x` at the current position in the
        accumulated array.
        """
        self._layout.complex(x)

    def datetime(self, x):
        """
        Appends a datetime value `x` at the current position in the
        accumulated array.
        """
        self._layout.datetime(x)

    def timedelta(self, x):
        """
        Appends a timedelta value `x` at the current position in the
        accumulated array.
        """
        self._layout.timedelta(x)

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

            >>> builder = ak.ArrayBuilder()
            >>> builder.begin_list()
            >>> builder.real(1.1)
            >>> builder.real(2.2)
            >>> builder.real(3.3)
            >>> builder.end_list()
            >>> builder.begin_list()
            >>> builder.end_list()
            >>> builder.begin_list()
            >>> builder.real(4.4)
            >>> builder.real(5.5)
            >>> builder.end_list()

        produces

            >>> builder.show()
            [[1.1, 2.2, 3.3],
             [],
             [4.4, 5.5]]
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

            >>> builder = ak.ArrayBuilder()
            >>> builder.begin_tuple(3)
            >>> builder.index(0).integer(1)
            >>> builder.index(1).real(1.1)
            >>> builder.index(2).string("one")
            >>> builder.end_tuple()
            >>> builder.begin_tuple(3)
            >>> builder.index(0).integer(2)
            >>> builder.index(1).real(2.2)
            >>> builder.index(2).string("two")
            >>> builder.end_tuple()

        produces

            >>> builder.show()
            [(1, 1.1, 'one'),
             (2, 2.2, 'two')]
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

            >>> builder.show()
            [{x: 1, y: 1.1},
             {x: 2, y: 2.2}]

        with type

            >>> builder.type.show()
            2 * points[
                x: float64,
                y: float64
            ]

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

    def append(self, obj):
        """
        Args:
            obj: The data to append (None, bool, int, float, bytes, str, or
                anything recognized by #ak.from_iter).

        Appends any type, which can be a shorthand for #null,
        #boolean, #integer, #real, #bytestring, or #string, but also
        an #ak.Array or #ak.Record to *reference* values from an existing
        dataset, or any Python object to *convert* to Awkward Array.

        If `obj` is an iterable (including dict), this is equivalent to
        #ak.from_iter except that it fills an existing #ak.ArrayBuilder,
        rather than creating a new one.
        """
        self._layout.fromiter(obj)

    def extend(self, obj):
        """
        Args:
            obj (iterable): Iterable of data to extend this ArrayBuilder with.

        Appends every value from `obj`.
        """
        for x in obj:
            self._layout.fromiter(x)

    class _Nested:
        def __init__(self, arraybuilder):
            self._arraybuilder = arraybuilder

        def __repr__(self):
            typestr = repr(self._arraybuilder.typestr)

            limit_type = 80 - len("<ArrayBuilder. type=>") - len(self._name)
            if len(typestr) > limit_type:
                typestr = typestr[: (limit_type - 4)] + "..." + typestr[-1]

            return f"<ArrayBuilder.{self._name} type={typestr}>"

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

            >>> builder = ak.ArrayBuilder()
            >>> with builder.list():
            ...     builder.real(1.1)
            ...     builder.real(2.2)
            ...     builder.real(3.3)
            ...
            >>> with builder.list():
            ...     pass
            ...
            >>> with builder.list():
            ...     builder.real(4.4)
            ...     builder.real(5.5)
            ...

        to produce the same result.

            >>> builder.show()
            [[1.1, 2.2, 3.3],
             [],
             [4.4, 5.5]]

        Since context managers aren't yet supported by Numba, this method
        can't be used in Numba.
        """
        return self.List(self)

    class Tuple(_Nested):
        _name = "tuple"

        def __init__(self, arraybuilder, numfields):
            super().__init__(arraybuilder)
            self._numfields = numfields

        def __enter__(self):
            self._arraybuilder.begin_tuple(self._numfields)

        def __exit__(self, type, value, traceback):
            self._arraybuilder.end_tuple()

    def tuple(self, numfields):
        """
        Context manager to prevent unpaired #begin_tuple and #end_tuple. The
        example in the #begin_tuple documentation can be rewritten as

            >>> builder = ak.ArrayBuilder()
            >>> with builder.tuple(3):
            ...     builder.index(0).integer(1)
            ...     builder.index(1).real(1.1)
            ...     builder.index(2).string("one")
            ...
            >>> with builder.tuple(3):
            ...     builder.index(0).integer(2)
            ...     builder.index(1).real(2.2)
            ...     builder.index(2).string("two")
            ...

        to produce the same result.

            >>> builder.show()
            [(1, 1.1, 'one'),
             (2, 2.2, 'two')]

        Since context managers aren't yet supported by Numba, this method
        can't be used in Numba.
        """
        return self.Tuple(self, numfields)

    class Record(_Nested):
        _name = "record"

        def __init__(self, arraybuilder, name):
            super().__init__(arraybuilder)
            self._name = name

        def __enter__(self):
            self._arraybuilder.begin_record(name=self._name)

        def __exit__(self, type, value, traceback):
            self._arraybuilder.end_record()

    def record(self, name=None):
        """
        Context manager to prevent unpaired #begin_record and #end_record. The
        example in the #begin_record documentation can be rewritten as

            >>> builder = ak.ArrayBuilder()
            >>> with builder.record("points"):
            ...     builder.field("x").real(1)
            ...     builder.field("y").real(1.1)
            ...
            >>> with builder.record("points"):
            ...     builder.field("x").real(2)
            ...     builder.field("y").real(2.2)
            ...

        to produce the same result.

            >>> builder.show()
            [{x: 1, y: 1.1},
             {x: 2, y: 2.2}]

        Since context managers aren't yet supported by Numba, this method
        can't be used in Numba.
        """
        return self.Record(self, name)


def ignore_in_to_list(getitem_function):
    getitem_function.ignore_in_to_list = True
    return getitem_function
