# ak.Array

Defined in [awkward.highlevel](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/highlevel.py) on [line 136](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/highlevel.py#L136).

#### *class* ak.Array(data, \*, behavior=None, with_name=None, check_valid=False, backend=None, attrs=None, named_axis=None)

* **Parameters:**
  * **data** ([`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content), [`ak.Array`](sphinx-llm:fe66bf03f1f8448b98db1c341ee55014), `np.ndarray`, `cp.ndarray`, `pyarrow.*`, str, dict, or iterable) – 

    Data to wrap or convert into an array.
    : - If a NumPy array, the regularity of its dimensions is preserved
        and the data are viewed, not copied.
      - CuPy arrays are treated the same way as NumPy arrays except that
        they default to `backend="cuda"`, rather than `backend="cpu"`.
      - If a pyarrow object, calls [`ak.from_arrow`](sphinx-llm:a01617c5c7474de09fdc0ddb61b2cced#ak.from_arrow), preserving as much
        metadata as possible, usually zero-copy.
      - If a dict of str → columns, combines the columns into an
        array of records (like Pandas’s DataFrame constructor).
      - If a string, the data are assumed to be JSON.
      - If an iterable, calls [`ak.from_iter`](sphinx-llm:40c924e275044a0993aef7f20508d933#ak.from_iter), which assumes all dimensions
        have irregular lengths.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for this Array only.
  * **with_name** (*None* *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Gives tuples and records a name that can be
    used to override their behavior (see below).
  * **check_valid** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, verify that the [`layout`](sphinx-llm:4f924c488f9a4faba48f9775e68bcd04) is valid.
  * **backend** (None, `"cpu"`, `"jax"`, `"cuda"`) – If `"cpu"`, the Array will be placed in
    main memory for use with other `"cpu"` Arrays and Records; if `"cuda"`,
    the Array will be placed in GPU global memory using CUDA; if `"jax"`, the structure
    is copied to the CPU for use with JAX. if None, the `data` are left untouched.

High-level array that can contain data of any type.

For most users, this is the only class in Awkward Array that matters: it
is the entry point for data analysis with an emphasis on usability. It
intentionally has a minimum of methods, preferring standalone functions
like:

```default
ak.num(array1)
ak.combinations(array1)
ak.cartesian([array1, array2])
ak.zip({"x": array1, "y": array2, "z": array3})
```

instead of bound methods like:

```default
array1.num()
array1.combinations()
array1.cartesian([array2, array3])
array1.zip(...)   # ?
```

because its namespace is valuable for domain-specific parameters and
functionality. For example, if records contain a field named `"num"`,
they can be accessed as:

```default
array1.num
```

instead of:

```default
array1["num"]
```

without any confusion or interference from [`ak.num`](sphinx-llm:137e3487aaa0453fa61d9c86fc644b7b#ak.num). The same is true
for domain-specific methods that have been attached to the data. For
instance, an analysis of mailing addresses might have a function that
computes zip codes, which can be attached to the data with a method
like:

```default
latlon.zip()
```

without any confusion or interference from [`ak.zip`](sphinx-llm:8d4df9d3d238429d957dd1efd0afa844#ak.zip). Custom methods like
this can be added with [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior), and so the namespace of Array
attributes must be kept clear for such applications.

See also [`ak.Record`](sphinx-llm:e158df99f80043309c758f6e933352b2#ak.Record).

## Interfaces to other libraries

### NumPy

When NumPy
[universal functions](https://docs.scipy.org/doc/numpy/reference/ufuncs.html)
(ufuncs) are applied to an ak.Array, they are passed through the Awkward
data structure, applied to the numerical data at its leaves, and the output
maintains the original structure.

For example,

```pycon
>>> array = ak.Array([[1, 4, 9], [], [16, 25]])
>>> np.sqrt(array)
<Array [[1, 2, 3], [], [4, 5]] type='3 * var * float64'>
```

See also [`ak.Array.__array_ufunc__`](sphinx-llm:84f9d01bbe7e46e6a44d380e6212412b).

Some NumPy functions other than ufuncs are also handled properly in
NumPy >= 1.17 (see
[NEP 18](https://numpy.org/neps/nep-0018-array-function-protocol.html))
and if an Awkward override exists. That is,:

```default
np.concatenate
```

can be used on an Awkward Array because:

```default
ak.concatenate
```

exists.

### Pandas

Ragged arrays (list type) can be converted into Pandas
[MultiIndex](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html)
rows and nested records can be converted into MultiIndex columns. If the
Awkward Array has only one “branch” of nested lists (i.e. different record
fields do not have different-length lists, but a single chain of lists-of-lists
is okay), then it can be losslessly converted into a single DataFrame.
Otherwise, multiple DataFrames are needed, though they can be merged (with a
loss of information).

The [`ak.to_dataframe`](sphinx-llm:470705b5237144ff83c26cdb7cdcee4d#ak.to_dataframe) function performs this conversion; if `how=None`, it
returns a list of DataFrames; otherwise, `how` is passed to `pd.merge` when
merging the resultant DataFrames.

### Numba

Arrays can be used in [Numba](http://numba.pydata.org/): they can be
passed as arguments to a Numba-compiled function or returned as return
values. The only limitation is that Awkward Arrays cannot be *created*
inside the Numba-compiled function; to make outputs, consider
[`ak.ArrayBuilder`](sphinx-llm:2fb9ffd5c3a441658693b9cc500d5518#ak.ArrayBuilder).

### Arrow

Arrays are convertible to and from [Apache Arrow](https://arrow.apache.org/),
a standard for representing nested data structures in columnar arrays.
See [`ak.to_arrow`](sphinx-llm:b7375fef0bca4ff1837b125c50ce5ad8#ak.to_arrow) and [`ak.from_arrow`](sphinx-llm:a01617c5c7474de09fdc0ddb61b2cced#ak.from_arrow).

### JAX

Derivatives of a calculation on an [`ak.Array`](sphinx-llm:fe66bf03f1f8448b98db1c341ee55014) (s) can be calculated with
[JAX](https://github.com/google/jax#readme), but only if the array
functions in `ak` / `numpy` are used, not the functions in the `jax`
library directly (apart from e.g. `jax.grad`).

Like NumPy ufuncs, the function and its derivatives are evaluated on the
numeric leaves of the data structure, maintaining structure in the output.

#### \_cpp_type *= None*

#### \_layout

#### \_behavior *= None*

#### \_attrs

#### *classmethod* \_\_init_subclass_\_(\*\*kwargs)

#### \_histogram_module_

#### \_\_dask_tokenize_\_()

#### \_update_class(restore=None)

#### *property* attrs *: awkward._attrs.Attrs*

The mapping containing top-level metadata, which is serialised
with the array during pickling.

Keys prefixed with `@` are identified as “transient” attributes
which are discarded prior to pickling, permitting the storage of
non-pickleable types.

#### *property* layout

The composable [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) elements that determine how this
Array is structured.

This may be considered a “low-level” view, as it distinguishes between
arrays that have the same logical meaning (i.e. same JSON output and
high-level #type) but different

* node types, such as [`ak.contents.ListArray`](sphinx-llm:e2e74f5b6c644d858b90eaa78cd4dedc#ak.contents.ListArray) and
  : [`ak.contents.ListOffsetArray`](sphinx-llm:00659f65177e4a1db8f94cec2f25984d#ak.contents.ListOffsetArray),
* integer type specialization, such as `int64` vs `int32`
* or specific values, such as gaps in a [`ak.contents.ListArray`](sphinx-llm:e2e74f5b6c644d858b90eaa78cd4dedc#ak.contents.ListArray).

The [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) elements are fully composable, whereas an
Array is not; the high-level Array is a single-layer “shell” around
its layout.

Layouts are rendered as XML instead of a nested list. For example,
the following `array`:

```default
ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
```

is presented as:

```default
<Array [[1.1, 2.2, 3.3], [], [4.4, 5.5]] type='3 * var * float64'>
```

but `array.layout` is presented as:

```default
<ListOffsetArray len='3'>
    <offsets><Index dtype='int64' len='4'>
        [0 3 3 5]
    </Index></offsets>
    <content>
        <NumpyArray dtype='float64' len='5'>[1.1 2.2 3.3 4.4 5.5]</NumpyArray>
    </content>
</ListOffsetArray>
```

(with truncation for large arrays).

#### *property* behavior

The `behavior` parameter passed into this Array’s constructor.

* If a dict, this `behavior` overrides the global [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior).
  : Any keys in the global [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) but not this `behavior` are
    still valid, but any keys in both are overridden by this
    `behavior`. Keys with a None value are equivalent to missing keys,
    so this `behavior` can effectively remove keys from the
    global [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior).
* If None, the Array defaults to the global [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior).

See [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for a list of recognized key patterns and their
meanings.

#### *property* positional_axis *: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...]*

#### *property* named_axis *: awkward._namedaxis.AxisMapping*

#### *property* mask

Whereas:

```default
array[array_of_booleans]
```

removes elements from `array` in which `array_of_booleans` is False,:

```default
array.mask[array_of_booleans]
```

returns data with the same length as the original `array` but False
values in `array_of_booleans` are mapped to None. Such an output
can be used in mathematical expressions with the original `array`
because they are still aligned.

See [filtering]() and [`ak.mask`](sphinx-llm:9a507fdd6567423185d9261d0dc62bad#ak.mask).

#### tolist()

Converts this Array into Python objects; same as [`ak.to_list`](sphinx-llm:c5d0843c37c34cb88bbec16becc6a6f0#ak.to_list)
(but without the underscore, like NumPy’s
[tolist](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.tolist.html)).

#### to_list()

Converts this Array into Python objects; same as [`ak.to_list`](sphinx-llm:c5d0843c37c34cb88bbec16becc6a6f0#ak.to_list).

#### to_numpy(allow_missing=True)

Converts this Array into a NumPy array, if possible; same as [`ak.to_numpy`](sphinx-llm:0dc12c9b410c4566b8e6578a4b063dbf#ak.to_numpy).

#### *property* nbytes

The total number of bytes in all the [`ak.index.Index`](sphinx-llm:c5dc164fb63f49c0946c0b5dfb338435#ak.index.Index),
and [`ak.contents.NumpyArray`](sphinx-llm:6f8cd60f60b04ce7a9499f2967197e79#ak.contents.NumpyArray) buffers in this array tree.

It does not count buffers that must be kept in memory because
of ownership, but are not directly used in the array. Nor does it count
the (small) Python objects that reference the (large) array buffers.

#### *property* ndim

Number of dimensions (nested variable-length lists and/or regular arrays)
before reaching a numeric type or a record.

There may be nested lists within the record, as field values, but this
number of dimensions does not count those.

(Some fields may have different depths than others, which is why they
are not counted.)

#### *property* fields

List of field names or tuple slot numbers (as strings) of the outermost
record or tuple in this array.

If the array contains nested records, only the fields of the outermost
record are shown. If it contains tuples instead of records, its fields
are string representations of integers, such as `"0"`, `"1"`, `"2"`, etc.
The records or tuples may be within multiple layers of nested lists.

If the array contains neither tuples nor records, it is an empty list.

See also [`ak.fields`](sphinx-llm:056b2c11871845eaa8962e15bfbffd9e#ak.fields).

#### *property* is_tuple

If True, the top-most record structure has no named fields, i.e. it’s a tuple.

#### \_ipython_key_completions_()

#### *property* type

The high-level type of this Array; same as [`ak.type`](sphinx-llm:65d3dedfc8ad48c98f3bacc6472ae8c8#ak.type).

Note that the outermost element of an Array’s type is always an
[`ak.types.ArrayType`](sphinx-llm:53688a8c7ce14810a6b2ed10d56f008f#ak.types.ArrayType), which specifies the number of elements in the array.

The type of a [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) (from [`ak.Array.layout`](sphinx-llm:4f924c488f9a4faba48f9775e68bcd04)) is not
wrapped by an [`ak.types.ArrayType`](sphinx-llm:53688a8c7ce14810a6b2ed10d56f008f#ak.types.ArrayType).

#### *property* typestr

The high-level type of this Array, presented as a string.

#### \_\_len_\_()

The length of this Array, only counting the outermost structure.

For example, the length of:

```default
ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
```

is `3`, not `5`.

#### \_\_iter_\_()

Iterates over this Array in Python.

Note that this is the *slowest* way to access data (even slower than
native Python objects, like lists and dicts). Usually, you should
express your problems in array-at-a-time operations.

In other words, do this:

```pycon
>>> np.sqrt(ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]))
<Array [[1.05, 1.48, 1.82], [], [2.1, 2.35]] type='3 * var * float64'>
```

not this:

```pycon
>>> for outer in ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]):
...     for inner in outer:
...         print(np.sqrt(inner))
...
1.0488088481701516
1.4832396974191326
1.816590212458495
2.0976176963403033
2.345207879911715
```

Iteration over Arrays exists so that they can be more easily inspected
as Python objects.

See also [`ak.to_list`](sphinx-llm:c5d0843c37c34cb88bbec16becc6a6f0#ak.to_list).

#### \_\_getitem_\_(where)

* **Parameters:**
  **where** (*many types supported; see below*) – Index of positions to
  select from this Array.

Select items from the Array using an extension of NumPy’s (already
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
  Indexes beyond the size of the array, either because they’re too
  large or because they’re too negative, raise errors. In
  particular, some nested lists might contain a desired element
  while others don’t; this would raise an error.
* **A slice** (either a Python `slice` object or the
  `start:stop:step` syntax) selects a range of elements. The
  `start` and `stop` values are zero-indexed; `start` is inclusive
  and `stop` is exclusive, like Python/NumPy. Negative `step`
  values are allowed, but a `step` of `0` is an error. Slices
  beyond the size of the array are not errors but are truncated,
  like Python/NumPy.
* **A string** selects a tuple or record field, even if its
  position in the tuple is to the left of the dimension where the
  tuple/record is defined. (See [projection]() below.) This is
  similar to NumPy’s
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
  NumPy’s
  [compress](https://docs.scipy.org/doc/numpy/reference/generated/numpy.compress.html)
  function.
* **An integer array** (or any iterable, other than the top-level
  tuple) selects elements like a single integer, but produces a
  regular dimension of as many as are desired. The array can have
  any length, any order, and it can have duplicates and incomplete
  coverage. The behavior is similar to NumPy’s
  [take](https://docs.scipy.org/doc/numpy/reference/generated/numpy.take.html)
  function.
* **An integer Array with missing (None) items** selects multiple
  values by index, as above, but None values are passed through
  to the output. This behavior matches pyarrow’s
  [Array.take](https://arrow.apache.org/docs/python/generated/pyarrow.Array.html#pyarrow.Array.take)
  which also manages arrays with missing values. See
  [option indexing]() below.
* **An Array of nested lists**, ultimately containing booleans or
  integers and having the same lengths of lists at each level as
  the Array to which they’re applied, selects by boolean or by
  integer at the deeply nested level. Missing items at any level
  above the deepest level must broadcast. See [nested indexing]() below.

A tuple of the above applies each slice item to a dimension of the
data, which can be very expressive. More than one flat boolean/integer
array are “iterated as one” as described in the
[NumPy documentation](https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing).

#### Filtering

A common use of selection by boolean arrays is to filter a dataset by
some property. For instance, to get the odd values of

```pycon
>>> array = ak.Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

one can put an array expression with True for each odd value inside
square brackets:

```pycon
>>> array[array % 2 == 1]
<Array [1, 3, 5, 7, 9] type='5 * int64'>
```

This technique is so common in NumPy and Pandas data analysis that it
is often read as a syntax, rather than a consequence of array slicing.

The extension to nested arrays like

```pycon
>>> array = ak.Array([[[0, 1, 2], [], [3, 4], [5]], [[6, 7, 8], [9]]])
```

allows us to use the same syntax more generally.

```pycon
>>> array[array % 2 == 1]
<Array [[[1], [], [3], [5]], [[7], [9]]] type='2 * var * var * int64'>
```

In this example, the boolean array is itself nested (see
[nested indexing]() below).

```pycon
>>> array % 2 == 1
<Array [[[False, True, False], ..., [True]], ...] type='2 * var * var * bool'>
```

This also applies to data with record structures.

For nested data, we often need to select the first or first two
elements from variable-length lists. That can be a problem if some
lists are empty. A function like [`ak.num`](sphinx-llm:137e3487aaa0453fa61d9c86fc644b7b#ak.num) can be useful for first
selecting by the lengths of lists.

```pycon
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
```

It’s sometimes also a problem that “cleaning” the dataset by dropping
empty lists changes its alignment, so that it can no longer be used
in calculations with “uncleaned” data. For this, [`ak.mask`](sphinx-llm:9a507fdd6567423185d9261d0dc62bad#ak.mask) can be
useful because it inserts None in positions that fail the filter,
rather than removing them.

```pycon
>>> ak.mask(array, ak.num(array) > 1)
<Array [[1.1, 2.2, 3.3], ..., [7.7, ..., 9.9]] type='6 * option[var * float64]'>
```

Note, however, that the `0` or `1` to pick the first or second
item of each nested list is in the second dimension, so the first
dimension of the slice must be a `:`.

```pycon
>>> ak.mask(array, ak.num(array) > 1)[:, 0]
<Array [1.1, None, 4.4, None, None, 7.7] type='6 * ?float64'>
>>> ak.mask(array, ak.num(array) > 1)[:, 1]
<Array [2.2, None, 5.5, None, None, 8.8] type='6 * ?float64'>
```

Another syntax for:

```default
ak.mask(array, array_of_booleans)
```

is:

```default
array.mask[array_of_booleans]
```

(which is 5 characters away from simply filtering the `array`).

#### Projection

The following

```pycon
>>> array = ak.Array([[{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}],
...                   [{"x": 3.3, "y": [3, 3, 3]}],
...                   [{"x": 0, "y": []}, {"x": 1.1, "y": [1, 1, 1]}]])
```

has records inside of nested lists:

```pycon
>>> array.type.show()
3 * var * {
    x: float64,
    y: var * int64
}
```

In principle, one should select nested lists before record fields,

```pycon
>>> array[2, :, "x"]
<Array [0, 1.1] type='2 * float64'>
>>> array[::2, :, "x"]
<Array [[1.1, 2.2], [0, 1.1]] type='2 * var * float64'>
```

but it’s also possible to select record fields first.

```pycon
>>> array["x"]
<Array [[1.1, 2.2], [3.3], [0, 1.1]] type='3 * var * float64'>
```

The string can “commute” to the left through integers and slices to
get the same result as it would in its “natural” position.

```pycon
>>> array[2, :, "x"]
<Array [0, 1.1] type='2 * float64'>
>>> array[2, "x", :]
<Array [0, 1.1] type='2 * float64'>
>>> array["x", 2, :]
<Array [0, 1.1] type='2 * float64'>
```

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
[`ak.zip`](sphinx-llm:8d4df9d3d238429d957dd1efd0afa844#ak.zip).

Note, however, that while a column string can “commute” with row
indexes to the left of its position in the tree, it can’t commute
to the right. For example, it’s possible to use slices inside
`"y"` because `"y"` is a list:

```pycon
>>> array[0, :, "y"]
<Array [[1], [2, 2]] type='2 * var * int64'>
>>> array[0, :, "y", 0]
<Array [1, 2] type='2 * int64'>
```

but it’s not possible to move `"y"` to the right

```pycon
>>> array[0, :, 0, "y"]
IndexError: while attempting to slice
    <Array [[{x: 1.1, y: [1]}, {...}], ...] type='3 * var * {x: float64, y:...'>
with
    (0, :, 0, 'y')
at inner NumpyArray of length 2, using sub-slice (0).
```

because the `array[0, :, 0, ...]` slice applies to both `"x"` and
`"y"` before `"y"` is selected, and `"x"` is a one-dimensional
NumpyArray that can’t take more than its share of slices.

Finally, note that the dot (`__getattr__`) syntax is equivalent to a single
string in a slice (`__getitem__`) if the field name is a valid Python
identifier and doesn’t conflict with [`ak.Array`](sphinx-llm:fe66bf03f1f8448b98db1c341ee55014) methods or properties.

```pycon
>>> array.x
<Array [[1.1, 2.2], [3.3], [0, 1.1]] type='3 * var * float64'>
>>> array.y
<Array [[[1], [2, 2]], ..., [[], [1, ...]]] type='3 * var * var * int64'>
```

#### Nested Projection

If records are nested within records, you can use a series of strings in
the selector to drill down. For instance, with the following

```pycon
>>> array = ak.Array([
...     {"a": {"x": 1, "y": 2}, "b": {"x": 10, "y": 20}, "c": {"x": 1.1, "y": 2.2}},
...     {"a": {"x": 1, "y": 2}, "b": {"x": 10, "y": 20}, "c": {"x": 1.1, "y": 2.2}},
...     {"a": {"x": 1, "y": 2}, "b": {"x": 10, "y": 20}, "c": {"x": 1.1, "y": 2.2}}])
```

we can go directly to the numerical data by specifying a string for the
outer field and a string for the inner field.

```pycon
>>> array["a", "x"]
<Array [1, 1, 1] type='3 * int64'>
>>> array["a", "y"]
<Array [2, 2, 2] type='3 * int64'>
>>> array["b", "y"]
<Array [20, 20, 20] type='3 * int64'>
>>> array["c", "y"]
<Array [2.2, 2.2, 2.2] type='3 * float64'>
```

As with single projections, the dot (`__getattr__`) syntax is equivalent
to a single string in a slice (`__getitem__`) if the field name is a valid
Python identifier and doesn’t conflict with [`ak.Array`](sphinx-llm:fe66bf03f1f8448b98db1c341ee55014) methods or properties.

```pycon
>>> array.a.x
<Array [1, 1, 1] type='3 * int64'>
```

You can even get every field of the same name within an outer record using
a list of field names for the outer record. The following selects the `"x"`
field of `"a"`, `"b"`, and `"c"` records:

```pycon
>>> array[["a", "b", "c"], "x"].show()
[{a: 1, b: 10, c: 1.1},
 {a: 1, b: 10, c: 1.1},
 {a: 1, b: 10, c: 1.1}]
```

You don’t need to get all fields:

```pycon
>>> array[["a", "b"], "x"].show()
[{a: 1, b: 10},
 {a: 1, b: 10},
 {a: 1, b: 10}]
```

And you can select lists of field names at all levels:

```pycon
>>> array[["a", "b"], ["x", "y"]].show()
[{a: {x: 1, y: 2}, b: {x: 10, y: 20}},
 {a: {x: 1, y: 2}, b: {x: 10, y: 20}},
 {a: {x: 1, y: 2}, b: {x: 10, y: 20}}]
```

#### Option indexing

NumPy arrays can be sliced by all of the above slice types except
arrays with missing values and arrays with nested lists, both of
which are inexpressible in NumPy. Missing values, represented by
None in Python, are called option types ([`ak.types.OptionType`](sphinx-llm:f8e0eb36e1334eef887fb5ab81458600#ak.types.OptionType)) in
Awkward Array and can be used as a slice.

For example,

```pycon
>>> array = ak.Array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
```

can be sliced with a boolean array

```pycon
>>> array[[False, False, False, False, True, False, True, False, True]]
<Array [5.5, 7.7, 9.9] type='3 * float64'>
```

or a boolean array containing None values:

```pycon
>>> array[[False, False, False, False, True, None, True, None, True]]
<Array [5.5, None, 7.7, None, 9.9] type='5 * ?float64'>
```

Similarly for arrays of integers and None:

```pycon
>>> array[[0, 1, None, None, 7, 8]]
<Array [1.1, 2.2, None, None, 8.8, 9.9] type='6 * ?float64'>
```

This is the same behavior as pyarrow’s
[Array.take](https://arrow.apache.org/docs/python/generated/pyarrow.Array.html#pyarrow.Array.take),
which establishes a convention for how to interpret slice arrays
with option type:

```pycon
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
```

#### Nested indexing

Awkward Array’s nested lists can be used as slices as well, as long
as the type at the deepest level of nesting is boolean or integer.

For example,

```pycon
>>> array = ak.Array([[[0.0, 1.1, 2.2], [], [3.3, 4.4]], [], [[5.5]]])
```

can be sliced at the top level with one-dimensional arrays:

```pycon
>>> array[[False, True, True]]
<Array [[], [[5.5]]] type='2 * var * var * float64'>
>>> array[[1, 2]]
<Array [[], [[5.5]]] type='2 * var * var * float64'>
```

with singly nested lists:

```pycon
>>> array[[[False, True, True], [], [True]]]
<Array [[[], [3.3, 4.4]], [], [[5.5]]] type='3 * var * var * float64'>
>>> array[[[1, 2], [], [0]]]
<Array [[[], [3.3, 4.4]], [], [[5.5]]] type='3 * var * var * float64'>
```

and with doubly nested lists:

```pycon
>>> array[[[[False, True, False], [], [True, False]], [], [[False]]]]
<Array [[[1.1], [], [3.3]], [], [[]]] type='3 * var * var * float64'>
>>> array[[[[1], [], [0]], [], [[]]]]
<Array [[[1.1], [], [3.3]], [], [[]]] type='3 * var * var * float64'>
```

The key thing is that the nested slice has the same number of elements
as the array it’s slicing at every level of nesting that it reproduces.
This is similar to the requirement that boolean arrays have the same
length as the array they’re filtering.

This kind of slicing is useful because NumPy’s
[universal functions](https://docs.scipy.org/doc/numpy/reference/ufuncs.html)
produce arrays with the same structure as the original array, which
can then be used as filters.

```pycon
>>> ((array * 10) % 2 == 1).show()
[[[False, True, False], [], [True, False]],
 [],
 [[True]]]
>>> (array[(array * 10) % 2 == 1]).show()
[[[1.1], [], [3.3]],
 [],
 [[5.5]]]
```

Functions whose names start with “arg” return index positions, which
can be used with the integer form.

```pycon
>>> np.argmax(array, axis=-1).show()
[[2, None, 1],
 [],
 [0]]
>>> array[np.argmax(array, axis=-1)].show()
[[[3.3, 4.4], None, []],
 [],
 [[5.5]]]
```

Here, the `np.argmax` returns the integer position of the maximum
element or None for empty arrays. It’s a nice example of
[option indexing]() with [nested indexing]().

When applying a nested index with missing (None) entries at levels
higher than the last level, the indexer must have the same dimension
as the array being indexed, and the resulting output will have missing
entries at the corresponding locations, e.g. for

```pycon
>>> array[ [[[0, None, 2, None, None], None, [1]], None, [[0]]] ].show()
[[[0, None, 2.2, None, None], None, [4.4]],
 None,
 [[5.5]]]
```

the sub-list at entry 0,0 is extended as the masked entries are
acting at the last level, while the higher levels of the indexer all
have the same dimension as the array being indexed.

#### \_\_bytes_\_() → [bytes](https://docs.python.org/3/library/stdtypes.html#bytes)

#### \_\_setitem_\_(where, what)

* **Parameters:**
  * **where** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple) *of* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Field name to add to records in the array.
  * **what** ([`ak.Array`](sphinx-llm:fe66bf03f1f8448b98db1c341ee55014)) – Array to add as the new field.

Unlike #_\_getitem_\_, which allows a wide variety of slice types,
only single field-slicing is supported for assignment.
([`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) arrays are immutable; field assignment replaces
the #layout with an array that has the new field using [`ak.with_field`](sphinx-llm:3c2fe1ca08b646e494d171d5ea7f2e33#ak.with_field).)

However, a field can be assigned deeply into a nested record e.g.

```pycon
>>> nested = ak.zip({"a" : ak.zip({"x" : [1, 2, 3]})})
>>> nested["a", "y"] = 2 * nested.a.x
>>> nested.show()
[{a: {x: 1, y: 2}},
 {a: {x: 2, y: 4}},
 {a: {x: 3, y: 6}}]
```

Note that the following does **not** work:

```pycon
>>> nested["a"]["y"] = 2 * nested.a.x # does not work, nested["a"] is a copy!
```

Always assign by passing the whole path to the top level

```pycon
>>> nested["a", "y"] = 2 * nested.a.x
```

If necessary, the new field will be broadcasted to fit the array.
For example, given

```pycon
>>> array = ak.Array([
...     [{"x": 1.1}, {"x": 2.2}, {"x": 3.3}], [], [{"x": 4.4}, {"x": 5.5}]
... ])
```

which has three elements with nested data in each, assigning

```pycon
>>> array["y"] = [100, 200, 300]
```

will result in

```pycon
>>> array.show()
[[{x: 1.1, y: 100}, {x: 2.2, y: 100}, {x: 3.3, y: 100}],
 [],
 [{x: 4.4, y: 300}, {x: 5.5, y: 300}]]
```

because the `100` in `what[0]` is broadcasted to all three nested
elements of `array[0]`, the `200` in `what[1]` is broadcasted to the
empty list `array[1]`, and the `300` in `what[2]` is broadcasted to
both elements of `array[2]`.

See [`ak.with_field`](sphinx-llm:3c2fe1ca08b646e494d171d5ea7f2e33#ak.with_field) for a variant that does not change the [`ak.Array`](sphinx-llm:fe66bf03f1f8448b98db1c341ee55014)
in-place. (Internally, this method uses [`ak.with_field`](sphinx-llm:3c2fe1ca08b646e494d171d5ea7f2e33#ak.with_field), so performance
is not a factor in choosing one over the other.)

#### \_\_delitem_\_(where)

* **Parameters:**
  **where** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple) *of* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Field name to remove from the array.

For example:

```pycon
>>> array = ak.Array([{"x": 3.3, "y": {"this": 10, "that": 20}}])
>>> del array["y", "that"]
>>> array.show()
[{x: 3.3, y: {this: 10}}]
```

See [`ak.without_field`](sphinx-llm:f7f408e734c14ae58c061536a46aa7a2#ak.without_field) for a variant that does not change the [`ak.Array`](sphinx-llm:fe66bf03f1f8448b98db1c341ee55014)
in-place. (Internally, this method uses [`ak.without_field`](sphinx-llm:f7f408e734c14ae58c061536a46aa7a2#ak.without_field), so performance
is not a factor in choosing one over the other.)

#### \_\_getattr_\_(where)

* **Parameters:**
  **where** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Attribute name to lookup

Whenever possible, fields can be accessed as attributes.

For example, the fields of

```pycon
>>> array = ak.Array([
...     [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}, {"x": 3.3, "y": [3, 3, 3]}],
...     [],
...     [{"x": 4.4, "y": [4, 4, 4, 4]}, {"x": 5.5, "y": [5, 5, 5, 5, 5]}]
... ])
```

can be accessed as

```pycon
>>> array.x
<Array [[1.1, 2.2, 3.3], [], [4.4, 5.5]] type='3 * var * float64'>
>>> array.y
<Array [[[1], [2, 2], [3, 3, 3]], [], [...]] type='3 * var * var * int64'>
```

which are equivalent to `array["x"]` and `array["y"]`. (See
[projection]().)

Fields can’t be accessed as attributes when

* [`ak.Array`](sphinx-llm:fe66bf03f1f8448b98db1c341ee55014) methods or properties take precedence,
* a domain-specific behavior has methods or properties that take
  : precedence, or
* the field name is not a valid Python identifier or is a Python
  : keyword.

Note that while fields can be accessed as attributes, they cannot be
*assigned* as attributes. See [`ak.Array.__setitem__`](sphinx-llm:836393adb59043318c46225a4d2ea5cb) for more.

#### \_\_setattr_\_(name, value)

* **Parameters:**
  **where** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Attribute name to set

Set an attribute on the array.

Only existing public attributes e.g. [`ak.Array.layout`](sphinx-llm:4f924c488f9a4faba48f9775e68bcd04), or private
attributes (with leading underscores), can be set.

Fields are not assignable to as attributes, i.e. the following doesn’t work:

```default
array.z = new_field
```

Instead, always use [`ak.Array.__setitem__`](sphinx-llm:836393adb59043318c46225a4d2ea5cb):

```default
array["z"] = new_field
```

or [`ak.with_field`](sphinx-llm:3c2fe1ca08b646e494d171d5ea7f2e33#ak.with_field):

```default
array = ak.with_field(array, new_field, "z")
```

to add or modify a field.

#### \_\_dir_\_()

Lists all methods, properties, and field names (see #_\_getattr_\_)
that can be accessed as attributes.

#### \_\_str_\_()

#### \_\_repr_\_()

#### \_repr(limit_cols)

#### show(limit_rows=20, limit_cols=80, \*, type=False, named_axis=False, nbytes=False, backend=False, all=False, stream=STDOUT, formatter=None, precision=3)

* **Parameters:**
  * **limit_rows** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Maximum number of rows (lines) to use in the output.
  * **limit_cols** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Maximum number of columns (characters wide).
  * **type** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, print the type as well. (Doesn’t count toward number
    of rows/lines limit.)
  * **named_axis** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, print the named axis as well. (Doesn’t count toward number
    of rows/lines limit.)
  * **nbytes** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, print the number of bytes as well. (Doesn’t count toward number
    of rows/lines limit.)
  * **backend** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, print the backend of the array as well. (Doesn’t count toward number
    of rows/lines limit.)
  * **all** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, print the ‘type’, ‘named axis’, ‘nbytes’, and ‘backend’ of the array. (Doesn’t count toward number
    of rows/lines limit.)
  * **stream** (object with a `write(str)` method or None) – Stream to write the
    output to. If None, return a string instead of writing to a stream.
  * **formatter** (*Mapping* *or* *None*) – Mapping of types/type-classes to string formatters.
    If None, use the default formatter.

Display the contents of the array within `limit_rows` and `limit_cols`, using
ellipsis (`...`) for hidden nested data.

The `formatter` argument controls the formatting of individual values, c.f.
[https://numpy.org/doc/stable/reference/generated/numpy.set_printoptions.html](https://numpy.org/doc/stable/reference/generated/numpy.set_printoptions.html)
As Awkward Array does not implement strings as a NumPy dtype, the `numpystr`
key is ignored; instead, a `"bytes"` and/or `"str"` key is considered when formatting
string values, falling back upon `"str_kind"`.

#### \_repr_mimebundle_(include=None, exclude=None)

#### \_\_array_\_(dtype=None, copy=None)

Intercepts attempts to convert this Array into a NumPy array and
either performs a conversion if possible or raises an error.
The array may be copied depending on the values of `dtype` and `copy`.
The rules for copying are specified in the
[np.asarray](https://docs.scipy.org/doc/numpy/reference/generated/numpy.asarray.html)
documentation.

This function is also called by the
[np.asarray](https://docs.scipy.org/doc/numpy/reference/generated/numpy.asarray.html)
family of functions, which have `copy=False` by default.

```pycon
>>> np.asarray(ak.Array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]))
array([[1.1, 2.2, 3.3],
       [4.4, 5.5, 6.6]])
```

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

#### \_\_arrow_array_\_(type=None)

#### \_\_array_ufunc_\_(ufunc, method, \*inputs, \*\*kwargs)

Intercepts attempts to pass this Array to a NumPy
[universal functions](https://docs.scipy.org/doc/numpy/reference/ufuncs.html)
(ufuncs) and passes it through the Array’s structure.

This method conforms to NumPy’s
[NEP 13](https://numpy.org/neps/nep-0013-ufunc-overrides.html)
for overriding ufuncs, which has been
[available since NumPy 1.13](https://numpy.org/devdocs/release/1.13.0-notes.html#array-ufunc-added)
(and thus NumPy 1.13 is the minimum allowed version).

When any ufunc is applied to an Awkward Array, it applies to the
innermost level of structure and preserves the structure through the
operation.

For example, with

```pycon
>>> array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
```

applying `np.sqrt` would yield

```pycon
>>> np.sqrt(array).show()
[[1.05, 1.48, 1.82],
 [],
 [2.1, 2.35]]
```

In addition, many unary and binary operators implicitly call ufuncs,
such as `np.power` in

```pycon
>>> (array**2).show()
[[1.21, 4.84, 10.9],
 [],
 [19.4, 30.2]]
```

In the above example, `array` is a nested list of records and `2` is
a scalar. Awkward Array applies the same broadcasting rules as NumPy
plus a few more to deal with nested structures. In addition to
broadcasting a scalar, as above, it is possible to broadcast
arrays with less depth into arrays with more depth, such as

```pycon
>>> (array + ak.Array([10, 20, 30])).show()
[[11.1, 12.2, 13.3],
 [],
 [34.4, 35.5]]
```

See [`ak.broadcast_arrays`](sphinx-llm:ecf144fb88044843b0e932a031de236c#ak.broadcast_arrays) for details about broadcasting and the
generalized set of broadcasting rules.

Third party libraries can create ufuncs, not just NumPy, so any library
that “plays well” with the NumPy ecosystem can be used with Awkward
Arrays:

```pycon
>>> import numba as nb
>>> @nb.vectorize([nb.float64(nb.float64)])
... def sqr(x):
...     return x * x
...
>>> sqr(array).show()
[[1.21, 4.84, 10.9],
 [],
 [19.4, 30.2]]
```

See also #_\_array_function_\_.

#### \_\_array_function_\_(func, types, args, kwargs)

Intercepts attempts to pass this Array to those NumPy functions other
than universal functions that have an Awkward equivalent.

This method conforms to NumPy’s
[NEP 18](https://numpy.org/neps/nep-0018-array-function-protocol.html)
for overriding functions, which has been
[available since NumPy 1.17](https://numpy.org/devdocs/release/1.17.0-notes.html#numpy-functions-now-always-support-overrides-with-array-function)
(and
[NumPy 1.16 with an experimental flag set](https://numpy.org/devdocs/release/1.16.0-notes.html#numpy-functions-now-support-overrides-with-array-function)).

See also #_\_array_ufunc_\_.

#### numba_type()

The type of this Array when it is used in Numba. It contains enough
information to generate low-level code for accessing any element,
down to the leaves.

See [Numba documentation](https://numba.pydata.org/numba-doc/dev/reference/types.html)
on types and signatures.

#### \_\_reduce_ex_\_(protocol: [int](https://docs.python.org/3/library/functions.html#int)) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)

#### \_\_setstate_\_(state)

#### \_\_copy_\_()

#### \_\_deepcopy_\_(memo)

#### \_\_bool_\_()

#### cpp_type()

The C++ type of this Array when it is used in cppyy.:

```default
cpp_type (None or str): Generated on demand when the Array needs to be passed
    to a C++ (possibly templated) function defined by a ``cppyy`` compiler.
```

See [cppyy documentation](https://cppyy.readthedocs.io/en/latest/index.html)
on types and signatures.

#### \_\_cast_cpp_\_()

The `__cast_cpp__` is called by cppyy to determine a C++ type of an `ak.Array`.
It returns the C++ dataset type that is already registered with cppyy with the
parameters needed to construct the C++ type of this Array when it is
used in cppyy.

## Classes

| `Mask`   |    |
|----------|----|
