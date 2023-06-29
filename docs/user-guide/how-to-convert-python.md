---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

How to convert to/from Python objects
=====================================

Builtin Python objects like dicts and lists can be converted into Awkward Arrays, and all Awkward Arrays can be converted into Python objects. Awkward type information, such as the distinction between fixed-size and variable-length lists, is lost in the transformation to Python objects.

```{code-cell} ipython3
import awkward as ak
import numpy as np
import pandas as pd
```

From Python to Awkward
----------------------

The function for Python → Awkward conversion is {func}`ak.from_iter`.

```{code-cell} ipython3
py_objects = [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
py_objects
```

```{code-cell} ipython3
ak_array = ak.from_iter(py_objects)
ak_array
```

See the sections below for how Python types are mapped to Awkward types.

Note that this should be considered a slow, memory-intensive function: not only does it need to iterate over Python data, but it needs to discover the type of the data progressively. Internally, this function uses an {class}`ak.ArrayBuilder` to accumulate data and discover types simultaneously. Don't, for instance, convert a large, numerical dataset from NumPy or Arrow into Python objects just to use {func}`ak.from_iter`. There are specialized functions for that: see their tutorials (left-bar or ≡ button on mobile).

This is also the fallback operation of the {class}`ak.Array` and {class}`ak.Record` constructors. Usually, small examples are built by passing Python objects directly to these constructors.

```{code-cell} ipython3
ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
```

```{code-cell} ipython3
ak.Record({"x": 1, "y": [1.1, 2.2]})
```

From Awkward to Python
----------------------

The function for Awkward → Python conversion is {func}`ak.to_list`.

```{code-cell} ipython3
ak_array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
ak_array
```

```{code-cell} ipython3
ak.to_list(ak_array)
```

```{code-cell} ipython3
ak_record = ak.Record({"x": 1, "y": [1.1, 2.2]})
ak_record
```

```{code-cell} ipython3
ak.to_list(ak_record)
```

Note that this should be considered a slow, memory-intensive function, like {func}`ak.from_iter`. Don't, for instance, convert a large, numerical dataset with {func}`ak.to_list` just to convert those lists into NumPy or Arrow. There are specialized functions for that: see their tutorials (left-bar or ≡ button on mobile).

Awkward Arrays and Records have a `to_list` method. For small datasets (or a small slice of a dataset), this is a convenient way to get a quick view.

```{code-cell} ipython3
x = ak.Array(np.arange(1000))
y = ak.Array(np.tile(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]), 100))
ak_array = ak.zip({"x": x, "y": y})
ak_array
```

```{code-cell} ipython3
ak_array[100].to_list()
```

```{code-cell} ipython3
ak_array[100:110].to_list()
```

Pandas-style constructor
------------------------

As we have seen, the {class}`ak.Array`) constructor interprets an iterable argument as the data that it is meant to represent, as in:

```{code-cell} ipython3
py_objects1 = [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
py_objects1
```

```{code-cell} ipython3
ak.Array(py_objects1)
```

But sometimes, you have several iterables that you want to use as columns of a table. The [Pandas DataFrame constructor](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) interprets a dict of iterables as columns:

```{code-cell} ipython3
py_objects2 = ["one", "two", "three"]
py_objects2
```

```{code-cell} ipython3
pd.DataFrame({"x": py_objects1, "y": py_objects2})
```

And so does the {class}`ak.Array` constructor:

```{code-cell} ipython3
ak_array = ak.Array({"x": py_objects1, "y": py_objects2})
ak_array
```

```{code-cell} ipython3
ak.type(ak_array)
```

```{code-cell} ipython3
ak.to_list(ak_array)
```

Note that this is the transpose of the way the data would be interpreted if it were in a list, rather than a dict. The `"x"` and `"y"` values are interpreted as being interleaved in each record. There is no potential for conflict between the {func}`ak.from_iter`-style and Pandas-style constructors because {func}`ak.from_iter` applied to a dict would always return an {class}`ak.Record`, rather than an {class}`ak.Array`.

```{code-cell} ipython3
ak_record = ak.from_iter({"x": py_objects1, "y": py_objects2})
ak_record
```

```{code-cell} ipython3
ak.type(ak_record)
```

```{code-cell} ipython3
ak.to_list(ak_record)
```

The {func}`ak.from_iter` function applied to a dict is also equivalent to the {class}`ak.Record` constructor.

```{code-cell} ipython3
ak.Record({"x": py_objects1, "y": py_objects2})
```

Conversion of numbers and booleans
----------------------------------

Python `float`, `int`, and `bool` (so-called "primitive" types) are converted to `float64`, `int64`, and `bool` types in Awkward Arrays.

All floating-point Awkward types are converted to Python's `float`, all integral Awkward types are converted to Python's `int`, and Awkward's boolean type is converted to Python's `bool`.

```{code-cell} ipython3
ak.Array([1.1, 2.2, 3.3])
```

```{code-cell} ipython3
ak.Array([1.1, 2.2, 3.3]).to_list()
```

```{code-cell} ipython3
ak.Array([1, 2, 3, 4, 5])
```

```{code-cell} ipython3
ak.Array([1, 2, 3, 4, 5]).to_list()
```

```{code-cell} ipython3
ak.Array([True, False, True, False, False])
```

```{code-cell} ipython3
ak.Array([True, False, True, False, False]).to_list()
```

Conversion of lists
-------------------

Python lists, as well as iterables other than dict, tuple, str, and bytes, are converted to Awkward's variable-length lists. It is not possible to construct fixed-size lists with {func}`ak.from_iter`. (One way to do that is by converting a NumPy array with {func}`ak.from_numpy`.)

Awkward's variable-length and fixed-size lists are converted into Python lists with {func}`ak.to_list`.

```{code-cell} ipython3
ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
```

```{code-cell} ipython3
ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).to_list()
```

```{code-cell} ipython3
ak.Array([[1, 2, 3], [4, 5, 6]])
```

```{code-cell} ipython3
ak.Array([[1, 2, 3], [4, 5, 6]]).to_list()
```

```{note}
Advanced topic: the rest of this section may be skipped if you don't care about the distinction between fixed-size and variable-length lists.
```

Note that a NumPy array is an iterable, so {func}`ak.from_iter` iterates over it, constructing variable-length Awkward lists. By contrast, {func}`ak.from_numpy` casts the data (without iteration) into fixed-size Awkward lists.

```{code-cell} ipython3
np_array = np.array([[100, 200], [101, 201], [103, 203]])
np_array
```

```{code-cell} ipython3
ak.from_iter(np_array)
```

```{code-cell} ipython3
ak.from_numpy(np_array)
```

Note that the types differ: `var * int64` vs `2 * int64`. The {class}`ak.Array` constructor uses {func}`ak.from_numpy` if given a NumPy array (with `dtype != "O"`) and {func}`ak.from_iter` if given an iterable that it does not recognize.

This can be particularly subtle when NumPy arrays are nested within iterables.

```{code-cell} ipython3
np_array = np.array([[100, 200], [101, 201], [103, 203]])
np_array
```

```{code-cell} ipython3
# This is a NumPy array: constructor uses ak.from_numpy to get an array of fixed-size lists.
ak.Array(np_array)
```

```{code-cell} ipython3
py_objects = [np.array([100, 200]), np.array([101, 201]), np.array([103, 203])]
py_objects
```

This is a list that contains NumPy arrays: constructor uses ak.from_iter to get an array of variable-length lists.
```{code-cell} ipython3
ak.Array(py_objects)
```

```{code-cell} ipython3
np_array_dtype_O = np.array([[100, 200], [101, 201], [103, 203]], dtype="O")
np_array_dtype_O
```

This NumPy array has dtype="O", so it cannot be cast without iteration:
```{code-cell} ipython3
:tags: [raises-exception]

ak.Array(np_array_dtype_O)
```

{class}`ak.Array` knows that this is a NumPy array, but Awkward does not support object dtypes. Instead, one must use 
`from_iter` to tell Awkward that this iteration is intentional.
```{code-cell} ipython3
ak.from_iter(np_array_dtype_O)
```

The logic behind this policy is that only NumPy arrays with `dtype != "O"` are guaranteed to have fixed-size contents. Other cases must have `var` type lists.

```{code-cell} ipython3
py_objects = [np.array([1.1, 2.2, 3.3]), np.array([]), np.array([4.4, 5.5])]
py_objects
```

```{code-cell} ipython3
ak.Array(py_objects)
```

```{code-cell} ipython3
np_array_dtype_O = np.array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], dtype="O")
np_array_dtype_O
```

```{code-cell} ipython3
ak.from_iter(np_array_dtype_O)
```

Conversion of strings and bytestrings
-------------------------------------

Python strings (type `str`) are converted to and from Awkward's UTF-8 encoded strings and Python bytestrings (type `bytes`) are converted to and from Awkward's unencoded bytestrings.

```{code-cell} ipython3
ak.Array(["one", "two", "three", "four"])
```

```{code-cell} ipython3
ak.Array(["one", "two", "three", "four"]).to_list()
```

```{code-cell} ipython3
ak.Array([b"one", b"two", b"three", b"four"])
```

```{code-cell} ipython3
ak.Array([b"one", b"two", b"three", b"four"]).to_list()
```

```{note}
Advanced topic: the rest of this section may be skipped if you don't care about internal representations.
```

Awkward's strings and bytestrings are _specializations_ of variable-length lists. Whereas a list might be internally represented by a {class}`ak.contents.ListArray` or a {class}`ak.contents.ListOffsetArray`,

```{code-cell} ipython3
ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).layout
```

Strings and bytestrings are just {class}`ak.contents.ListArray`s and {class}`ak.contents.ListOffsetArray`s of one-byte integers with special parameters:

```{code-cell} ipython3
ak.Array(["one", "two", "three", "four"]).layout
```

These parameters indicate that the arrays of strings should have special behaviors, such as equality-per-string, rather than equality-per-character.

```{code-cell} ipython3
ak.Array([[1.1, 2.2], [], [3.3]]) == ak.Array([[1.1, 200], [], [3.3]])
```

```{code-cell} ipython3
ak.Array(["one", "two", "three", "four"]) == ak.Array(
    ["one", "TWO", "thirty three", "four"]
)
```

(Without this overloaded behavior, the string comparison would yield `[True, True, True]` for `"one" == "one"` and would fail to broadcast `"three"` and `"thirty three"`.)


The fact that strings are really just variable-length lists is worth keeping in mind, since they might behave in unexpectedly list-like ways. If you notice any behavior that ought to be overloded for strings, recommend it as a [feature request](https://github.com/scikit-hep/awkward-1.0/issues/new?assignees=&labels=feature&template=feature-request.md&title=).

+++

Conversion of dicts and tuples
------------------------------

Python dicts with string-valued keys are converted to and from Awkward's record type with named fields. The data associated with different fields can have different types, but you generally want data associated with all instances of the same field to have the same type. Python dicts with non-string valued keys have no equivalent in Awkward Array (records are very different from mappings).

Python tuples are converted to and from Awkward's record type with unnamed fields. Note that Awkward views Python's lists and tuples in very different ways: lists are expected to be variable-length with all elements having the same type, while tuples are expected to be fixed-size with elements having potentially different types, just like a record.

In the following example, the `"x"` field has type `int64` and the `"y"` field has type `var * int64`.

```{code-cell} ipython3
ak_array_rec = ak.Array([{"x": 1, "y": [1, 2]}, {"x": 2, "y": []}])
ak_array_rec
```

```{code-cell} ipython3
ak_array_rec.to_list()
```

Here is the corresponding example with tuples:

```{code-cell} ipython3
ak_array_tup = ak.Array([(1, [1, 2]), (2, [])])
ak_array_tup
```

```{code-cell} ipython3
ak_array_tup.to_list()
```

Both of these Awkward types, `{"x": int64, "y": var * int64}` and `(int64, var * int64)`, have two fields, but the first one has names for those fields.

Both can be extracted using strings between square brackets, though the strings must be `"0"` and `"1"` for the tuple.

```{code-cell} ipython3
ak_array_rec["y"]
```

```{code-cell} ipython3
ak_array_rec["y", 1]
```

```{code-cell} ipython3
ak_array_tup["1"]
```

```{code-cell} ipython3
ak_array_tup["1", 1]
```

Note the difference in meaning between the `"1"` and the `1` in the above example. For safety, you may want to use {func}`ak.unzip`

```{code-cell} ipython3
x, y = ak.unzip(ak_array_rec)
y
```

```{code-cell} ipython3
slot0, slot1 = ak.unzip(ak_array_tup)
slot1
```

That way, you can name the variables anything you like.

If fields are missing from some records, the missing values are filled in with None (option type: more on that below).

```{code-cell} ipython3
ak.Array([{"x": 1, "y": [1, 2]}, {"x": 2}])
```

If some tuples have different lengths, the resulting Awkward Array is taken to be heterogeneous (union type: more on that below).

```{code-cell} ipython3
ak.Array([(1, [1, 2]), (2,)])
```

An Awkward Record is a scalar drawn from a record array, so an {class}`ak.Record` can be built from a single dict with string-valued keys.

```{code-cell} ipython3
ak.Record({"x": 1, "y": [1, 2], "z": 3.3})
```

The same is not true for tuples. The {class}`ak.Record` constructor expects named fields.

```{code-cell} ipython3
:tags: [raises-exception]

ak.Record((1, [1, 2], 3.3))
```

Missing values: Python None
---------------------------

Python's None can appear anywhere in the structure parsed by {func}`ak.from_iter`. It makes all data at that level of nesting have option type and is represented in {func}`ak.to_list` as None.

```{code-cell} ipython3
ak.Array([1.1, 2.2, None, 3.3, None, 4.4])
```

```{code-cell} ipython3
ak.Array([1.1, 2.2, None, 3.3, None, 4.4]).to_list()
```

```{note}
Advanced topic: the rest of this section describes the equivalence of missing record fields and record fields with None values, which is only relevant to datasets with missing fields.
```

As described above, fields that are absent from some records but not others are filled in with None. As a consequence, conversions from Python to Awkward Array back to Python don't necessarily result in the original expression:

```{code-cell} ipython3
ak.Array(
    [
        {"x": 1.1, "y": [1]},
        {"x": 2.2, "z": "two"},
        {"x": 3.3, "y": [1, 2, 3], "z": "three"},
    ]
).to_list()
```

This is a deliberate choice. It would have been possible to convert records with missing fields into arrays with union type (more on that below), for which {class}`ak.to_list` would result in the original expression,

```{code-cell} ipython3
ak.concatenate(
    [
        ak.Array([{"x": 1.1, "y": [1]}]),
        ak.Array([{"x": 2.2, "z": "two"}]),
        ak.Array([{"x": 3.3, "y": [1, 2, 3], "z": "three"}]),
    ]
).to_list()
```

But typical datasets of records with different sets of fields represent missing fields, rather than entirely different types of objects. (Even in particle physics applications that mix "electron objects" with "photon objects," both types of objects have the same trajectory fields `"x"`, `"y"`, `"z"` and differ in fields that exist for one and not the other, such as `"charge"` for electrons but not photons.)

The memory use of union arrays scales with the number of different types, up to $2^n$ for records with $n$ potentially missing fields. Option types of completely disjoint records with $n_1$ and $n_2$ fields use a memory footprint that scales as $n_1 + n_2$. Assuming that disjoint records are a single record type with missing fields is a recoverable mistake, but assuming that a single record type with missing fields are distinct for every combination of missing fields is potentially disastrous.

Tuples of different lengths, on the other hand, are assumed to be different types because mistaking slot $i$ for slot $i + 1$ would create unions anyway.

```{code-cell} ipython3
ak.Array(
    [
        (1.1, [1]),
        (2.2, "two"),
        (3.3, [1, 2, 3], "three"),
    ]
).to_list()
```

Union types: heterogeneous data
-------------------------------

If the data in a Python iterable have different types at the same level of nesting ("heterogeneous"), the Awkward Arrays produced by {func}`ak.from_iter` have union types.

Most Awkward operations are defined on union typed Arrays, but they're not generally not as efficient as the same operations on simply typed Arrays.

The following example mixes numbers (`float64`) with lists (`var * int64`).

```{code-cell} ipython3
ak.Array([1.1, 2.2, [], [1], [1, 2], 3.3])
```

The {func}`ak.to_list` function converts it back into a heterogeneous Python list.

```{code-cell} ipython3
ak.Array([1.1, 2.2, [], [1], [1, 2], 3.3]).to_list()
```

Any types may be mixed: numbers and lists, lists and records, missing data, etc.

```{code-cell} ipython3
ak.Array([[1, 2, 3], {"x": 1, "y": 2}, None])
```

One exception is that numerical data are merged without creating a union type: integers are expanded to floating point numbers.

```{code-cell} ipython3
ak.Array([1, 2, 3, 4, 5.5, 6.6, 7.7, 8, 9])
```

But booleans are not merged with integers.

```{code-cell} ipython3
ak.Array([1, 2, 3, True, True, False, 4, 5])
```

As described above, records with different sets of fields are presumed to be a single record type with missing values.

```{code-cell} ipython3
ak.type(
    ak.Array(
        [
            {"x": 1.1, "y": [1]},
            {"x": 2.2, "z": "two"},
            {"x": 3.3, "y": [1, 2, 3], "z": "three"},
        ]
    )
)
```

But tuples with different lengths are presumed to be distinct types.

```{code-cell} ipython3
ak.type(
    ak.Array(
        [
            (1.1, [1]),
            (2.2, "two"),
            (3.3, [1, 2, 3], "three"),
        ]
    )
)
```

More control over conversions
-----------------------------

The conversions described above are applied by {func}`ak.from_iter` when it maps data into an {class}`ak.ArrayBuilder`. For more control over the conversion process (e.g. to make unions of records), use {class}`ak.ArrayBuilder` directly.
