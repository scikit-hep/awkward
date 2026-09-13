# ak.Record

Defined in [awkward.highlevel](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/highlevel.py) on [line 1818](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/highlevel.py#L1818).

#### *class* ak.Record(data, \*, behavior=None, with_name=None, check_valid=False, backend=None, attrs=None, named_axis=None)

* **Parameters:**
  * **data** ([`ak.record.Record`](sphinx-llm:c769a66df3124cdc8442e55ad70f6980#ak.record.Record), [`ak.Record`](sphinx-llm:fcd355d6f1ad4539be1909f743dcc19e), str, or dict) – Data to wrap or convert into a record.
    If a string, the data are assumed to be JSON.
    If a dict, calls [`ak.from_iter`](sphinx-llm:40c924e275044a0993aef7f20508d933#ak.from_iter), which assumes all inner
    dimensions have irregular lengths.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for this Record only.
  * **with_name** (*None* *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Gives the record type a name that can be
    used to override its behavior (see below).
  * **check_valid** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, verify that the [`layout`](sphinx-llm:aac1988394ce48d2a97f54db1b1c7bc8) is valid.
  * **backend** (None, `"cpu"`, `"jax"`, `"cuda"`) – If `"cpu"`, the Array will be placed in
    main memory for use with other `"cpu"` Arrays and Records; if `"cuda"`,
    the Array will be placed in GPU global memory using CUDA; if `"jax"`, the structure
    is copied to the CPU for use with JAX. if None, the `data` are left untouched.

High-level record that can contain fields of any type.

Most users won’t be creating Records manually. This class primarily exists
to be overridden in the same way as [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array).

Records can be used in [Numba](http://numba.pydata.org/): they can be
passed as arguments to a Numba-compiled function or returned as return
values. The only limitation is that they cannot be *created*
inside the Numba-compiled function; to make outputs, consider
[`ak.ArrayBuilder`](sphinx-llm:2fb9ffd5c3a441658693b9cc500d5518#ak.ArrayBuilder).

See also [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) and [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior).

#### \_layout

#### \_behavior *= None*

#### \_attrs *= None*

#### *classmethod* \_\_init_subclass_\_(\*\*kwargs)

#### \_update_class(restore=None)

#### *property* attrs *: awkward._attrs.Attrs*

The mapping containing top-level metadata, which is serialised
with the record during pickling.

Keys prefixed with `@` are identified as “transient” attributes
which are discarded prior to pickling, permitting the storage of
non-pickleable types.

#### *property* layout

The [`ak.record.Record`](sphinx-llm:c769a66df3124cdc8442e55ad70f6980#ak.record.Record) that contains composable [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content)
elements to determine how the array is structured.

See [`ak.Array.layout`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array.layout) for a more complete description.

The [`ak.record.Record`](sphinx-llm:c769a66df3124cdc8442e55ad70f6980#ak.record.Record) is not a subclass of [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) in
Python and it is not composable with them: [`ak.record.Record`](sphinx-llm:c769a66df3124cdc8442e55ad70f6980#ak.record.Record) contains
one [`ak.contents.RecordArray`](sphinx-llm:2475335b802d4bf79c938eaf015c0299#ak.contents.RecordArray) (which is a [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content)), but
[`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) nodes cannot contain a [`ak.record.Record`](sphinx-llm:c769a66df3124cdc8442e55ad70f6980#ak.record.Record).

A [`ak.record.Record`](sphinx-llm:c769a66df3124cdc8442e55ad70f6980#ak.record.Record) is not an independent entity from its
[`ak.contents.RecordArray`](sphinx-llm:2475335b802d4bf79c938eaf015c0299#ak.contents.RecordArray); it’s really just a marker indicating which
element to select. The XML representation reflects that:

```pycon
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
```

#### *property* behavior

The `behavior` parameter passed into this Record’s constructor.

* If a dict, this `behavior` overrides the global [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior).
  : Any keys in the global [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) but not this `behavior` are
    still valid, but any keys in both are overridden by this
    `behavior`. Keys with a None value are equivalent to missing keys,
    so this `behavior` can effectively remove keys from the
    global [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior).
* If None, the Record defaults to the global [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior).

See [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for a list of recognized key patterns and their
meanings.

#### *property* positional_axis *: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...]*

#### *property* named_axis *: awkward._namedaxis.AxisMapping*

#### tolist()

Converts this Record into Python objects; same as [`ak.to_list`](sphinx-llm:c5d0843c37c34cb88bbec16becc6a6f0#ak.to_list)
(but without the underscore, like NumPy’s
[tolist](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.tolist.html)).

#### to_list()

Converts this Record into Python objects; same as [`ak.to_list`](sphinx-llm:c5d0843c37c34cb88bbec16becc6a6f0#ak.to_list).

#### *property* nbytes

The total number of bytes in all the [`ak.index.Index`](sphinx-llm:c5dc164fb63f49c0946c0b5dfb338435#ak.index.Index),
and [`ak.contents.NumpyArray`](sphinx-llm:6f8cd60f60b04ce7a9499f2967197e79#ak.contents.NumpyArray) buffers in this array tree.

It does not count buffers that must be kept in memory because
of ownership, but are not directly used in the array. Nor does it count
the (small) Python objects that reference the (large)
array buffers.

#### *property* fields

List of field names or tuple slot numbers (as strings) of this record.

If this is actually a tuple its fields are string representations of
integers, such as `"0"`, `"1"`, `"2"`, etc.

See also [`ak.fields`](sphinx-llm:056b2c11871845eaa8962e15bfbffd9e#ak.fields).

#### *property* is_tuple

If True, the top-most record structure has no named fields, i.e. it’s a tuple.

#### \_ipython_key_completions_()

#### \_\_iter_\_ *= None*

#### *property* type

The high-level type of this Record; same as [`ak.type`](sphinx-llm:65d3dedfc8ad48c98f3bacc6472ae8c8#ak.type).

Note that the outermost element of a Record’s type is always an
[`ak.types.ScalarType`](sphinx-llm:ad298acb59fd4c1bb1bf103171505226#ak.types.ScalarType), which .

The type of a [`ak.record.Record`](sphinx-llm:c769a66df3124cdc8442e55ad70f6980#ak.record.Record) (from [`ak.Array.layout`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array.layout)) is not
wrapped by an [`ak.types.ScalarType`](sphinx-llm:ad298acb59fd4c1bb1bf103171505226#ak.types.ScalarType).

#### *property* typestr

The high-level type of this Record, presented as a string.

#### \_\_getitem_\_(where)

* **Parameters:**
  **where** (*many types supported; see below*) – Index of positions to
  select from this Record.

Select items from the Record using an extension of NumPy’s (already
quite extensive) rules.

See [`ak.Array.__getitem__`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array.__getitem__) for a more complete description. Since
this is a record, the first item in the slice tuple must be a
string, selecting a field.

For example, with

```pycon
>>> record = ak.Record({"x": 3.3, "y": [1, 2, 3]})
```

we can select

```pycon
>>> record["x"]
3.3
>>> record["y"]
<Array [1, 2, 3] type='3 * int64'>
>>> record["y", 1]
2
```

#### \_\_setitem_\_(where, what)

* **Parameters:**
  * **where** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple) *of* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Field name to add data to the record.
  * **what** – Data to add as the new field.

For example:

```pycon
>>> record = ak.Record({"x": 3.3})
>>> record["y"] = 4
>>> record["z"] = {"another": "record"}
>>> record.show()
{x: 3.3,
 y: 4,
 z: {another: 'record'}}
```

See [`ak.with_field`](sphinx-llm:3c2fe1ca08b646e494d171d5ea7f2e33#ak.with_field) for a variant that does not change the [`ak.Record`](sphinx-llm:fcd355d6f1ad4539be1909f743dcc19e)
in-place. (Internally, this method uses [`ak.with_field`](sphinx-llm:3c2fe1ca08b646e494d171d5ea7f2e33#ak.with_field), so performance
is not a factor in choosing one over the other.)

#### \_\_delitem_\_(where)

* **Parameters:**
  **where** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple) *of* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Field name to remove from the record.

For example:

```pycon
>>> record = ak.Record({"x": 3.3, "y": {"this": 10, "that": 20}})
>>> del record["y", "that"]
>>> record.show()
{x: 3.3,
 y: {this: 10}}
```

See [`ak.without_field`](sphinx-llm:f7f408e734c14ae58c061536a46aa7a2#ak.without_field) for a variant that does not change the [`ak.Record`](sphinx-llm:fcd355d6f1ad4539be1909f743dcc19e)
in-place. (Internally, this method uses [`ak.without_field`](sphinx-llm:f7f408e734c14ae58c061536a46aa7a2#ak.without_field), so performance
is not a factor in choosing one over the other.)

#### \_\_getattr_\_(where)

Whenever possible, fields can be accessed as attributes.

For example, the fields of

```pycon
>>> record = ak.Record({"x": 1.1, "y": [2, 2], "z": "three"})
```

can be accessed as

```pycon
>>> record.x
1.1
>>> record.y
<Array [2, 2] type='2 * int64'>
>>> record.z
'three'
```

which are equivalent to `record["x"]`, `record["y"]`, and
`record["z"]`.

Fields can’t be accessed as attributes when

* [`ak.Record`](sphinx-llm:fcd355d6f1ad4539be1909f743dcc19e) methods or properties take precedence,
* a domain-specific behavior has methods or properties that take
  : precedence, or
* the field name is not a valid Python identifier or is a Python
  : keyword.

#### \_\_setattr_\_(name, value)

* **Parameters:**
  **where** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Attribute name to set

Set an attribute on the record.

Only existing public attributes e.g. [`ak.Record.layout`](sphinx-llm:aac1988394ce48d2a97f54db1b1c7bc8), or private
attributes (with leading underscores), can be set.

Fields are not assignable to as attributes, i.e. the following doesn’t work:

```default
record.z = new_field
```

Instead, always use [`ak.Record.__setitem__`](sphinx-llm:a2921cabba39441cb84439d6a335b1db):

```default
record["z"] = new_field
```

or [`ak.with_field`](sphinx-llm:3c2fe1ca08b646e494d171d5ea7f2e33#ak.with_field):

```default
record = ak.with_field(record, new_field, "z")
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

Display the contents of the record within `limit_rows` and `limit_cols`, using
ellipsis (`...`) for hidden nested data.

The `formatter` argument controls the formatting of individual values, c.f.
[https://numpy.org/doc/stable/reference/generated/numpy.set_printoptions.html](https://numpy.org/doc/stable/reference/generated/numpy.set_printoptions.html)
As Awkward Array does not implement strings as a NumPy dtype, the `numpystr`
key is ignored; instead, a `"bytes"` and/or `"str"` key is considered when formatting
string values, falling back upon `"str_kind"`.

#### \_repr_mimebundle_(include=None, exclude=None)

#### \_\_array_ufunc_\_(ufunc, method, \*inputs, \*\*kwargs)

Intercepts attempts to pass this Record to a NumPy
[universal functions](https://docs.scipy.org/doc/numpy/reference/ufuncs.html)
(ufuncs) and passes it through the Record’s structure.

This method conforms to NumPy’s
[NEP 13](https://numpy.org/neps/nep-0013-ufunc-overrides.html)
for overriding ufuncs, which has been
[available since NumPy 1.13](https://numpy.org/devdocs/release/1.13.0-notes.html#array-ufunc-added)
(and thus NumPy 1.13 is the minimum allowed version).

See [`ak.Array.__array_ufunc__`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array.__array_ufunc__) for a more complete description.

#### *property* numba_type

The type of this Record when it is used in Numba. It contains enough
information to generate low-level code for accessing any element,
down to the leaves.

See [Numba documentation](https://numba.pydata.org/numba-doc/dev/reference/types.html)
on types and signatures.

#### \_\_reduce_ex_\_(protocol: [int](https://docs.python.org/3/library/functions.html#int)) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)

#### \_\_setstate_\_(state)

#### \_\_copy_\_()

#### \_\_deepcopy_\_(memo)

#### \_\_bool_\_()
