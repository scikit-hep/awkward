# ak.ArrayBuilder

Defined in [awkward.highlevel](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/highlevel.py) on [line 2610](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/highlevel.py#L2610).

#### *class* ak.ArrayBuilder(\*, behavior=None, attrs=None, initial=1024, resize=8)

* **Parameters:**
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for arrays built by
    this ArrayBuilder.
  * **initial** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Initial size (in bytes) of buffers used by the `ak::ArrayBuilder`.
  * **resize** ([*float*](https://docs.python.org/3/library/functions.html#float)) – Resize multiplier for buffers used by the `ak::ArrayBuilder`;
    should be strictly greater than 1.

General tool for building arrays of nested data structures from a sequence
of commands. Most data types can be constructed by calling commands in the
right order, similar to printing tokens to construct JSON output.

To illustrate how this works, consider the following example.:

```default
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
```

To get an array, we take a [`snapshot`](sphinx-llm:e9f141424d5f4dc69c9775a563682cde) of the ArrayBuilder’s current state.

```pycon
>>> b.snapshot()
<Array [[1, 2, 3], ..., [{x: 1, y: ..., ...}]] type='4 * var * union[?float...'>
>>> b.snapshot().show()
[[1, 2, 3],
 [],
 [4, None, 5],
 [{x: 1, y: [2, 3]}]]
```

The full set of filling commands is the following.

* [`null`](sphinx-llm:b5c575848cd24d0d990e0430b9a469b8): appends a None value.
* [`boolean`](sphinx-llm:464be01ffcfb4abea327f7b19166f3f5): appends True or False.
* [`integer`](sphinx-llm:4438c3b63712470aace47ada10d656da): appends an integer.
* [`real`](sphinx-llm:f8e3e1bc96c54ff39307a5158e1909da): appends a floating-point value.
* [`complex`](sphinx-llm:7f16c6d752864d16bf7ef09bbc5451d7): appends a complex value.
* [`datetime`](sphinx-llm:e642f34286b2441f95b4a28c73e25f86): appends a datetime value.
* [`timedelta`](sphinx-llm:25c12b808d5f4924b13516d987753df4): appends a timedelta value.
* [`bytestring`](sphinx-llm:43c0fb7aeb004750bb98fb6ca37faa4f): appends an unencoded string (raw bytes).
* [`string`](sphinx-llm:3c12d49cd12944398ca538e567c398e2): appends a UTF-8 encoded string.
* [`begin_list`](sphinx-llm:3dc85755454c45d9b85465a065bf7604): begins filling a list; must be closed with [`end_list`](sphinx-llm:79072660265e4ce0922746144d0d93ba).
* [`end_list`](sphinx-llm:79072660265e4ce0922746144d0d93ba): ends a list.
* [`begin_tuple`](sphinx-llm:5462ba1e1fc54afb8d6166f14c32d21a): begins filling a tuple; must be closed with [`end_tuple`](sphinx-llm:6173b04fc7db4fa2af7a8544cca232f5).
* [`index`](sphinx-llm:a09ace0d4fba4d71a85ae9fdbe1cb27a): selects a tuple slot to fill; must be followed by a command
  : that actually fills that slot.
* [`end_tuple`](sphinx-llm:6173b04fc7db4fa2af7a8544cca232f5): ends a tuple.
* [`begin_record`](sphinx-llm:e7bd0b6cd8234387bcfafcb426cd2cdc): begins filling a record; must be closed with
  : [`end_record`](sphinx-llm:56cc4d23a46149d79dbdb3871422a57f).
* [`field`](sphinx-llm:388e123044de42e7b38a6741a85ef151): selects a record field to fill; must be followed by a command
  : that actually fills that field.
* [`end_record`](sphinx-llm:56cc4d23a46149d79dbdb3871422a57f): ends a record.
* [`append`](sphinx-llm:9045cab409fa4f8b9a2d951bc988a69c): generic method for filling [`null`](sphinx-llm:b5c575848cd24d0d990e0430b9a469b8), [`boolean`](sphinx-llm:464be01ffcfb4abea327f7b19166f3f5), [`integer`](sphinx-llm:4438c3b63712470aace47ada10d656da), [`real`](sphinx-llm:f8e3e1bc96c54ff39307a5158e1909da),
  : [`bytestring`](sphinx-llm:43c0fb7aeb004750bb98fb6ca37faa4f), [`string`](sphinx-llm:3c12d49cd12944398ca538e567c398e2), [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array), [`ak.Record`](sphinx-llm:e158df99f80043309c758f6e933352b2#ak.Record), or arbitrary Python data.
* [`extend`](sphinx-llm:277aa1fba9604a7d85f97967d45070f0): appends all the items from an iterable.
* [`list`](sphinx-llm:67dc96f2da874405bad91442a650bb23): context manager for [`begin_list`](sphinx-llm:3dc85755454c45d9b85465a065bf7604) and [`end_list`](sphinx-llm:79072660265e4ce0922746144d0d93ba).
* [`tuple`](sphinx-llm:dfc711e033614349b2d229fd75389268): context manager for [`begin_tuple`](sphinx-llm:5462ba1e1fc54afb8d6166f14c32d21a) and [`end_tuple`](sphinx-llm:6173b04fc7db4fa2af7a8544cca232f5).
* [`record`](sphinx-llm:8f9252405c5644ba8be7b7403842a2c0): context manager for [`begin_record`](sphinx-llm:e7bd0b6cd8234387bcfafcb426cd2cdc) and [`end_record`](sphinx-llm:56cc4d23a46149d79dbdb3871422a57f).

ArrayBuilders can be used in [Numba](http://numba.pydata.org/): they can
be passed as arguments to a Numba-compiled function or returned as return
values. (Since ArrayBuilder works by accumulating side-effects, it’s not
strictly necessary to return the object.)

The primary limitation is that ArrayBuilders cannot be *created* and
[`snapshot`](sphinx-llm:e9f141424d5f4dc69c9775a563682cde) cannot be called inside the Numba-compiled function. Awkward
Array uses Numba as a transformer: [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) and an empty [`ak.ArrayBuilder`](sphinx-llm:a85e9cc757b7437183e23b2c761cec2c)
go in and a filled [`ak.ArrayBuilder`](sphinx-llm:a85e9cc757b7437183e23b2c761cec2c) is the result; [`snapshot`](sphinx-llm:e9f141424d5f4dc69c9775a563682cde) can be called
outside of the compiled function.

Also, context managers (Python’s `with` statement) are not supported in
Numba yet, so the [`list`](sphinx-llm:67dc96f2da874405bad91442a650bb23), [`tuple`](sphinx-llm:dfc711e033614349b2d229fd75389268), and [`record`](sphinx-llm:8f9252405c5644ba8be7b7403842a2c0) methods are not available
in Numba-compiled functions.

Here is an example of filling an ArrayBuilder in Numba, which makes a
tree of dynamic depth.

```pycon
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
```

Note that this is a *general* method for building arrays; if the type is
known in advance, more specialized procedures can be faster. This should
be considered the “least effort” approach.

#### \_layout

#### \_behavior *= None*

#### \_attrs *= None*

#### *classmethod* \_wrap(layout, behavior=None, attrs=None)

* **Parameters:**
  * **layout** (`ak._ext.ArrayBuilder`) – Low-level builder to wrap.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for arrays built by
    this ArrayBuilder.

Wraps a low-level `ak._ext.ArrayBuilder` as a high-level
`ak.ArrayBulider`.

The [`ak.ArrayBuilder`](sphinx-llm:a85e9cc757b7437183e23b2c761cec2c) constructor creates a new `ak._ext.ArrayBuilder`
with no accumulated data, but Numba needs to wrap existing data
when returning from a lowered function.

#### *property* attrs *: awkward._attrs.Attrs*

The mapping containing top-level metadata, which is serialised
with the array during pickling.

Keys prefixed with `@` are identified as “transient” attributes
which are discarded prior to pickling, permitting the storage of
non-pickleable types.

#### *property* behavior

The `behavior` parameter passed into this ArrayBuilder’s constructor.

* If a dict, this `behavior` overrides the global [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior).
  : Any keys in the global [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) but not this `behavior` are
    still valid, but any keys in both are overridden by this
    `behavior`. Keys with a None value are equivalent to missing keys,
    so this `behavior` can effectively remove keys from the
    global [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior).
* If None, the Array defaults to the global [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior).

See [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for a list of recognized key patterns and their
meanings.

#### tolist()

Converts this Array into Python objects; same as [`ak.to_list`](sphinx-llm:c5d0843c37c34cb88bbec16becc6a6f0#ak.to_list)
(but without the underscore, like NumPy’s
[tolist](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.tolist.html)).

#### to_list()

Converts this Array into Python objects; same as [`ak.to_list`](sphinx-llm:c5d0843c37c34cb88bbec16becc6a6f0#ak.to_list).

#### to_numpy(allow_missing=True)

Converts this Array into a NumPy array, if possible; same as [`ak.to_numpy`](sphinx-llm:0dc12c9b410c4566b8e6578a4b063dbf#ak.to_numpy).

#### *property* type

The high-level type of the accumulated array; same as [`ak.type`](sphinx-llm:65d3dedfc8ad48c98f3bacc6472ae8c8#ak.type).

Note that the outermost element of an Array’s type is always an
[`ak.types.ArrayType`](sphinx-llm:53688a8c7ce14810a6b2ed10d56f008f#ak.types.ArrayType), which specifies the number of elements in the array.

The type of a [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) (from [`ak.Array.layout`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array.layout)) is not
wrapped by an [`ak.types.ArrayType`](sphinx-llm:53688a8c7ce14810a6b2ed10d56f008f#ak.types.ArrayType).

#### *property* typestr

The high-level type of this accumulated array, presented as a string.

#### \_\_len_\_()

The current length of the accumulated array.

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

Display the contents of the array builder within `limit_rows` and `limit_cols`, using
ellipsis (`...`) for hidden nested data.

The `formatter` argument controls the formatting of individual values, c.f.
[https://numpy.org/doc/stable/reference/generated/numpy.set_printoptions.html](https://numpy.org/doc/stable/reference/generated/numpy.set_printoptions.html)
As Awkward Array does not implement strings as a NumPy dtype, the `numpystr`
key is ignored; instead, a `"bytes"` and/or `"str"` key is considered when formatting
string values, falling back upon `"str_kind"`.

This method takes a snapshot of the data and calls show on it, and a snapshot
copies data.

#### \_\_array_\_(dtype=None, copy=None)

Intercepts attempts to convert a #snapshot of this array into a
NumPy array and either performs a conversion if possible or raises an error.

See [`ak.Array.__array__`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array.__array__) for a more complete description.

#### \_\_arrow_array_\_(type=None)

#### *property* numba_type

The type of this Array when it is used in Numba. It contains enough
information to generate low-level code for accessing any element,
down to the leaves.

See [Numba documentation](https://numba.pydata.org/numba-doc/dev/reference/types.html)
on types and signatures.

#### \_\_bool_\_()

#### snapshot()

Converts the currently accumulated data into an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array).

The currently accumulated data are *copied* into the new array.

#### null()

Appends a None value at the current position in the accumulated array.

#### boolean(x)

Appends a boolean value `x` at the current position in the accumulated
array.

#### integer(x)

Appends an integer `x` at the current position in the accumulated
array.

#### real(x)

Appends a floating point number `x` at the current position in the
accumulated array.

#### complex(x)

Appends a floating point number `x` at the current position in the
accumulated array.

#### datetime(x)

Appends a datetime value `x` at the current position in the
accumulated array.

#### timedelta(x)

Appends a timedelta value `x` at the current position in the
accumulated array.

#### bytestring(x)

Appends an unencoded string (raw bytes) `x` at the current position
in the accumulated array.

#### string(x)

Appends a UTF-8 encoded string `x` at the current position in the
accumulated array.

#### begin_list()

Begins filling a list; must be closed with #end_list.

For example,

```pycon
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
```

produces

```pycon
>>> builder.show()
[[1.1, 2.2, 3.3],
 [],
 [4.4, 5.5]]
```

#### end_list()

Ends a list.

#### begin_tuple(numfields)

Begins filling a tuple with `numfields` fields; must be closed with
#end_tuple.

For example,

```pycon
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
```

produces

```pycon
>>> builder.show()
[(1, 1.1, 'one'),
 (2, 2.2, 'two')]
```

#### index(i)

* **Parameters:**
  **i** ([*int*](https://docs.python.org/3/library/functions.html#int)) – The tuple slot to fill.

This method also returns the [`ak.ArrayBuilder`](sphinx-llm:a85e9cc757b7437183e23b2c761cec2c), so that it can be
chained with the value that fills the slot.

Prepares to fill a tuple slot; see #begin_tuple for an example.

#### end_tuple()

Ends a tuple.

#### begin_record(name=None)

Begins filling a record with an optional `name`; must be closed with
#end_record.

For example,

```pycon
>>> builder = ak.ArrayBuilder()
>>> builder.begin_record("points")
>>> builder.field("x").real(1)
>>> builder.field("y").real(1.1)
>>> builder.end_record()
>>> builder.begin_record("points")
>>> builder.field("x").real(2)
>>> builder.field("y").real(2.2)
>>> builder.end_record()
```

produces

```pycon
>>> builder.show()
[{x: 1, y: 1.1},
 {x: 2, y: 2.2}]
```

with type

```pycon
>>> builder.type.show()
2 * points[
    x: float64,
    y: float64
]
```

The record type is named `"points"` because its `"__record__"`
parameter is set to that value:

```pycon
>>> builder.snapshot().layout.parameters
{'__record__': 'points'}
```

The `"__record__"` parameter can be used to add behavior to the records
in the array, as described in [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array), [`ak.Record`](sphinx-llm:e158df99f80043309c758f6e933352b2#ak.Record), and [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior).

#### field(key)

* **Parameters:**
  **key** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – The field key to fill.

This method also returns the [`ak.ArrayBuilder`](sphinx-llm:a85e9cc757b7437183e23b2c761cec2c), so that it can be
chained with the value that fills the slot.

Prepares to fill a field; see #begin_record for an example.

#### end_record()

Ends a record.

#### append(obj)

* **Parameters:**
  **obj** – The data to append (None, bool, int, float, bytes, str, or
  anything recognized by [`ak.from_iter`](sphinx-llm:40c924e275044a0993aef7f20508d933#ak.from_iter)).

Appends any type, which can be a shorthand for #null,
#boolean, #integer, #real, #bytestring, or #string, but also
an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) or [`ak.Record`](sphinx-llm:e158df99f80043309c758f6e933352b2#ak.Record) to *reference* values from an existing
dataset, or any Python object to *convert* to Awkward Array.

If `obj` is an iterable (including dict), this is equivalent to
[`ak.from_iter`](sphinx-llm:40c924e275044a0993aef7f20508d933#ak.from_iter) except that it fills an existing [`ak.ArrayBuilder`](sphinx-llm:a85e9cc757b7437183e23b2c761cec2c),
rather than creating a new one.

#### extend(obj)

* **Parameters:**
  **obj** (*iterable*) – Iterable of data to extend this ArrayBuilder with.

Appends every value from `obj`.

#### list()

Context manager to prevent unpaired #begin_list and #end_list. The
example in the #begin_list documentation can be rewritten as

```pycon
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
```

to produce the same result.

```pycon
>>> builder.show()
[[1.1, 2.2, 3.3],
 [],
 [4.4, 5.5]]
```

Since context managers aren’t yet supported by Numba, this method
can’t be used in Numba.

#### tuple(numfields)

Context manager to prevent unpaired #begin_tuple and #end_tuple. The
example in the #begin_tuple documentation can be rewritten as

```pycon
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
```

to produce the same result.

```pycon
>>> builder.show()
[(1, 1.1, 'one'),
 (2, 2.2, 'two')]
```

Since context managers aren’t yet supported by Numba, this method
can’t be used in Numba.

#### record(name=None)

Context manager to prevent unpaired #begin_record and #end_record. The
example in the #begin_record documentation can be rewritten as

```pycon
>>> builder = ak.ArrayBuilder()
>>> with builder.record("points"):
...     builder.field("x").real(1)
...     builder.field("y").real(1.1)
...
>>> with builder.record("points"):
...     builder.field("x").real(2)
...     builder.field("y").real(2.2)
...
```

to produce the same result.

```pycon
>>> builder.show()
[{x: 1, y: 1.1},
 {x: 2, y: 2.2}]
```

Since context managers aren’t yet supported by Numba, this method
can’t be used in Numba.

## Classes

| `_Nested`   |    |
|-------------|----|
| `List`      |    |
| `Tuple`     |    |
| `Record`    |    |
