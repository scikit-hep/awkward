# How to create arrays with ArrayBuilder (easy and general)

If you’re not getting data from a file or conversion from another format, you may need to create it from scratch. [`ak.ArrayBuilder`](sphinx-llm:2fb9ffd5c3a441658693b9cc500d5518#ak.ArrayBuilder) is a general, high-level way to do that, though it has performance limitations.

The biggest difference between an [`ak.ArrayBuilder`](sphinx-llm:2fb9ffd5c3a441658693b9cc500d5518#ak.ArrayBuilder) and an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) is that you can append data to a builder, but an array is immutable ([see qualifications on mutability]()). It’s a bit like a Python list, which has an `append` method, but [`ak.ArrayBuilder`](sphinx-llm:2fb9ffd5c3a441658693b9cc500d5518#ak.ArrayBuilder) has many methods for appending different types of structures.

## Appending

```ipython3
import awkward as ak
```

When a builder is first created, it has zero length and unknown type.

```ipython3
builder = ak.ArrayBuilder()
builder
```

Calling its `append` method adds data and also determines its type.

```ipython3
builder.append(1)
builder
```

```ipython3
builder.append(2.2)
builder
```

```ipython3
builder.append(3 + 1j)
builder
```

Note that this can include missing data by promoting to an [`option-type`](sphinx-llm:f8e0eb36e1334eef887fb5ab81458600#ak.types.OptionType),

```ipython3
builder.append(None)
builder
```

and mix types by promoting to a [`union-type`](sphinx-llm:5e1485b00d7b47f39f0082bb755fa92f#ak.types.UnionType):

```ipython3
builder.append("five")
builder
```

```ipython3
builder.type.show()
```

```myst-ansi
5 * union[
    ?complex128,
    ?string
]
```

We’ve been using “`append`” because it is generic (it recognizes the types of its arguments and builds that), but there are also methods for building structure explicitly.

```ipython3
builder = ak.ArrayBuilder()
builder.boolean(False)
builder.integer(1)
builder.real(2.2)
builder.complex(3 + 1j)
builder.null()
builder.string("five")
builder
```

```ipython3
builder.type.show()
```

```myst-ansi
6 * union[
    ?bool,
    ?complex128,
    ?string
]
```

## Snapshot

To turn an [`ak.ArrayBuilder`](sphinx-llm:2fb9ffd5c3a441658693b9cc500d5518#ak.ArrayBuilder) into an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array), call `snapshot`. This is an inexpensive operation (may be done multiple times; the builder is unaffected).

```ipython3
array = builder.snapshot()
array
```

Builders don’t have all the high-level methods that arrays do, so if you want to use the array for normal analysis, remember to take a snapshot.

## Nested lists

The most useful of these create nested data structures:

* `begin_list`/`end_list`
* `begin_record`/`end_record`
* `begin_tuple`/`end_tuple`

which switch into a mode that starts filling inside of a list, record, or tuple. For records and tuples, you additionally have to specify the `field` or `index` of the record or tuple (respectively).

```ipython3
builder = ak.ArrayBuilder()

builder.begin_list()
builder.append(1.1)
builder.append(2.2)
builder.append(3.3)
builder.end_list()

builder.begin_list()
builder.end_list()

builder.begin_list()
builder.append(4.4)
builder.append(5.5)
builder.end_list()

builder
```

Appending after the `begin_list` puts data inside the list, rather than outside:

```ipython3
builder.append(9.9)
builder
```

This `9.9` is outside of the lists, and hence the type is now “lists of numbers *or* numbers.”

```ipython3
builder.type.show()
```

```myst-ansi
4 * union[
    var * float64,
    float64
]
```

Since `begin_list` and `end_list` are imperative, the nesting structure of an array can be determined by program flow:

```ipython3
def arbitrary_nesting(builder, depth):
    if depth == 0:
        builder.append(1)
        builder.append(2)
        builder.append(3)
    else:
        builder.begin_list()
        arbitrary_nesting(builder, depth - 1)
        builder.end_list()


builder = ak.ArrayBuilder()
arbitrary_nesting(builder, 5)
builder
```

Often, you’ll know the exact depth of nesting you want. The Python `with` statement can be used to restrict the generality (nd free you from having to remember to `end` what you `begin`).

```ipython3
builder = ak.ArrayBuilder()

with builder.list():
    with builder.list():
        builder.append(1)
        builder.append(2)
        builder.append(3)

builder
```

(Note that the Python `with` statement, a.k.a. “context manager,” is not available in Numba jit-compiled functions, in case you’re using [`ak.ArrayBuilder`](sphinx-llm:2fb9ffd5c3a441658693b9cc500d5518#ak.ArrayBuilder) in Numba.)

## Nested records

When using `begin_record`/`end_record` (or the equivalent `record` in the `with` statement), you have to specify which field each “`append`” is associated with.

* `field("fieldname")`: switches to fill a field with a given name (and returns the builder, for convenience).

```ipython3
builder = ak.ArrayBuilder()

with builder.record():
    builder.field("x").append(1)
    builder.field("y").append(2.2)
    builder.field("z").append("three")

builder
```

The record type can also be given a name.

```ipython3
builder = ak.ArrayBuilder()

with builder.record("Point"):
    builder.field("x").real(1.1)
    builder.field("y").real(2.2)
    builder.field("z").real(3.3)

builder
```

This gives the resulting records a type named “`Point`”, which might have specialized behaviors.

```ipython3
array = builder.snapshot()
array
```

## Nested tuples

The same is true for tuples, but the next field to fill is selected by “`index`” (integer), rather than “`field`” (string), and the tuple size has to be given up-front.

```ipython3
builder = ak.ArrayBuilder()

with builder.tuple(3):
    builder.index(0).append(1)
    builder.index(1).append(2.2)
    builder.index(2).append("three")

builder
```

## Records and unions

If the set of fields changes while collecting records, the builder algorithm could handle it one of two possible ways:

1. Assume that the new field or fields have simply been missing up to this point, and that any now-unspecified fields are also missing.
2. Assume that a different set of fields means a different type and make a union.

By default, [`ak.ArrayBuilder`](sphinx-llm:2fb9ffd5c3a441658693b9cc500d5518#ak.ArrayBuilder) follows policy (1), but it can be made to follow policy (2) if the names of the records are different.

```ipython3
policy1 = ak.ArrayBuilder()

with policy1.record():
    policy1.field("x").append(1)
    policy1.field("y").append(1.1)

with policy1.record():
    policy1.field("y").append(2.2)
    policy1.field("z").append("three")

policy1.type.show()
```

```myst-ansi
2 * {
    x: ?int64,
    y: float64,
    z: ?string
}
```

```ipython3
policy2 = ak.ArrayBuilder()

with policy2.record("First"):
    policy2.field("x").append(1)
    policy2.field("y").append(1.1)

with policy2.record("Second"):
    policy2.field("y").append(2.2)
    policy2.field("z").append("three")

policy2.type.show()
```

```myst-ansi
2 * union[
    First[
        x: int64,
        y: float64
    ],
    Second[
        y: float64,
        z: string
    ]
]
```

## Comments on union-type

Although it’s easy to make [`union-type`](sphinx-llm:5e1485b00d7b47f39f0082bb755fa92f#ak.types.UnionType) data with [`ak.ArrayBuilder`](sphinx-llm:2fb9ffd5c3a441658693b9cc500d5518#ak.ArrayBuilder), the applications of union-type data are more limited. For instance, we can select a field that belongs to *all* types of the union, but not any fields that don’t share that field.

```ipython3
array2 = policy2.snapshot()
array2
```

```ipython3
array2.y
```

```ipython3
array2.x
```

```ipythontb
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
Cell In[25], line 1
----> 1 array2.x

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/highlevel.py:1312, in Array.__getattr__(self, where)
   1307         raise AttributeError(
   1308             f"while trying to get field {where!r}, an exception "
   1309             f"occurred:\n{type(err)}: {err!s}"
   1310         ) from err
   1311 else:
-> 1312     raise AttributeError(f"no field named {where!r}")

AttributeError: no field named 'x'
```

The above would be no problem for records collected using policy 1 (see previous section).

```ipython3
array1 = policy1.snapshot()
array1
```

```ipython3
array1.y
```

```ipython3
array1.x
```

At the time of writing, [`union-type`](sphinx-llm:5e1485b00d7b47f39f0082bb755fa92f#ak.types.UnionType) are not supported in Numba ([issue 174](https://github.com/scikit-hep/awkward-1.0/issues/174)).

## Use in Numba

[`ak.ArrayBuilder`](sphinx-llm:2fb9ffd5c3a441658693b9cc500d5518#ak.ArrayBuilder) can be used in Numba-compiled functions, and that can often be the most convenient way to build up an array, relatively quickly (see below).

There are a few limitations, though:

* At the time of writing, [Numba doesn’t support Python’s `with` statement](https://numba.pydata.org/numba-doc/dev/reference/pysupported.html#language) (context manager), so `begin_list`/`end_list` will have to be used instead.
* Builders cannot be constructed inside of the compiled function; they have to be passed in.
* The `snapshot` method cannot be called inside of the compiled function; it has to be applied to the output.

Therefore, a common pattern is:

```ipython3
import numba as nb


@nb.jit
def build(builder):
    builder.begin_list()
    builder.append(1.1)
    builder.append(2.2)
    builder.append(3.3)
    builder.end_list()
    builder.begin_list()
    builder.end_list()
    builder.begin_list()
    builder.append(4.4)
    builder.append(5.5)
    builder.end_list()
    return builder


array = build(ak.ArrayBuilder()).snapshot()
array
```

## Setting the type of empty lists

In addition to supporting type-discovery at execution time, [`ak.ArrayBuilder`](sphinx-llm:2fb9ffd5c3a441658693b9cc500d5518#ak.ArrayBuilder) also makes it convenient to work with complex, ragged arrays when the type is known ahead of time. Although it is not the most performant means of constructing an array whose type is already known, it provides a readable abstraction in the event that building the array is not a limiting factor for performance. However, due to this “on-line” type-discovery, it is possible that for certain data the result of [`ak.ArrayBuilder.snapshot()`](sphinx-llm:2fb9ffd5c3a441658693b9cc500d5518#ak.ArrayBuilder.snapshot) will have different types. Consider this function that builds an array from the contents of some iterable:

```ipython3
def process_data(builder, data):
    for item in data:
        if item < 0:
            builder.null()
        else:
            builder.integer(item)
    return builder.snapshot()
```

If we pass in only positive integers, the result is an array of integers:

```ipython3
process_data(
    ak.ArrayBuilder(),
    [1, 2, 3, 4],
)
```

If we pass in only negative integers, the result is an array of `None`s with an unknown type:

```ipython3
process_data(
    ak.ArrayBuilder(),
    [-1, -2, -3, -4],
)
```

It is only if we pass in a mix of these values that we see the “full” array type:

```ipython3
process_data(
    ak.ArrayBuilder(),
    [1, 2, 3, 4, -1, -2, -3, -4],
)
```

A simple way to solve this problem is to explore all code branches explicitly, and remove the generated entry(ies) from the final array:

```ipython3
def process_data(builder, data):
    for item in data:
        if item < 0:
            builder.null()
        else:
            builder.integer(item)

    # Ensure we have the proper type
    builder.integer(1)
    builder.null()
    return builder.snapshot()[:-2]
```

The previous examples now have the same type:

```ipython3
process_data(
    ak.ArrayBuilder(),
    [1, 2, 3, 4],
)
```

```ipython3
process_data(
    ak.ArrayBuilder(),
    [-1, -2, -3, -4],
)
```

```ipython3
process_data(
    ak.ArrayBuilder(),
    [1, 2, 3, 4, -1, -2, -3, -4],
)
```

## Comments on performance

Although [`ak.ArrayBuilder`](sphinx-llm:2fb9ffd5c3a441658693b9cc500d5518#ak.ArrayBuilder) is implemented in C++, it is dynamically typed by design. The advantage of compiled code over interpreted code often comes in the knowledge of data types at compile-time, enabling fewer runtime checks and more compiler optimizations.

If you’re using a builder in Python, there’s also the overhead of calling from Python.

If you’re using a builder in Numba, the builder calls are external function calls and LLVM can’t inline them for optimizations.

Whenever you have a choice between

1. using the [`ak.ArrayBuilder`](sphinx-llm:2fb9ffd5c3a441658693b9cc500d5518#ak.ArrayBuilder),
2. constructing an array manually from layouts (next chapter), or
3. filling a NumPy array and using it as an index,

the alternatives are often faster. The point of [`ak.ArrayBuilder`](sphinx-llm:2fb9ffd5c3a441658693b9cc500d5518#ak.ArrayBuilder) is that it is *easy*.
