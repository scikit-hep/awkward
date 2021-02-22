---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.6.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

How to create arrays with ArrayBuilder (easy and general)
=========================================================

If you're not getting data from a file or conversion from another format, you may need to create it from scratch. [ak.ArrayBuilder](https://awkward-array.readthedocs.io/en/latest/_auto/ak.ArrayBuilder.html) is a general, high-level way to do that, though it has performance limitations.

The biggest difference between an [ak.ArrayBuilder](https://awkward-array.readthedocs.io/en/latest/_auto/ak.ArrayBuilder.html) and an [ak.Array](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Array.html) is that you can append data to a builder, but an array is immutable ([see qualifications on mutability](how-to-convert-numpy.html#mutability-of-awkward-arrays-from-numpy)). It's a bit like a Python list, which has an `append` method, but [ak.ArrayBuilder](https://awkward-array.readthedocs.io/en/latest/_auto/ak.ArrayBuilder.html) has many methods for appending different types of structures.

+++

Appending
---------

```{code-cell} ipython3
import awkward as ak
```

When a builder is first created, it has zero length and unknown type.

```{code-cell} ipython3
builder = ak.ArrayBuilder()
builder
```

Calling its `append` method adds data and also determines its type.

```{code-cell} ipython3
builder.append(1)
builder
```

```{code-cell} ipython3
builder.append(2.2)
builder
```

```{code-cell} ipython3
builder.append(3+1j)
builder
```

Note that this can include missing data by promoting to an [option-type](https://awkward-array.readthedocs.io/en/latest/ak.types.OptionType.html),

```{code-cell} ipython3
builder.append(None)
builder
```

and mix types by promoting to a [union-type](https://awkward-array.readthedocs.io/en/latest/ak.types.UnionType.html):

```{code-cell} ipython3
builder.append("five")
builder
```

```{code-cell} ipython3
builder.type
```

We've been using "`append`" because it is generic (it recognizes the types of its arguments and builds that), but there are also methods for building structure explicitly.

```{code-cell} ipython3
builder = ak.ArrayBuilder()
builder.boolean(False)
builder.integer(1)
builder.real(2.2)
builder.complex(3+1j)
builder.null()
builder.string("five")
builder
```

```{code-cell} ipython3
builder.type
```

Snapshot
--------

To turn an [ak.ArrayBuilder](https://awkward-array.readthedocs.io/en/latest/_auto/ak.ArrayBuilder.html) into an [ak.Array](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Array.html), call `snapshot`. This is an inexpensive operation (may be done multiple times; the builder is unaffacted).

```{code-cell} ipython3
array = builder.snapshot()
array
```

Builders don't have all the high-level methods that arrays do, so if you want to use the array for normal analysis, remember to take a snapshot.

+++

Nested lists
------------

The most useful of these create nested data structures:

   * `begin_list`/`end_list`
   * `begin_record`/`end_record`
   * `begin_tuple`/`end_tuple`

which switch into a mode that starts filling inside of a list, record, or tuple. For records and tuples, you additionally have to specify the `field` or `index` of the record or tuple (respectively).

```{code-cell} ipython3
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

```{code-cell} ipython3
builder.append(9.9)
builder
```

This `9.9` is outside of the lists, and hence the type is now "lists of numbers *or* numbers."

```{code-cell} ipython3
builder.type
```

Since `begin_list` and `end_list` are imperative, the nesting structure of an array can be determined by program flow:

```{code-cell} ipython3
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

Often, you'll know the exact depth of nesting you want. The Python `with` statement can be used to restrict the generality (nd free you from having to remember to `end` what you `begin`).

```{code-cell} ipython3
builder = ak.ArrayBuilder()

with builder.list():
    with builder.list():
        builder.append(1)
        builder.append(2)
        builder.append(3)

builder
```

(Note that the Python `with` statement, a.k.a. "context manager," is not available in Numba jit-compiled functions, in case you're using [ak.ArrayBuilder](https://awkward-array.readthedocs.io/en/latest/_auto/ak.ArrayBuilder.html) in Numba.)

+++

Nested records
--------------

When using `begin_record`/`end_record` (or the equivalent `record` in the `with` statement), you have to specify which field each "`append`" is associated with.

   * `field("fieldname")`: switches to fill a field with a given name (and returns the builder, for convenience).

```{code-cell} ipython3
builder = ak.ArrayBuilder()

with builder.record():
    builder.field("x").append(1)
    builder.field("y").append(2.2)
    builder.field("z").append("three")

builder
```

The record type can also be given a name.

```{code-cell} ipython3
builder = ak.ArrayBuilder()

with builder.record("Point"):
    builder.field("x").real(1.1)
    builder.field("y").real(2.2)
    builder.field("z").real(3.3)

builder
```

This gives the resulting records a type named "`Point`", which might have [specialized behaviors](how-to-specialize.html).

```{code-cell} ipython3
array = builder.snapshot()
array
```

```{code-cell} ipython3
array.type
```

Nested tuples
-------------

The same is true for tuples, but the next field to fill is selected by "`index`" (integer), rather than "`field`" (string), and the tuple size has to be given up-front.

```{code-cell} ipython3
builder = ak.ArrayBuilder()

with builder.tuple(3):
    builder.index(0).append(1)
    builder.index(1).append(2.2)
    builder.index(2).append("three")

builder
```

Records and unions
------------------

If the set of fields changes while collecting records, the builder algorithm could handle it one of two possible ways:

   1. Assume that the new field or fields have simply been missing up to this point, and that any now-unspecified fields are also missing.
   2. Assume that a different set of fields means a different type and make a union.

By default, [ak.ArrayBuilder](https://awkward-array.readthedocs.io/en/latest/_auto/ak.ArrayBuilder.html) follows policy (1), but it can be made to follow policy (2) if the names of the records are different.

```{code-cell} ipython3
policy1 = ak.ArrayBuilder()

with policy1.record():
    policy1.field("x").append(1)
    policy1.field("y").append(1.1)

with policy1.record():
    policy1.field("y").append(2.2)
    policy1.field("z").append("three")

print(policy1)
policy1.type
```

```{code-cell} ipython3
policy2 = ak.ArrayBuilder()

with policy2.record("First"):
    policy2.field("x").append(1)
    policy2.field("y").append(1.1)

with policy2.record("Second"):
    policy2.field("y").append(2.2)
    policy2.field("z").append("three")

print(policy2)
policy2.type
```

Comments on union-type
----------------------

Although it's easy to make [union-type](https://awkward-array.readthedocs.io/en/latest/ak.types.UnionType.html) data with [ak.ArrayBuilder](https://awkward-array.readthedocs.io/en/latest/_auto/ak.ArrayBuilder.html), the applications of union-type data are more limited. For instance, we can select a field that belongs to _all_ types of the union, but not any fields that don't share that field.

```{code-cell} ipython3
array2 = policy2.snapshot()
array2
```

```{code-cell} ipython3
array2.y
```

```{code-cell} ipython3
:tags: [raises-exception]

array2.x
```

The above would be no problem for records collected using policy 1 (see previous section).

```{code-cell} ipython3
array1 = policy1.snapshot()
array1
```

```{code-cell} ipython3
array1.y
```

```{code-cell} ipython3
array1.x
```

At the time of writing, [union-types](https://awkward-array.readthedocs.io/en/latest/ak.types.UnionType.html) are not supported in Numba ([issue 174](https://github.com/scikit-hep/awkward-1.0/issues/174)).

+++

Use in Numba
------------

[ak.ArrayBuilder](https://awkward-array.readthedocs.io/en/latest/_auto/ak.ArrayBuilder.html) can be used in Numba-compiled functions, and that can often be the most convenient way to build up an array, relatively quickly (see below).

There are a few limitations, though:

   * At the time of writing, [Numba doesn't support Python's `with` statement](https://numba.pydata.org/numba-doc/dev/reference/pysupported.html#language) (context manager), so `begin_list`/`end_list` will have to be used instead.
   * Builders cannot be constructed inside of the compiled function; they have to be passed in.
   * The `snapshot` method cannot be called inside of the compiled function; it has to be applied to the output.

Therefore, a common pattern is:

```{code-cell} ipython3
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

Appending parts of an existing array
------------------------------------

If the argument of the `append` function is part of another Awkward Array, that array will be *linked into* the new array, rather than reconstructing the original by iterating over it. That can be a performance advantage (appending records with 1000 fields takes as much time as appending records with 1 field), but it can prevent large data structures from being garbage-collected, because a reference to them exists in the new array.

```{code-cell} ipython3
original = ak.Array([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}])

builder = ak.ArrayBuilder()
builder.append(original[2])
builder.append(original[1])
builder.append(original[0])
builder.append(original[1])
builder.append(original[1])
builder.append(original[0])
builder.append(original[2])
builder.append(original[2])

new_array = builder.snapshot()
new_array
```

```{code-cell} ipython3
new_array.layout
```

Above, we see that `new_array` is just making references ([ak.layout.IndexedArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.IndexedArray.html)) of an [ak.layout.RecordArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.RecordArray.html) with `x = [1, 2, 3]` and `y = [1.1, 2.2, 3.3]`.

+++

Comments on performance
-----------------------

Although [ak.ArrayBuilder](https://awkward-array.readthedocs.io/en/latest/_auto/ak.ArrayBuilder.html) is implemented in C++, it is dynamically typed by design. The advantage of compiled code over interpreted code often comes in the knowledge of data types at compile-time, enabling fewer runtime checks and more compiler optimizations.

If you're using a builder in Python, there's also the overhead of calling from Python.

If you're using a builder in Numba, the builder calls are external function calls and LLVM can't inline them for optimizations.

Whenever you have a choice between

   1. using the [ak.ArrayBuilder](https://awkward-array.readthedocs.io/en/latest/_auto/ak.ArrayBuilder.html),
   2. constructing an array manually from layouts (next chapter), or
   3. filling a NumPy array and using it as an index,

the alternatives are often faster. The point of [ak.ArrayBuilder](https://awkward-array.readthedocs.io/en/latest/_auto/ak.ArrayBuilder.html) is that it is *easy*.
