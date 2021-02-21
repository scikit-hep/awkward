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

The same is true for tuples, but 

+++

Records and unions
------------------

remember that naming changes the behavior
