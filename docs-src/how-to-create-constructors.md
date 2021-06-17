---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

Direct constructors (fastest)
=============================

If you're willing to think about your data in a columnar way, directly constructing layouts and wrapping them in [ak.Array](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Array.html) interfaces is the fastest way to make them. (All other methods do this at some level.)

"Thinking about data in a columnar way" is the crucial difference between this method and [ArrayBuilder](how-to-create-arraybuilder) and [LayoutBuilder](how-to-create-layoutbuilder). Both of the builders let you think about a data structure the way you would think about Python objects, in which all fields of a given record or elements of a list are "together" and one record or list is "separate" from another record or list. For example,

```{code-cell}
import awkward as ak
import numpy as np
```

```{code-cell}
builder = ak.ArrayBuilder()

with builder.list():
    with builder.record():
        builder.field("x").real(1.1)
        with builder.field("y").list():
            builder.integer(1)
    with builder.record():
        builder.field("x").real(2.2)
        with builder.field("y").list():
            builder.integer(1)
            builder.integer(2)
    with builder.record():
        builder.field("x").real(3.3)
        with builder.field("y").list():
            builder.integer(1)
            builder.integer(2)
            builder.integer(3)

with builder.list():
    pass

with builder.list():
    with builder.record():
        builder.field("x").real(4.4)
        with builder.field("y").list():
            builder.integer(3)
            builder.integer(2)
            
    with builder.record():
        builder.field("x").real(5.5)
        with builder.field("y").list():
            builder.integer(3)

array = builder.snapshot()
array
```

```{code-cell}
array.tolist()
```

gets all of the items in the first list separately from the items in the second or third lists, and both fields of each record (_x_ and _y_) are expressed near each other in the flow of the code and in the times when they're appended.

By contrast, the physical data are laid out in columns, with all _x_ values next to each other, regardless of which records or lists they're in, and all the _y_ values next to each other in another buffer.

```{code-cell}
array.layout
```

To build arrays using the layout constructors, you need to be able to write them in a form similar to the above, with the data already arranged as columns, in a tree representing the _type structure_ of items in the array, not separate trees for each array element.

+++

![](img/example-hierarchy.svg)

+++

The Awkward Array library has a closed set of node types. Building the array structure you want will require you to understand the node types that you use.

The node types (with validity rules for each) are documented under [ak.layout.Content](https://awkward-array.readthedocs.io/en/latest/ak.layout.Content.html), but this tutorial will walk through them explaining the situations in which you'd want to use each.

+++

Contents and Indexes
--------------------

[ak.layout.Content](https://awkward-array.readthedocs.io/en/latest/ak.layout.Content.html) is the _abstract_ superclass of all node types. All Content nodes that you would create are concrete subclasses of this class. The superclass is useful for checking `isinstance(some_object, ak.layout.Content)`, since there are some attributes that are only allowed to be Content nodes.

[ak.layout.Index](https://awkward-array.readthedocs.io/en/latest/ak.layout.Index.html) instances are buffers of integers that are used to give structure to an array. For instance, the `offsets` in the ListOffsetArrays, above, are Indexes, but the NumpyArray of _y_ list contents are not. [ak.layout.NumpyArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.NumpyArray.html) is a subclass of [ak.layout.Content](https://awkward-array.readthedocs.io/en/latest/ak.layout.Content.html), and [ak.layout.Index](https://awkward-array.readthedocs.io/en/latest/ak.layout.Index.html) is not. Indexes are more restricted than general NumPy arrays (must be one-dimensional, C-contiguous integers; dtypes are also prescribed) because they are frequently manipulated by Awkward Array operations, such as slicing.

There are five Index specializations, and each [ak.layout.Content](https://awkward-array.readthedocs.io/en/latest/ak.layout.Content.html) subclass has limitations on which ones it can use.

   * **Index8:** an Index of signed 8-bit integers
   * **IndexU8:** an Index of unsigned 8-bit integers
   * **Index32:** an Index of signed 32-bit integers
   * **IndexU32:** an Index of unsigned 32-bit integers
   * **Index64:** an Index of signed 64-bit integers

+++

Parameters
----------

Each layout node can have arbitrary metadata, called "parameters." Some parameters have built-in meanings, which are described below, and others can be given meanings by defining functions in [ak.behavior](https://awkward-array.readthedocs.io/en/latest/ak.behavior.html).

+++

Content > EmptyArray
--------------------

[ak.layout.EmptyArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.EmptyArray.html) is one of the two possible leaf types of a layout tree; the other is [ak.layout.NumpyArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.NumpyArray.html) (A third, corner-case "leaf type" is a [ak.layout.RecordArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.RecordArray.html) with zero fields).

EmptyArray is a trivial node type: it can only represent empty arrays with unknown type.

```{code-cell}
ak.layout.EmptyArray()
```

```{code-cell}
ak.Array(ak.layout.EmptyArray())
```

Since this is such a simple node type, let's use it to show examples of adding parameters.

```{code-cell}
ak.layout.EmptyArray(parameters={"name1": "value1", "name2": {"more": ["complex", "value"]}})
```

Content > NumpyArray
--------------------

[ak.layout.NumpyArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.NumpyArray.html) is one of the two possible leaf types of a layout tree; the other is [ak.layout.EmptyArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.EmptyArray.html). (A third, corner-case "leaf type" is a [ak.layout.RecordArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.RecordArray.html) with zero fields)

NumpyArray represents data the same way as a NumPy [np.ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html). That is, it can be multidimensional, but only rectilinear arrays.

```{code-cell}
ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
```

```{code-cell}
ak.layout.NumpyArray(np.array([[1, 2, 3], [4, 5, 6]], np.int16))
```

```{code-cell}
ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])[::2])
```

```{code-cell}
ak.layout.NumpyArray(np.array([[1, 2, 3], [4, 5, 6]], np.int16)[:, 1:])
```

In most array structures, the NumpyArrays only need to be 1-dimensional, since regular-length dimensions can be represented by [ak.layout.RegularArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.RegularArray.html) and variable-length dimensions can be represented by [ak.layout.ListArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.ListArray.html) or [ak.layout.ListOffsetArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.ListOffsetArray.html).

The [ak.from_numpy](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_numpy.html) function has a `regulararray` argument to choose between putting multiple dimensions into the NumpyArray node or nesting a 1-dimensional NumpyArray in RegularArray nodes.

```{code-cell}
ak.from_numpy(np.array([[1, 2, 3], [4, 5, 6]], np.int16), regulararray=False, highlevel=False)
```

```{code-cell}
ak.from_numpy(np.array([[1, 2, 3], [4, 5, 6]], np.int16), regulararray=True, highlevel=False)
```

All of these representations look the same in an [ak.Array](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Array.html) (high-level view).

```{code-cell}
ak.Array(ak.layout.NumpyArray(np.array([[1, 2, 3], [4, 5, 6]])))
```

```{code-cell}
ak.Array(ak.layout.RegularArray(ak.layout.NumpyArray(np.array([1, 2, 3, 4, 5, 6])), 3))
```

If you are _producing_ arrays, you can pick any representation that is convenient. If you are _consuming_ arrays, you need to be aware of the different representations.

+++

Content > RegularArray
----------------------

[ak.layout.RegularArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.RegularArray.html) represents regular-length lists (lists with all the same length). This was shown above as being equivalent to dimensions in a [ak.layout.NumpyArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.NumpyArray.html), but it can also contain irregular data.

```{code-cell}
layout = ak.layout.RegularArray(
    ak.from_iter([1, 2, 3, 4, 5, 6], highlevel=False),
    3,
)
layout
```

```{code-cell}
ak.Array(layout)
```

```{code-cell}
layout = ak.layout.RegularArray(
    ak.from_iter([[], [1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]], highlevel=False),
    3,
)
layout
```

```{code-cell}
ak.Array(layout)
```

The data type for a RegularArray is [ak.types.RegularType](https://awkward-array.readthedocs.io/en/latest/ak.types.RegularType.html), printed above as the "`3 *`" in the type above. (The "`2 *`" describes the length of the array itself, which is always "regular" in the sense that there's only one of them, equal to itself.)

The "`var *`" is the type of variable-length lists, nested inside of the RegularArray.

+++

RegularArray is the first array type that can have unreachable data: the length of its nested content might not evenly divide the RegularArray's regular `size`.

```{code-cell}
ak.Array(
    ak.layout.RegularArray(
        ak.layout.NumpyArray(np.array([1, 2, 3, 4, 5, 6, 7])),
        3,
    )
)
```

In the high-level array, we only see `[[1, 2, 3], [4, 5, 6]]` and not `7`. Since the 7 items in the nested NumpyArray can't be subdivided into lists of length 3. This `7` exists in the underlying physical data, but in the high-level view, it is as though it did not.

+++

Content > ListArray
-------------------

[ak.layout.ListArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.ListArray.html) and [ak.layout.ListOffsetArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.ListOffsetArray.html) are the two node types that describe variable-length lists ([ak.types.ListType](https://awkward-array.readthedocs.io/en/latest/ak.types.ListType.html), represented in type strings as "`var *`").

[ak.layout.ListArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.ListArray.html) is the most general. It takes two Indexes, `starts` and `stops`, which indicate where each nested list starts and stops.

```{code-cell}
layout = ak.layout.ListArray64(
    ak.layout.Index64(np.array([0, 3, 3])),
    ak.layout.Index64(np.array([3, 3, 5])),
    ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])),
)
layout
```

```{code-cell}
ak.Array(layout)
```

The nested content, `[1.1, 2.2, 3.3, 4.4, 5.5]` is divided into three lists, `[1.1, 2.2, 3.3]`, `[]`, `[4.4, 5.5]` by `starts=[0, 3, 3]` and `stops=[3, 3, 5]`. That is to say, the first list is drawn from indexes `0` through `3` of the content, the second is empty (from `3` to `3`), and the third is drawn from indexes `3` through `5`.

+++

Content > ListOffsetArray
-------------------------

[ak.layout.ListOffsetArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.ListOffsetArray.html) and [ak.layout.ListArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.ListArray.html) are the two node types that describe variable-length lists ([ak.types.ListType](https://awkward-array.readthedocs.io/en/latest/ak.types.ListType.html), represented in type strings as "`var *`").

+++

Nested lists
------------

As with all non-leaf Content nodes, arbitrarily deep nested lists can be built by nesting RegularArrays, ListArrays, and ListOffsetArrays. In each case, the `starts`, `stops`, or `offsets` only index the next level down in structure.

For example, here is an array of 5 lists, whose length is approximately 20 each.

```{code-cell}
layout = ak.layout.ListOffsetArray64(
    ak.layout.Index64(np.array([0, 18, 42, 59, 83, 100])),
    ak.layout.NumpyArray(np.arange(100))
)
array = ak.Array(layout)
array[0], array[1], array[2], array[3], array[4]
```

Making an array of 3 lists that contains these 5 lists requires us to make indexes that go up to 5, not 100.

```{code-cell}
array = ak.Array(
    ak.layout.ListOffsetArray64(
        ak.layout.Index64(np.array([0, 3, 3, 5])),
        layout,
    )
)
array[0], array[1], array[2]
```

Strings and bytestrings
-----------------------

As described above, any Content node can have any parameters, but some have special meanings. Most parameters are intended to associate runtime behaviors with data structures, the way that methods add computational abilities to a class. Unicode strings and raw bytestrings are an example of this.

Awkward Array has no "StringArray" type because such a thing would be stored, sliced, and operated upon in most circumatances as a list-type array (RegularArray, ListArray, ListOffsetArray) of bytes. A parameter named "`__array__`" adds behaviors, such as the string interpretation, to arrays.

   * `"__array__": "byte"` interprets an [ak.layout.NumpyArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.NumpyArray.html) as raw bytes in a bytestring,
   * `"__array__": "char"` interprets an [ak.layout.NumpyArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.NumpyArray.html) as characters in a Unicode-encoded string,
   * `"__array__": "bytestring"` interprets a list-type array ([ak.layout.RegularArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.RegularArray.html), [ak.layout.ListArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.ListArray.html), [ak.layout.ListOffsetArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.ListOffsetArray.html)) as a raw bytestring,
   * `"__array__": "string"` interprets a list-type array ([ak.layout.RegularArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.RegularArray.html), [ak.layout.ListArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.ListArray.html), [ak.layout.ListOffsetArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.ListOffsetArray.html)) as a Unicode-encoded string.

The [ak.layout.NumpyArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.NumpyArray.html) must be directly nested within the list-type array, which can be checked with [ak.is_valid](https://awkward-array.readthedocs.io/en/latest/_auto/ak.is_valid.html) and [ak.validity_error](https://awkward-array.readthedocs.io/en/latest/_auto/ak.validity_error.html).

+++

Here is an example of a raw bytestring:

```{code-cell}
ak.Array(
    ak.layout.ListOffsetArray64(
        ak.layout.Index64(np.array([0, 3, 8, 11, 15])),
        ak.layout.NumpyArray(
            np.array([104, 101, 121, 116, 104, 101, 114, 101, 121, 111, 117, 103, 117, 121, 115], np.uint8),
            parameters={"__array__": "byte"}
        ),
        parameters={"__array__": "bytestring"}
    )
)
```

And here is an example of a Unicode-encoded string (UTF-8):

```{code-cell}
ak.Array(
    ak.layout.ListOffsetArray64(
        ak.layout.Index64(np.array([0, 3, 12, 15, 19])),
        ak.layout.NumpyArray(
            np.array([104, 101, 121, 226, 128, 148, 226, 128, 148, 226, 128, 148, 121, 111, 117, 103, 117, 121, 115], np.uint8),
            parameters={"__array__": "char"}
        ),
        parameters={"__array__": "string"}
    )
)
```

As with any other lists, strings can be nested within lists. Only the [ak.layout.RegularArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.RegularArray.html)/[ak.layout.ListArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.ListArray.html)/[ak.layout.ListOffsetArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.ListOffsetArray.html) corresponding to the strings should have the `"__array__": "string"` or `"bytestring"` parameter.

```{code-cell}
ak.Array(
    ak.layout.ListOffsetArray64(
        ak.layout.Index64([0, 2, 4]),
        ak.layout.ListOffsetArray64(
            ak.layout.Index64(np.array([0, 3, 12, 15, 19])),
            ak.layout.NumpyArray(
                np.array([104, 101, 121, 226, 128, 148, 226, 128, 148, 226, 128, 148, 121, 111, 117, 103, 117, 121, 115], np.uint8),
                parameters={"__array__": "char"}
            ),
            parameters={"__array__": "string"}
        )
    )
)
```

Content > RecordArray
---------------------

[ak.layout.RecordArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.RecordArray.html) and [ak.layout.UnionArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.UnionArray.html) are the only two node types that have multiple `contents`, not just a single `content` (and the property is pluralized to reflect this fact). RecordArrays represent a "[product type](https://en.wikipedia.org/wiki/Product_type)," data containing records with fields _x_, _y_, and _z_ have _x_'s type AND _y_'s type AND _z_'s type, whereas UnionArrays represent a "[sum type](https://en.wikipedia.org/wiki/Tagged_union)," data that are _x_'s type OR _y_'s type OR _z_'s type.

RecordArrays have no [ak.layout.Index](https://awkward-array.readthedocs.io/en/latest/ak.layout.Index.html)-valued properties; they may be thought of as metadata-only groupings of Content nodes. Since the RecordArray node holds an _array_ for each field, it is a "[struct of arrays](https://en.wikipedia.org/wiki/AoS_and_SoA)," rather than an "array of structs."

+++

RecordArray fields are ordered and provided as an ordered list of `contents` and field names (the `recordlookup`).

```{code-cell}
layout = ak.layout.RecordArray(
    [
        ak.from_iter([1.1, 2.2, 3.3, 4.4, 5.5], highlevel=False),
        ak.from_iter([[1], [1, 2], [1, 2, 3], [3, 2], [3]], highlevel=False),
    ],
    [
        "x",
        "y",
    ],
)
layout
```

```{code-cell}
ak.Array(layout)
```

```{code-cell}
ak.to_list(layout)
```

RecordArray fields do not need to have names. If the `recordlookup` is `None`, the RecordArray is interpreted as an array of tuples. (The word "tuple," in statically typed environments, usually means a fixed-length type in which each element may be a different type.)

```{code-cell}
layout = ak.layout.RecordArray(
    [
        ak.from_iter([1.1, 2.2, 3.3, 4.4, 5.5], highlevel=False),
        ak.from_iter([[1], [1, 2], [1, 2, 3], [3, 2], [3]], highlevel=False),
    ],
    None,
)
layout
```

```{code-cell}
ak.Array(layout)
```

```{code-cell}
ak.to_list(layout)
```

Since the RecordArray node holds an array for each of its fields, it is possible for these arrays to have different lengths. In such a case, the length of the RecordArray can be given explicitly or it is taken to be the length of the shortest field-array.

```{code-cell}
content0 = ak.layout.NumpyArray(np.array([1, 2, 3, 4, 5, 6, 7, 8]))
content1 = ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
content2 = ak.from_iter([[1], [1, 2], [1, 2, 3], [3, 2, 1], [3, 2], [3]], highlevel=False)
print(f"{len(content0) = }, {len(content1) = }, {len(content2) = }")

layout = ak.layout.RecordArray([content0, content1, content2], ["x", "y", "z"])
print(f"{len(layout) = }")
```

```{code-cell}
layout = ak.layout.RecordArray([content0, content1, content2], ["x", "y", "z"], length=3)
print(f"{len(layout) = }")
```

RecordArrays are also allowed to have zero fields. This is an unusual case, but it is one that allows a RecordArray to be a leaf node (like [ak.layout.EmptyArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.EmptyArray.html) and [ak.layout.NumpyArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.NumpyArray.html)). If a RecordArray has no fields, a length _must_ be given.

```{code-cell}
ak.Array(ak.layout.RecordArray([], [], length=5))
```

```{code-cell}
ak.Array(ak.layout.RecordArray([], None, length=5))
```

Scalar Records
--------------

An [ak.layout.RecordArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.RecordArray.html) is an _array_ of records. Just as you can extract a scalar number from an array of numbers, you can extract a scalar record. Unlike numbers, records may still be sliced in some ways like Awkward Arrays:

```{code-cell}
array = ak.Array(
    ak.layout.RecordArray(
        [
            ak.from_iter([1.1, 2.2, 3.3, 4.4, 5.5], highlevel=False),
            ak.from_iter([[1], [1, 2], [1, 2, 3], [3, 2], [3]], highlevel=False),
        ],
        [
            "x",
            "y",
        ],
    )
)
record = array[2]
record
```

```{code-cell}
record["y", -1]
```

Therefore, we need an [ak.layout.Record](https://awkward-array.readthedocs.io/en/latest/ak.layout.Record.html) type, but this Record is not an array, so it is not a subclass of [ak.layout.Content](https://awkward-array.readthedocs.io/en/latest/ak.layout.Content.html).

Due to the columnar orientation of Awkward Array, a RecordArray does not contain Records, a Record contains a RecordArray.

```{code-cell}
record.layout
```

It can be built by passing a RecordArray as its first argument and the item of interest in its second argument.

```{code-cell}
layout = ak.layout.Record(
    ak.layout.RecordArray(
        [
            ak.from_iter([1.1, 2.2, 3.3, 4.4, 5.5], highlevel=False),
            ak.from_iter([[1], [1, 2], [1, 2, 3], [3, 2], [3]], highlevel=False),
        ],
        [
            "x",
            "y",
        ],
    ),
    2
)
record = ak.Record(layout)   # note the high-level ak.Record, rather than ak.Array
record
```

Naming record types
-------------------

The records discussed so far are generic. Naming a record not only makes it easier to read type strings, it's also how [ak.behavior](https://awkward-array.readthedocs.io/en/latest/ak.behavior.html) overloads functions and adds methods to records as though they were classes in object-oriented programming.

A name is given to an [ak.layout.RecordArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.RecordArray.html) node through its "`__record__`" parameter.

```{code-cell}
layout = ak.layout.RecordArray(
    [
        ak.from_iter([1.1, 2.2, 3.3, 4.4, 5.5], highlevel=False),
        ak.from_iter([[1], [1, 2], [1, 2, 3], [3, 2], [3]], highlevel=False),
    ],
    [
        "x",
        "y",
    ],
    parameters={"__record__": "Special"}
)
layout
```

```{code-cell}
ak.Array(layout)
```

```{code-cell}
ak.type(layout)
```

Behavioral overloads are presented in more depth in [ak.behavior](https://awkward-array.readthedocs.io/en/latest/ak.behavior.html), but here are three examples:

```{code-cell}
ak.behavior[np.sqrt, "Special"] = lambda special: np.sqrt(special.x)

np.sqrt(ak.Array(layout))
```

```{code-cell}
class SpecialRecord(ak.Record):
    def len_y(self):
        return len(self.y)

ak.behavior["Special"] = SpecialRecord

ak.Record(layout[2]).len_y()
```

```{code-cell}
class SpecialArray(ak.Array):
    def len_y(self):
        return ak.num(self.y)

ak.behavior["*", "Special"] = SpecialArray

ak.Array(layout).len_y()
```

Content > IndexedArray
----------------------

[ak.layout.IndexedArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.IndexedArray.html)

+++

Content > IndexedOptionArray
----------------------------

[ak.layout.IndexedOptionArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.IndexedOptionArray.html)

+++

Content > ByteMaskedArray
-------------------------

[ak.layout.ByteMaskedArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.ByteMaskedArray.html)

+++

Content > BitMaskedArray
------------------------

[ak.layout.BitMaskedArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.BitMaskedArray.html)

+++

Content > UnmaskedArray
-----------------------

[ak.layout.UnmaskedArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.UnmaskedArray.html)

+++

Content > UnionArray
--------------------

[ak.layout.UnionArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.UnionArray.html)

+++

Content > VirtualArray
----------------------

[ak.layout.VirtualArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.VirtualArray.html)

+++

PartitionedArray and IrregularlyPartitionedArray
------------------------------------------------

[ak.partition.PartitionedArray](https://awkward-array.readthedocs.io/en/latest/_auto/ak.partition.PartitionedArray.html)'s subclass, [ak.partition.IrregularlyPartitionedArray](https://awkward-array.readthedocs.io/en/latest/_auto/ak.partition.IrregularlyPartitionedArray.html), is not a [ak.layout.Content](https://awkward-array.readthedocs.io/en/latest/ak.layout.Content.html) subclass, deliberately preventing it from being used within a layout tree. It can, however, be the _root_ of a layout tree within an [ak.Array](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Array.html). This reflects our view that partitioning is only useful (and easiest to get right) for whole arrays, not internal nodes of a columnar structure.

+++

Relationship to ak.from_buffers
-------------------------------

The [generic buffers](how-to-convert-buffers) tutorial describes a function, [ak.from_buffers](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_buffers.html) that builds an array from one-dimensional buffers and an [ak.forms.Form](https://awkward-array.readthedocs.io/en/latest/ak.forms.Form.html). Forms describe the complete tree structure of an array without the array data or lengths, and the array data are in the buffers. The [ak.from_buffers](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_buffers.html) function was designed to operate on data produced by [ak.to_buffers](https://awkward-array.readthedocs.io/en/latest/_auto/ak.to_buffers.html), but you can also prepare its `form`, `length`, and `buffers` manually.

The [ak.from_buffers](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_buffers.html) builds arrays using the above constructors, but the interface allows these structures to be built as data, rather than function calls. (Forms have a JSON representation.) If you are always building the same type of array, directly calling the constructors is likely easier. If you're generating different data types programmatically, preparing data for [ak.from_buffers](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_buffers.html) may be easier than generating and evaluating Python code that call these constructors.

Every [ak.layout.Content](https://awkward-array.readthedocs.io/en/latest/ak.layout.Content.html) subclass has a corresponding [ak.forms.Form](https://awkward-array.readthedocs.io/en/latest/ak.forms.Form.html), and you can see a layout's Form through its `form` property.

```{code-cell}
array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
array.layout
```

```{code-cell}
# Abbreviated JSON representation
array.layout.form
```

```{code-cell}
# Full JSON representation
print(array.layout.form.tojson(pretty=True))
```

In this way, you can figure out how to generate Forms corresponding to the Content nodes you want [ak.from_buffers](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_buffers.html) to make.
