# Direct constructors (fastest)

If you’re willing to think about your data in a columnar way, directly constructing layouts and wrapping them in [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) interfaces is the fastest way to make them. (All other methods do this at some level.)

“Thinking about data in a columnar way” is the crucial difference between this method and [using ArrayBuilder](sphinx-llm:26cb6d67e3ae4372aacb2b87a0852a58). The builder method lets you think about a data structure the way you would think about Python objects, in which all fields of a given record or elements of a list are “together” and one record or list is “separate” from another record or list. For example,

```ipython3
import awkward as ak
import numpy as np
```

```ipython3
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

```ipython3
array.to_list()
```

gets all of the items in the first list separately from the items in the second or third lists, and both fields of each record (*x* and *y*) are expressed near each other in the flow of the code and in the times when they’re appended.

By contrast, the physical data are laid out in columns, with all *x* values next to each other, regardless of which records or lists they’re in, and all the *y* values next to each other in another buffer.

```ipython3
array.layout
```

To build arrays using the layout constructors, you need to be able to write them in a form similar to the above, with the data already arranged as columns, in a tree representing the *type structure* of items in the array, not separate trees for each array element.

![](image/example-hierarchy.svg)

The Awkward Array library has a closed set of node types. Building the array structure you want will require you to understand the node types that you use.

The node types (with validity rules for each) are documented under [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content), but this tutorial will walk through them explaining the situations in which you’d want to use each.

## Content classes

[`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) is the *abstract* superclass of all node types. All Content nodes that you would create are concrete subclasses of this class. The superclass is useful for checking `isinstance(some_object, ak.contents.Content)`, since there are some attributes that are only allowed to be Content nodes.

Sections in this document about a subclass of Content are named “`Content >: XYZ`” (using the “[is superclass of](https://stackoverflow.com/q/7759361/1623645)” operator).

## Parameters

Each layout node can have arbitrary metadata<sup>[1](#foot)</sup>, called “parameters.” Some parameters have built-in meanings, which are described below, and others can be given meanings by defining functions in [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior).

## Index classes

[`ak.index.Index`](sphinx-llm:c5dc164fb63f49c0946c0b5dfb338435#ak.index.Index) instances are buffers of integers that are used to give structure to an array. For instance, the `offsets` in the ListOffsetArrays, above, are Indexes, but the NumpyArray of *y* list contents are not. [`ak.contents.NumpyArray`](sphinx-llm:6f8cd60f60b04ce7a9499f2967197e79#ak.contents.NumpyArray) is a subclass of [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content), and [`ak.index.Index`](sphinx-llm:c5dc164fb63f49c0946c0b5dfb338435#ak.index.Index) is not. Indexes are more restricted than general NumPy arrays (must be one-dimensional, C-contiguous integers; dtypes are also prescribed) because they are frequently manipulated by Awkward Array operations, such as slicing.

There are five Index specializations, and each [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass has limitations on which ones it can use.

* **Index8:** an Index of signed 8-bit integers
* **IndexU8:** an Index of unsigned 8-bit integers
* **Index32:** an Index of signed 32-bit integers
* **IndexU32:** an Index of unsigned 32-bit integers
* **Index64:** an Index of signed 64-bit integers

## Content >: EmptyArray

[`ak.contents.EmptyArray`](sphinx-llm:23afb799007a46fc9153ea486777646c#ak.contents.EmptyArray) is one of the two possible leaf types of a layout tree; the other is [`ak.contents.NumpyArray`](sphinx-llm:6f8cd60f60b04ce7a9499f2967197e79#ak.contents.NumpyArray) (A third, corner-case “leaf type” is a [`ak.contents.RecordArray`](sphinx-llm:2475335b802d4bf79c938eaf015c0299#ak.contents.RecordArray) with zero fields).

EmptyArray is a trivial node type: it can only represent empty arrays with unknown type. It is an identity — when merging an array against an empty array, the empty array has no effect upon the result type. As such, this node cannot have user-defined parameters; `ak.contents.EmptyArray.parameters` is always empty.

```ipython3
ak.contents.EmptyArray()
```

```ipython3
ak.Array(ak.contents.EmptyArray())
```

## Content >: NumpyArray

[`ak.contents.NumpyArray`](sphinx-llm:6f8cd60f60b04ce7a9499f2967197e79#ak.contents.NumpyArray) is one of the two possible leaf types of a layout tree; the other is [`ak.contents.EmptyArray`](sphinx-llm:23afb799007a46fc9153ea486777646c#ak.contents.EmptyArray). (A third, corner-case “leaf type” is a [`ak.contents.RecordArray`](sphinx-llm:2475335b802d4bf79c938eaf015c0299#ak.contents.RecordArray) with zero fields)

NumpyArray represents data the same way as a NumPy [np.ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html). That is, it can be multidimensional, but only rectilinear arrays.

```ipython3
ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
```

```ipython3
ak.contents.NumpyArray(np.array([[1, 2, 3], [4, 5, 6]], np.int16))
```

```ipython3
ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])[::2])
```

```ipython3
ak.contents.NumpyArray(np.array([[1, 2, 3], [4, 5, 6]], np.int16)[:, 1:])
```

In most array structures, the NumpyArrays only need to be 1-dimensional, since regular-length dimensions can be represented by [`ak.contents.RegularArray`](sphinx-llm:c74b5b5ed0864b78bcc7b754cd89cd00#ak.contents.RegularArray) and variable-length dimensions can be represented by [`ak.contents.ListArray`](sphinx-llm:e2e74f5b6c644d858b90eaa78cd4dedc#ak.contents.ListArray) or [`ak.contents.ListOffsetArray`](sphinx-llm:00659f65177e4a1db8f94cec2f25984d#ak.contents.ListOffsetArray).

The [`ak.from_numpy`](sphinx-llm:efd7416013fa442595132125c93aa4ea#ak.from_numpy) function has a `regulararray` argument to choose between putting multiple dimensions into the NumpyArray node or nesting a 1-dimensional NumpyArray in RegularArray nodes.

```ipython3
ak.from_numpy(
    np.array([[1, 2, 3], [4, 5, 6]], np.int16), regulararray=False, highlevel=False
)
```

```ipython3
ak.from_numpy(
    np.array([[1, 2, 3], [4, 5, 6]], np.int16), regulararray=True, highlevel=False
)
```

All of these representations look the same in an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) (high-level view).

```ipython3
ak.Array(ak.contents.NumpyArray(np.array([[1, 2, 3], [4, 5, 6]])))
```

```ipython3
ak.Array(
    ak.contents.RegularArray(ak.contents.NumpyArray(np.array([1, 2, 3, 4, 5, 6])), 3)
)
```

If you are *producing* arrays, you can pick any representation that is convenient. If you are *consuming* arrays, you need to be aware of the different representations.

Since this is such a simple node type, let’s use it to show examples of adding parameters.

```ipython3
ak.Array(
    ak.contents.NumpyArray(
        np.array([[1, 2, 3], [4, 5, 6]]),
        parameters={"name1": "value1", "name2": {"more": ["complex", "value"]}}
    )
)
```

## Content >: RegularArray

[`ak.contents.RegularArray`](sphinx-llm:c74b5b5ed0864b78bcc7b754cd89cd00#ak.contents.RegularArray) represents regular-length lists (lists with all the same length). This was shown above as being equivalent to dimensions in a [`ak.contents.NumpyArray`](sphinx-llm:6f8cd60f60b04ce7a9499f2967197e79#ak.contents.NumpyArray), but it can also contain irregular data.

```ipython3
layout = ak.contents.RegularArray(
    ak.from_iter([1, 2, 3, 4, 5, 6], highlevel=False),
    3,
)
layout
```

```ipython3
ak.Array(layout)
```

```ipython3
layout = ak.contents.RegularArray(
    ak.from_iter(
        [[], [1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]], highlevel=False
    ),
    3,
)
layout
```

```ipython3
ak.Array(layout)
```

The data type for a RegularArray is [`ak.types.RegularType`](sphinx-llm:a729849acf504584b9a7d4ae728c6263#ak.types.RegularType), printed above as the “`3 *`” in the type above. (The “`2 *`” describes the length of the array itself, which is always “regular” in the sense that there’s only one of them, equal to itself.)

The “`var *`” is the type of variable-length lists, nested inside of the RegularArray.

RegularArray is the first array type that can have unreachable data: the length of its nested content might not evenly divide the RegularArray’s regular `size`.

```ipython3
ak.Array(
    ak.contents.RegularArray(
        ak.contents.NumpyArray(np.array([1, 2, 3, 4, 5, 6, 7])),
        3,
    )
)
```

In the high-level array, we only see `[[1, 2, 3], [4, 5, 6]]` and not `7`. Since the 7 items in the nested NumpyArray can’t be subdivided into lists of length 3. This `7` exists in the underlying physical data, but in the high-level view, it is as though it did not.

## Content >: ListArray

[`ak.contents.ListArray`](sphinx-llm:e2e74f5b6c644d858b90eaa78cd4dedc#ak.contents.ListArray) and [`ak.contents.ListOffsetArray`](sphinx-llm:00659f65177e4a1db8f94cec2f25984d#ak.contents.ListOffsetArray) are the two node types that describe variable-length lists ([`ak.types.ListType`](sphinx-llm:649f7f37622a4fbc999e70526300cf12#ak.types.ListType), represented in type strings as “`var *`”). [`ak.contents.ListArray`](sphinx-llm:e2e74f5b6c644d858b90eaa78cd4dedc#ak.contents.ListArray) is the most general. It takes two Indexes, `starts` and `stops`, which indicate where each nested list starts and stops.

```ipython3
layout = ak.contents.ListArray(
    ak.index.Index64(np.array([0, 3, 3])),
    ak.index.Index64(np.array([3, 3, 5])),
    ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])),
)
layout
```

```ipython3
ak.Array(layout)
```

The nested content, `[1.1, 2.2, 3.3, 4.4, 5.5]` is divided into three lists, `[1.1, 2.2, 3.3]`, `[]`, `[4.4, 5.5]` by `starts=[0, 3, 3]` and `stops=[3, 3, 5]`. That is to say, the first list is drawn from indexes `0` through `3` of the content, the second is empty (from `3` to `3`), and the third is drawn from indexes `3` through `5`.

## Content >: ListOffsetArray

[`ak.contents.ListOffsetArray`](sphinx-llm:00659f65177e4a1db8f94cec2f25984d#ak.contents.ListOffsetArray) and [`ak.contents.ListArray`](sphinx-llm:e2e74f5b6c644d858b90eaa78cd4dedc#ak.contents.ListArray) are the two node types that describe variable-length lists ([`ak.types.ListType`](sphinx-llm:649f7f37622a4fbc999e70526300cf12#ak.types.ListType), represented in type strings as “`var *`”). [`ak.contents.ListOffsetArray`](sphinx-llm:00659f65177e4a1db8f94cec2f25984d#ak.contents.ListOffsetArray) is an important special case, in which

```python
starts = offsets[:-1]
stops  = offsets[1:]
```

for a single array `offsets`. If we were only representing arrays and not doing computations on them, we would always use ListOffsetArrays, because they are the most compact. Knowing that a node is a ListOffsetArray can also simplify the implementation of some operations. In a sense, operations that produce ListArrays (previous section) can be thought of as delaying the part of the operation that would propagate down into the `content`, as an optimization.

```ipython3
layout = ak.contents.ListOffsetArray(
    ak.index.Index64(np.array([0, 3, 3, 5])),
    ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])),
)
layout
```

```ipython3
ak.Array(layout)
```

The length of the `offsets` array is one larger than the length of the array itself; an empty array has an `offsets` of length 1.

However, the `offsets` does not need to start at `0` or stop at `len(content)`.

```ipython3
ak.Array(
    ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([1, 3, 3, 4])),
        ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])),
    )
)
```

In the above, `offsets[0] == 1` means that the `1.1` in the `content` is unreachable, and `offsets[-1] == 4` means that the `5.5` is unreachable. Just as in ListArrays, unreachable data in ListOffsetArrays usually comes about because we don’t want computations to always propagate all the way down large trees.

## Nested lists

As with all non-leaf Content nodes, arbitrarily deep nested lists can be built by nesting RegularArrays, ListArrays, and ListOffsetArrays. In each case, the `starts`, `stops`, or `offsets` only index the next level down in structure.

For example, here is an array of 5 lists, whose length is approximately 20 each.

```ipython3
layout = ak.contents.ListOffsetArray(
    ak.index.Index64(np.array([0, 18, 42, 59, 83, 100])),
    ak.contents.NumpyArray(np.arange(100)),
)
array = ak.Array(layout)
array[0], array[1], array[2], array[3], array[4]
```

Making an array of 3 lists that contains these 5 lists requires us to make indexes that go up to 5, not 100.

```ipython3
array = ak.Array(
    ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 3, 3, 5])),
        layout,
    )
)
array[0], array[1], array[2]
```

## Strings and bytestrings

As described above, any Content node can have any parameters, but some have special meanings. Most parameters are intended to associate runtime behaviors with data structures, the way that methods add computational abilities to a class. Unicode strings and raw bytestrings are an example of this.

Awkward Array has no “StringArray” type because such a thing would be stored, sliced, and operated upon in most circumatances as a list-type array (RegularArray, ListArray, ListOffsetArray) of bytes. A parameter named “`__array__`” adds behaviors, such as the string interpretation, to arrays.

* `"__array__": "byte"` interprets an [`ak.contents.NumpyArray`](sphinx-llm:6f8cd60f60b04ce7a9499f2967197e79#ak.contents.NumpyArray) as raw bytes in a bytestring,
* `"__array__": "char"` interprets an [`ak.contents.NumpyArray`](sphinx-llm:6f8cd60f60b04ce7a9499f2967197e79#ak.contents.NumpyArray) as characters in a Unicode-encoded string,
* `"__array__": "bytestring"` interprets a list-type array ([`ak.contents.RegularArray`](sphinx-llm:c74b5b5ed0864b78bcc7b754cd89cd00#ak.contents.RegularArray), [`ak.contents.ListArray`](sphinx-llm:e2e74f5b6c644d858b90eaa78cd4dedc#ak.contents.ListArray), [`ak.contents.ListOffsetArray`](sphinx-llm:00659f65177e4a1db8f94cec2f25984d#ak.contents.ListOffsetArray)) as a raw bytestring,
* `"__array__": "string"` interprets a list-type array ([`ak.contents.RegularArray`](sphinx-llm:c74b5b5ed0864b78bcc7b754cd89cd00#ak.contents.RegularArray), [`ak.contents.ListArray`](sphinx-llm:e2e74f5b6c644d858b90eaa78cd4dedc#ak.contents.ListArray), [`ak.contents.ListOffsetArray`](sphinx-llm:00659f65177e4a1db8f94cec2f25984d#ak.contents.ListOffsetArray)) as a Unicode-encoded string.

The [`ak.contents.NumpyArray`](sphinx-llm:6f8cd60f60b04ce7a9499f2967197e79#ak.contents.NumpyArray) must be directly nested within the list-type array, which can be checked with [`ak.is_valid()`](sphinx-llm:cb72d589ba0e433d9cedf3f12cc0da1c#ak.is_valid) and [`ak.validity_error()`](sphinx-llm:ccbf3feebdb343f8ae3e23ce953f394a#ak.validity_error).

Here is an example of a raw bytestring:

```ipython3
ak.Array(
    ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 3, 8, 11, 15])),
        ak.contents.NumpyArray(
            np.array(
                [
                    104,
                    101,
                    121,
                    116,
                    104,
                    101,
                    114,
                    101,
                    121,
                    111,
                    117,
                    103,
                    117,
                    121,
                    115,
                ],
                np.uint8,
            ),
            parameters={"__array__": "byte"},
        ),
        parameters={"__array__": "bytestring"},
    )
)
```

And here is an example of a Unicode-encoded string (UTF-8):

```ipython3
ak.Array(
    ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 3, 12, 15, 19])),
        ak.contents.NumpyArray(
            np.array(
                [
                    104,
                    101,
                    121,
                    226,
                    128,
                    148,
                    226,
                    128,
                    148,
                    226,
                    128,
                    148,
                    121,
                    111,
                    117,
                    103,
                    117,
                    121,
                    115,
                ],
                np.uint8,
            ),
            parameters={"__array__": "char"},
        ),
        parameters={"__array__": "string"},
    )
)
```

As with any other lists, strings can be nested within lists. Only the [`ak.contents.RegularArray`](sphinx-llm:c74b5b5ed0864b78bcc7b754cd89cd00#ak.contents.RegularArray)/[`ak.contents.ListArray`](sphinx-llm:e2e74f5b6c644d858b90eaa78cd4dedc#ak.contents.ListArray)/[`ak.contents.ListOffsetArray`](sphinx-llm:00659f65177e4a1db8f94cec2f25984d#ak.contents.ListOffsetArray) corresponding to the strings should have the `"__array__": "string"` or `"bytestring"` parameter.

```ipython3
ak.Array(
    ak.contents.ListOffsetArray(
        ak.index.Index64([0, 2, 4]),
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 3, 12, 15, 19])),
            ak.contents.NumpyArray(
                np.array(
                    [
                        104,
                        101,
                        121,
                        226,
                        128,
                        148,
                        226,
                        128,
                        148,
                        226,
                        128,
                        148,
                        121,
                        111,
                        117,
                        103,
                        117,
                        121,
                        115,
                    ],
                    np.uint8,
                ),
                parameters={"__array__": "char"},
            ),
            parameters={"__array__": "string"},
        ),
    )
)
```

## Content >: RecordArray

[`ak.contents.RecordArray`](sphinx-llm:2475335b802d4bf79c938eaf015c0299#ak.contents.RecordArray) and [`ak.contents.UnionArray`](sphinx-llm:dd41ab713e6a4626a50f392463f45ad1#ak.contents.UnionArray) are the only two node types that have multiple `contents`, not just a single `content` (and the property is pluralized to reflect this fact). RecordArrays represent a “[product type](https://en.wikipedia.org/wiki/Product_type),” data containing records with fields *x*, *y*, and *z* have *x*’s type AND *y*’s type AND *z*’s type, whereas UnionArrays represent a “[sum type](https://en.wikipedia.org/wiki/Tagged_union),” data that are *x*’s type OR *y*’s type OR *z*’s type.

RecordArrays have no [`ak.index.Index`](sphinx-llm:c5dc164fb63f49c0946c0b5dfb338435#ak.index.Index)-valued properties; they may be thought of as metadata-only groupings of Content nodes. Since the RecordArray node holds an *array* for each field, it is a “[struct of arrays](https://en.wikipedia.org/wiki/AoS_and_SoA),” rather than an “array of structs.”

RecordArray fields are ordered and provided as an ordered list of `contents` and field names (the `recordlookup`).

```ipython3
layout = ak.contents.RecordArray(
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

```ipython3
ak.Array(layout)
```

```ipython3
ak.to_list(layout)
```

RecordArray fields do not need to have names. If the `recordlookup` is `None`, the RecordArray is interpreted as an array of tuples. (The word “tuple,” in statically typed environments, usually means a fixed-length type in which each element may be a different type.)

```ipython3
layout = ak.contents.RecordArray(
    [
        ak.from_iter([1.1, 2.2, 3.3, 4.4, 5.5], highlevel=False),
        ak.from_iter([[1], [1, 2], [1, 2, 3], [3, 2], [3]], highlevel=False),
    ],
    None,
)
layout
```

```ipython3
ak.Array(layout)
```

```ipython3
ak.to_list(layout)
```

Since the RecordArray node holds an array for each of its fields, it is possible for these arrays to have different lengths. In such a case, the length of the RecordArray can be given explicitly or it is taken to be the length of the shortest field-array.

```ipython3
content0 = ak.contents.NumpyArray(np.array([1, 2, 3, 4, 5, 6, 7, 8]))
content1 = ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
content2 = ak.from_iter(
    [[1], [1, 2], [1, 2, 3], [3, 2, 1], [3, 2], [3]], highlevel=False
)
print(f"{len(content0) = }, {len(content1) = }, {len(content2) = }")

layout = ak.contents.RecordArray([content0, content1, content2], ["x", "y", "z"])
print(f"{len(layout) = }")
```

```myst-ansi
len(content0) = 8, len(content1) = 5, len(content2) = 6
len(layout) = 5
```

```ipython3
layout = ak.contents.RecordArray(
    [content0, content1, content2], ["x", "y", "z"], length=3
)
print(f"{len(layout) = }")
```

```myst-ansi
len(layout) = 3
```

RecordArrays are also allowed to have zero fields. This is an unusual case, but it is one that allows a RecordArray to be a leaf node (like [`ak.contents.EmptyArray`](sphinx-llm:23afb799007a46fc9153ea486777646c#ak.contents.EmptyArray) and [`ak.contents.NumpyArray`](sphinx-llm:6f8cd60f60b04ce7a9499f2967197e79#ak.contents.NumpyArray)). If a RecordArray has no fields, a length *must* be given.

```ipython3
ak.Array(ak.contents.RecordArray([], [], length=5))
```

```ipython3
ak.Array(ak.contents.RecordArray([], None, length=5))
```

## Scalar Records

An [`ak.contents.RecordArray`](sphinx-llm:2475335b802d4bf79c938eaf015c0299#ak.contents.RecordArray) is an *array* of records. Just as you can extract a scalar number from an array of numbers, you can extract a scalar record. Unlike numbers, records may still be sliced in some ways like Awkward Arrays:

```ipython3
array = ak.Array(
    ak.contents.RecordArray(
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

```ipython3
record["y", -1]
```

Therefore, we need an [`ak.record.Record`](sphinx-llm:c769a66df3124cdc8442e55ad70f6980#ak.record.Record) type, but this Record is not an array, so it is not a subclass of [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content).

Due to the columnar orientation of Awkward Array, a RecordArray does not contain Records, a Record contains a RecordArray.

```ipython3
record.layout
```

It can be built by passing a RecordArray as its first argument and the item of interest in its second argument.

```ipython3
layout = ak.record.Record(
    ak.contents.RecordArray(
        [
            ak.from_iter([1.1, 2.2, 3.3, 4.4, 5.5], highlevel=False),
            ak.from_iter([[1], [1, 2], [1, 2, 3], [3, 2], [3]], highlevel=False),
        ],
        [
            "x",
            "y",
        ],
    ),
    2,
)
record = ak.Record(layout)  # note the high-level ak.Record, rather than ak.Array
record
```

## Naming record types

The records discussed so far are generic. Naming a record not only makes it easier to read type strings, it’s also how [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) overloads functions and adds methods to records as though they were classes in object-oriented programming.

A name is given to an [`ak.contents.RecordArray`](sphinx-llm:2475335b802d4bf79c938eaf015c0299#ak.contents.RecordArray) node through its “`__record__`” parameter.

```ipython3
layout = ak.contents.RecordArray(
    [
        ak.from_iter([1.1, 2.2, 3.3, 4.4, 5.5], highlevel=False),
        ak.from_iter([[1], [1, 2], [1, 2, 3], [3, 2], [3]], highlevel=False),
    ],
    [
        "x",
        "y",
    ],
    parameters={"__record__": "Special"},
)
layout
```

```ipython3
ak.Array(layout)
```

```ipython3
ak.type(layout)
```

Behavioral overloads are presented in more depth in [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior), but here are three examples:

```ipython3
ak.behavior[np.sqrt, "Special"] = lambda special: np.sqrt(special.x)

np.sqrt(ak.Array(layout))
```

```ipython3
class SpecialRecord(ak.Record):
    def len_y(self):
        return len(self.y)


ak.behavior["Special"] = SpecialRecord

ak.Record(layout[2]).len_y()
```

```ipython3
class SpecialArray(ak.Array):
    def len_y(self):
        return ak.num(self.y)


ak.behavior["*", "Special"] = SpecialArray

ak.Array(layout).len_y()
```

## Content >: IndexedArray

[`ak.contents.IndexedArray`](sphinx-llm:0ec798a3207a4746a193b3e535c37f41#ak.contents.IndexedArray) is the only Content node that has exactly the same type as its `content`. An IndexedArray rearranges the elements of its `content`, as though it were a lazily applied array-slice.

```ipython3
layout = ak.contents.IndexedArray(
    ak.index.Index64(np.array([2, 0, 0, 1, 2])),
    ak.contents.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3])),
)
layout
```

```ipython3
ak.Array(layout)
```

As such, IndexedArrays can be used to (lazily) remove items, permute items, and/or duplicate items.

IndexedArrays are used as an optimization when performing some operations on [`ak.contents.RecordArray`](sphinx-llm:2475335b802d4bf79c938eaf015c0299#ak.contents.RecordArray), to avoid propagating them down to every field. This is more relevant for RecordArrays than other nodes because RecordArrays can contain multiple `contents` (and UnionArrays, which also have multiple `contents`, have an `index` to merge an array-slice into).

For example, slicing the following `recordarray` by `[3, 2, 4, 4, 1, 0, 3]` does not affect any of the RecordArray’s fields.

```ipython3
recordarray = ak.contents.RecordArray(
    [
        ak.from_iter([1.1, 2.2, 3.3, 4.4, 5.5], highlevel=False),
        ak.from_iter([[1], [1, 2], [1, 2, 3], [3, 2], [3]], highlevel=False),
    ],
    None,
)
recordarray
```

```ipython3
recordarray[[3, 2, 4, 4, 1, 0, 3]]
```

## Categorical data

IndexedArrays, with the `"__array__": "categorical"` parameter, can represent categorical data (sometimes called dictionary-encoding).

```ipython3
layout = ak.contents.IndexedArray(
    ak.index.Index64(np.array([2, 2, 1, 4, 0, 5, 3, 3, 0, 1])),
    ak.from_iter(["zero", "one", "two", "three", "four", "five"], highlevel=False),
    parameters={"__array__": "categorical"},
)
ak.to_list(layout)
```

The above has only one copy of strings from `"zero"` to `"five"`, but they are effectively replicated 10 times in the array.

Any IndexedArray can perform this lazy replication, but labeling it as `"__array__": "categorical"` is a promise that the `content` contains only unique values.

Functions like [`ak.to_arrow()`](sphinx-llm:b7375fef0bca4ff1837b125c50ce5ad8#ak.to_arrow) and [`ak.to_parquet()`](sphinx-llm:34bdf9ca41984994beeef0eed40a786b#ak.to_parquet) will project (eagerly evaluate) an IndexedArray that is not labeled as `"__array__": "categorical"` and dictionary-encode an IndexedArray that is. This distinguishes between the use of IndexedArrays as invisible optimizations and intentional, user-visible ones.

## Content >: IndexedOptionArray

[`ak.contents.IndexedOptionArray`](sphinx-llm:3f1e6f3c5e154ca39fcd977247e4a221#ak.contents.IndexedOptionArray) is the most general of the four nodes that allow for missing data ([`ak.contents.IndexedOptionArray`](sphinx-llm:3f1e6f3c5e154ca39fcd977247e4a221#ak.contents.IndexedOptionArray), [`ak.contents.ByteMaskedArray`](sphinx-llm:98670076b6264905953fbfa4fd1faf70#ak.contents.ByteMaskedArray), [`ak.contents.BitMaskedArray`](sphinx-llm:a08c35b97c5a4c40b230a9d2289feebd#ak.contents.BitMaskedArray), and [`ak.contents.UnmaskedArray`](sphinx-llm:c65ee37a79b9404bbe625ef99e63bade#ak.contents.UnmaskedArray)). Missing data is also known as “[option type](https://en.wikipedia.org/wiki/Option_type)” (denoted by a question mark `?` or the word `option` in type strings).

[`ak.contents.IndexedOptionArray`](sphinx-llm:3f1e6f3c5e154ca39fcd977247e4a221#ak.contents.IndexedOptionArray) is an [`ak.contents.IndexedArray`](sphinx-llm:0ec798a3207a4746a193b3e535c37f41#ak.contents.IndexedArray) in which negative values in the `index` are interpreted as missing. Since it has an `index`, the `content` does not need to have “dummy values” at each index of a missing value. It is therefore more compact for record-type data with many missing values, but [`ak.contents.ByteMaskedArray`](sphinx-llm:98670076b6264905953fbfa4fd1faf70#ak.contents.ByteMaskedArray) and especially [`ak.contents.BitMaskedArray`](sphinx-llm:a08c35b97c5a4c40b230a9d2289feebd#ak.contents.BitMaskedArray) are more compact for numerical data or data without many missing values.

```ipython3
layout = ak.contents.IndexedOptionArray(
    ak.index.Index64(np.array([2, -1, 0, -1, -1, 1, 2])),
    ak.contents.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3])),
)
layout
```

```ipython3
ak.Array(layout)
```

Because of its flexibility, most operations that output potentially missing values use an IndexedOptionArray. [`ak.to_packed()`](sphinx-llm:e05e976068e942e983b7102df31bdca8#ak.to_packed) and conversions to/from Arrow or Parquet convert it back into a more compact type.

## Content >: ByteMaskedArray

[`ak.contents.ByteMaskedArray`](sphinx-llm:98670076b6264905953fbfa4fd1faf70#ak.contents.ByteMaskedArray) is the simplest of the four nodes that allow for missing data ([`ak.contents.IndexedOptionArray`](sphinx-llm:3f1e6f3c5e154ca39fcd977247e4a221#ak.contents.IndexedOptionArray), [`ak.contents.ByteMaskedArray`](sphinx-llm:98670076b6264905953fbfa4fd1faf70#ak.contents.ByteMaskedArray), [`ak.contents.BitMaskedArray`](sphinx-llm:a08c35b97c5a4c40b230a9d2289feebd#ak.contents.BitMaskedArray), and [`ak.contents.UnmaskedArray`](sphinx-llm:c65ee37a79b9404bbe625ef99e63bade#ak.contents.UnmaskedArray)). Missing data is also known as “[option type](https://en.wikipedia.org/wiki/Option_type)” (denoted by a question mark `?` or the word `option` in type strings).

[`ak.contents.ByteMaskedArray`](sphinx-llm:98670076b6264905953fbfa4fd1faf70#ak.contents.ByteMaskedArray) is most similar to NumPy’s [masked arrays](https://numpy.org/doc/stable/reference/maskedarray.html), except that Awkward ByteMaskedArrays can contain any data type and variable-length structures. The `valid_when` parameter lets you choose whether `True` means an element is valid (not masked/not `None`) or `False` means an element is valid.

```ipython3
layout = ak.contents.ByteMaskedArray(
    ak.index.Index8(np.array([False, False, True, True, False, True, False], np.int8)),
    ak.contents.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
    valid_when=False,
)
layout
```

```ipython3
ak.Array(layout)
```

## Content >: BitMaskedArray

[`ak.contents.BitMaskedArray`](sphinx-llm:a08c35b97c5a4c40b230a9d2289feebd#ak.contents.BitMaskedArray) is the most compact of the four nodes that allow for missing data ([`ak.contents.IndexedOptionArray`](sphinx-llm:3f1e6f3c5e154ca39fcd977247e4a221#ak.contents.IndexedOptionArray), [`ak.contents.ByteMaskedArray`](sphinx-llm:98670076b6264905953fbfa4fd1faf70#ak.contents.ByteMaskedArray), [`ak.contents.BitMaskedArray`](sphinx-llm:a08c35b97c5a4c40b230a9d2289feebd#ak.contents.BitMaskedArray), and [`ak.contents.UnmaskedArray`](sphinx-llm:c65ee37a79b9404bbe625ef99e63bade#ak.contents.UnmaskedArray)). Missing data is also known as “[option type](https://en.wikipedia.org/wiki/Option_type)” (denoted by a question mark `?` or the word `option` in type strings).

[`ak.contents.BitMaskedArray`](sphinx-llm:a08c35b97c5a4c40b230a9d2289feebd#ak.contents.BitMaskedArray) is just like [`ak.contents.ByteMaskedArray`](sphinx-llm:98670076b6264905953fbfa4fd1faf70#ak.contents.ByteMaskedArray) except that the booleans are bit-packed. It is motivated primarily by Apache Arrow, which uses bit-packed masks; supporting bit-packed masks in Awkward Array allows us to represent Arrow data directly. Most operations immediately convert BitMaskedArrays into ByteMaskedArrays.

Since bits always come in groups of at least 8, an explicit `length` must be supplied to the constructor. Also, `lsb_order=True` or `False` determines whether the bytes are interpreted [least-significant bit](https://en.wikipedia.org/wiki/Bit_numbering#LSB_0_bit_numbering) first or [most-significant bit](https://en.wikipedia.org/wiki/Bit_numbering#MSB_0_bit_numbering) first, respectively.

```ipython3
layout = ak.contents.BitMaskedArray(
    ak.index.IndexU8(
        np.packbits(np.array([False, False, True, True, False, True, False], np.uint8))
    ),
    ak.contents.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
    valid_when=False,
    length=7,
    lsb_order=True,
)
layout
```

```ipython3
ak.Array(layout)
```

Changing only the `lsb_order` changes the interpretation in important ways!

```ipython3
ak.Array(
    ak.contents.BitMaskedArray(
        layout.mask,
        layout.content,
        layout.valid_when,
        len(layout),
        lsb_order=False,
    )
)
```

## Content >: UnmaskedArray

[`ak.contents.UnmaskedArray`](sphinx-llm:c65ee37a79b9404bbe625ef99e63bade#ak.contents.UnmaskedArray) describes formally missing data, but in a case in which no data are actually missing. It is a corner case of the four nodes that allow for missing data ([`ak.contents.IndexedOptionArray`](sphinx-llm:3f1e6f3c5e154ca39fcd977247e4a221#ak.contents.IndexedOptionArray), [`ak.contents.ByteMaskedArray`](sphinx-llm:98670076b6264905953fbfa4fd1faf70#ak.contents.ByteMaskedArray), [`ak.contents.BitMaskedArray`](sphinx-llm:a08c35b97c5a4c40b230a9d2289feebd#ak.contents.BitMaskedArray), and [`ak.contents.UnmaskedArray`](sphinx-llm:c65ee37a79b9404bbe625ef99e63bade#ak.contents.UnmaskedArray)). Missing data is also known as “[option type](https://en.wikipedia.org/wiki/Option_type)” (denoted by a question mark `?` or the word `option` in type strings).

```ipython3
layout = ak.contents.UnmaskedArray(
    ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
)
layout
```

```ipython3
ak.Array(layout)
```

The only distinguishing feature of an UnmaskedArray is the question mark `?` or `option` in its type.

```ipython3
ak.type(layout)
```

## Content >: UnionArray

[`ak.contents.UnionArray`](sphinx-llm:dd41ab713e6a4626a50f392463f45ad1#ak.contents.UnionArray) and [`ak.contents.RecordArray`](sphinx-llm:2475335b802d4bf79c938eaf015c0299#ak.contents.RecordArray) are the only two node types that have multiple `contents`, not just a single `content` (and the property is pluralized to reflect this fact). RecordArrays represent a “[product type](https://en.wikipedia.org/wiki/Product_type),” data containing records with fields *x*, *y*, and *z* have *x*’s type AND *y*’s type AND *z*’s type, whereas UnionArrays represent a “[sum type](https://en.wikipedia.org/wiki/Tagged_union),” data that are *x*’s type OR *y*’s type OR *z*’s type.

In addition, [`ak.contents.UnionArray`](sphinx-llm:dd41ab713e6a4626a50f392463f45ad1#ak.contents.UnionArray) has two [`ak.index.Index`](sphinx-llm:c5dc164fb63f49c0946c0b5dfb338435#ak.index.Index)-typed attributes, `tags` and `index`; it is the most complex node type. The `tags` specify which `content` array to draw each array element from, and the `index` specifies which element from that `content`.

The UnionArray element at index `i` is therefore:

```python
contents[tags[i]][index[i]]
```

Although the ability to make arrays with mixed data type is very expressive, not all operations support union type (including iteration in Numba). If you intend to make union-type data for an application, be sure to verify that it will work by generating some test data using [`ak.from_iter()`](sphinx-llm:40c924e275044a0993aef7f20508d933#ak.from_iter).

Awkward Array’s UnionArray is equivalent to Apache Arrow’s [dense union](https://arrow.apache.org/docs/format/Columnar.html#dense-union). Awkward Array has no counterpart for Apache Arrow’s [sparse union](https://arrow.apache.org/docs/format/Columnar.html#sparse-union) (which has no `index`). [`ak.from_arrow()`](sphinx-llm:a01617c5c7474de09fdc0ddb61b2cced#ak.from_arrow) generates an `index` on demand when reading sparse union from Arrow.

```ipython3
layout = ak.contents.UnionArray(
    ak.index.Index8(np.array([0, 1, 2, 0, 0, 1, 1, 2, 2, 0], np.int8)),
    ak.index.Index64(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])),
    [
        ak.contents.NumpyArray(
            np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        ),
        ak.from_iter(
            [
                [],
                [1],
                [1, 2],
                [1, 2, 3],
                [1, 2, 3, 4],
                [1, 2, 3, 4, 5],
                [6],
                [6, 7],
                [6, 7, 8],
                [6, 7, 8, 9],
            ],
            highlevel=False,
        ),
        ak.from_iter(
            [
                "zero",
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
            ],
            highlevel=False,
        ),
    ],
)
layout
```

```ipython3
ak.to_list(layout)
```

The `index` can be used to prevent the need to set up “dummy values” for all contents other than the one specified by a given tag. The above example could thus be more compact with the following (no unreachable data):

```ipython3
layout = ak.contents.UnionArray(
    ak.index.Index8(np.array([0, 1, 2, 0, 0, 1, 1, 2, 2, 0], np.int8)),
    ak.index.Index64(np.array([0, 0, 0, 1, 2, 1, 2, 1, 2, 3])),
    [
        ak.contents.NumpyArray(np.array([0.0, 3.3, 4.4, 9.9])),
        ak.from_iter([[1], [1, 2, 3, 4, 5], [6]], highlevel=False),
        ak.from_iter(["two", "seven", "eight"], highlevel=False),
    ],
)
layout
```

```ipython3
ak.to_list(layout)
```

[`ak.from_iter()`](sphinx-llm:40c924e275044a0993aef7f20508d933#ak.from_iter) is by far the easiest way to create UnionArrays for small tests.

## Relationship to ak.from_buffers

The [generic buffers](sphinx-llm:59e10d0d69d145e0a99a6ae988238865) tutorial describes a function, [`ak.from_buffers()`](sphinx-llm:fba1faa82f7649e28003bf8ebdda86f8#ak.from_buffers) that builds an array from one-dimensional buffers and an [`ak.forms.Form`](sphinx-llm:6295086f75dd4713a4cbcfec3594d1a1#ak.forms.Form). Forms describe the complete tree structure of an array without the array data or lengths, and the array data are in the buffers. The [`ak.from_buffers()`](sphinx-llm:fba1faa82f7649e28003bf8ebdda86f8#ak.from_buffers) function was designed to operate on data produced by [`ak.to_buffers()`](sphinx-llm:0f37278395f642a289396d6393d4d837#ak.to_buffers), but you can also prepare its `form`, `length`, and `buffers` manually.

The [`ak.from_buffers()`](sphinx-llm:fba1faa82f7649e28003bf8ebdda86f8#ak.from_buffers) builds arrays using the above constructors, but the interface allows these structures to be built as data, rather than function calls. (Forms have a JSON representation.) If you are always building the same type of array, directly calling the constructors is likely easier. If you’re generating different data types programmatically, preparing data for [`ak.from_buffers()`](sphinx-llm:fba1faa82f7649e28003bf8ebdda86f8#ak.from_buffers) may be easier than generating and evaluating Python code that call these constructors.

Every [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass has a corresponding [`ak.forms.Form`](sphinx-llm:6295086f75dd4713a4cbcfec3594d1a1#ak.forms.Form), and you can see a layout’s Form through its `form` property.

```ipython3
array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
array.layout
```

```ipython3
# Abbreviated JSON representation
array.layout.form
```

```ipython3
# Full JSON representation
print(array.layout.form.to_json())
```

```myst-ansi
{"class": "ListOffsetArray", "offsets": "i64", "content": {"class": "NumpyArray", "primitive": "float64", "inner_shape": [], "parameters": {}, "form_key": null}, "parameters": {}, "form_key": null}
```

In this way, you can figure out how to generate Forms corresponding to the Content nodes you want [`ak.from_buffers()`](sphinx-llm:fba1faa82f7649e28003bf8ebdda86f8#ak.from_buffers) to make.

---
* <a id='foot'>**[1]**</a> Except for `ak.contents.EmptyArray`, which is an identity type.
