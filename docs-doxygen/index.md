[![](https://raw.githubusercontent.com/scikit-hep/awkward-1.0/main/docs-img/logo/logo-300px.png)](https://github.com/scikit-hep/awkward-1.0)

Awkward Array is a library for **nested, variable-sized data**, including arbitrary-length lists, records, mixed types, and missing data, using **NumPy-like idioms**.

Arrays are **dynamically typed**, but operations on them are **compiled and fast**. Their behavior coincides with NumPy when array dimensions are regular and generalizes when they’re not.

## Documentation

   * [Main page (quickstart)](https://awkward-array.org>)
   * C++ API reference: **this site**
   * [Python API reference](../index.html)
   * [GitHub repository](https://github.com/scikit-hep/awkward-1.0)

## Navigation

The Awkward Array project is divided into 3 layers with 5 main components.

<img src="awkward-1-0-layers.svg" style="max-width: 500px">

The high-level interface and Numba implementation are described in the [Python API reference](../index.html), though the Numba implementation is considered an internal detail (in the `_connect` submodule, which starts with an underscore and is therefore private in Python).

This reference describes the

   * **C++ classes:** in [namespace awkward](namespaceawkward.html) (often abbreviated as "ak"), which are compiled into **libawkward.so** (or dylib or lib). This library is fully usable from C++, without Python (see [dependent-project](https://github.com/scikit-hep/awkward-1.0/tree/main/dependent-project) for an example).
   * **pybind11 interface:** no namespace, but contained entirely within the [python directory](dir_91f33a3f1dd6262845ebd1570075970c.html), which are compiled into **awkward._ext** for use in Python.
   * **CPU kernels:** no namespace, but contained entirely within the [kernels directory](dir_6225843069e7cc68401bbec110a1667f.html), which are compiled into **libawkward-cpu-kernels.so** (or dylib or lib). This library is fully usable from any language that can call functions through [FFI](https://en.wikipedia.org/wiki/Foreign_function_interface).
   * **GPU kernels:** FIXME! (not implemented yet)

### Array layout nodes

Awkward Arrays are a tree of layout nodes nested within each other, some of which carry one-dimensional arrays that get reinterpreted as part of the structure. These descend from the abstract base class [ak::Content](classawkward_1_1Content.html) and are reflected through pybind11 as [ak.layout.Content](../ak.layout.Content.html) and subclasses in Python.

   * [ak::Content](classawkward_1_1Content.html): the abstract base class.
   * [ak::EmptyArray](classawkward_1_1EmptyArray.html): an array of unknown type with no elements (usually produced by [ak::ArrayBuilder](classawkward_1_1ArrayBuilder.html), which can't determine type at a given level without samples).
   * [ak::RawArrayOf<T>](classawkward_1_1RawArrayOf.html): a one-dimensional array of any fixed-bytesize C++ type `T`. [RawArray.h](RawArray_8h.html) is a header-only implementation and [ak::RawArrayOf<T>](classawkward_1_1RawArrayOf.html) are not used anywhere else in the Awkward Array project, Python in particular. This node type only exists for pure C++ dependent projects.
   * [ak::NumpyArray](classawkward_1_1NumpyArray.html): any NumPy array (e.g. multidimensional shape, arbitrary dtype), though usually only one-dimensional arrays of numbers. Contrary to its name, [ak::NumpyArray](classawkward_1_1NumpyArray.html) can be used in pure C++, without dependence on pybind11 or Python. The NumPy array is described in terms of `std::vectors` and a `std::string`.
   * [ak::RegularArray](classawkward_1_1RegularArray.html): splits its nested content into equal-length lists.
   * [ak::ListArrayOf<T>](classawkward_1_1ListArrayOf.html): splits its nested content into variable-length lists with full generality (may use its content non-contiguously, overlapping, or out-of-order).
   * [ak::ListOffsetArrayOf<T>](classawkward_1_1ListOffsetArrayOf.html): splits its nested content into variable-length lists, assuming contiguous, non-overlapping, in-order content.
   * [ak::RecordArray](classawkward_1_1RecordArray.html): represents a logical array of records with a "struct of arrays" layout in memory.
   * [ak::Record](classawkward_1_1Record.html): represents a single record (a subclass of [ak::Content](classawkward_1_1Content.html) in C++, but not Python).
   * [ak::IndexedArrayOf<T, ISOPTION>](classawkward_1_1IndexedArrayOf.html): rearranges and/or duplicates its content by lazily applying an integer index.
   * [ak::ByteMaskedArray](classawkward_1_1ByteMaskedArray.html): represents its content with missing values with an 8-bit boolean mask.
   * [ak::BitMaskedArray](classawkward_1_1BitMaskedArray.html): represents its content with missing values with a 1-bit boolean mask.
   * [ak::UnmaskedArray](classawkward_1_1UnmaskedArray.html): specifies that its content can contain missing values in principle, but no mask is supplied because all elements are non-missing.
   * [ak::UnionArrayOf<T, I>](classawkward_1_1UnionArrayOf.html): interleaves a set of arrays as a tagged union, can represent heterogeneous data.
   * [ak::VirtualArray](classawkward_1_1VirtualArray.html): generates an array on demand from an [ak::ArrayGenerator](classawkward_1_1ArrayGenerator.html) or a [ak::SliceGenerator](classawkward_1_1SliceGenerator.html) and optionally caches the generated array in an [ak::ArrayCache](classawkward_1_1ArrayCache.html).
   * [ak::None](classawkward_1_1None.html): represents a missing value that will be converted to `None` in Python (a subclass of [ak::Content](classawkward_1_1Content.html) in C++).

The [ak::Record](classawkward_1_1Record.html), [ak::None](classawkward_1_1None.html), and [ak::NumpyArray](classawkward_1_1NumpyArray.html) with empty [shape](classawkward_1_1NumpyArray.html#ab4eec3bfd0e50bc035c26e62974d209d) are technically [ak::Content](classawkward_1_1Content.html) in C++ even though they represent scalar data, rather than arrays. (This can be checked with the [isscalar](classawkward_1_1Content.html#a878ae38b66c14067b231469863d6d1a1) method.) This is because they are possible return values of methods that would ordinarily return [ak::Contents](classawkward_1_1Content.html), so they are subclasses to simplify the type hierarchy. However, in the [Python layer](dir_91f33a3f1dd6262845ebd1570075970c.html), they are converted directly into Python scalars, such as [ak.layout.Record](../ak.layout.Record.html) (which isn't an [ak.layout.Content](../ak.layout.Content.html) subclass), Python's `None`, or a Python number/bool.

Most layout nodes contain another content node ([ak::RecordArray](classawkward_1_1RecordArray.html) and [ak::UnionArrayOf<T, I>](classawkward_1_1UnionArrayOf.html) can contain more than one), thus forming a tree. Only [ak::EmptyArray](classawkward_1_1EmptyArray.html), [ak::RawArrayOf<T>](classawkward_1_1RawArrayOf.html), and [ak::NumpyArray](classawkward_1_1NumpyArray.html) cannot contain a content, and hence these are leaves of the tree.

Note that [ak::PartitionedArray](classawkward_1_1PartitionedArray.html) and its concrete class, [ak::IrregularlyPartitionedArray](classawkward_1_1IrregularlyPartitionedArray.html), are not [ak::Content](classawkward_1_1Content.html) because they cannot be nested within a tree. Partitioning is only allowed at the root of the tree.

**Iterator for layout nodes:** [ak::Iterator](classawkward_1_1Iterator.html).

**Index for layout nodes:** integer and boolean arrays that define the shape of the data structure, such as boolean masks in [ak::ByteMaskedArray](classawkward_1_1ByteMaskedArray.html), are not [ak::NumpyArray](classawkward_1_1NumpyArray.html) but a more constrained type called [ak::IndexOf<T>](classawkward_1_1IndexOf.html).

**Identities for layout nodes:** [ak::IdentitiesOf<T>](classawkward_1_1IdentitiesOf.html) are an optional surrogate key for certain join operations. (Not yet used.)

### High-level data types

This is the type of data in a high-level [ak.Array](../_auto/ak.Array.html) or [ak.Record](../_auto/ak.Record.html) as reported by [ak.type](../_auto/ak.type.html). It represents as much information as a data analyst needs to know (e.g. the distinction between variable and fixed-length lists, but not the distinction between [ak::ListArrayOf<T>](classawkward_1_1ListArrayOf.html) and [ak::ListOffsetArrayOf<T>](classawkward_1_1ListOffsetArrayOf.html)).

   * [ak::Type](classawkward_1_1Type.html): the abstract base class.
   * [ak::ArrayType](classawkward_1_1ArrayType.html): type of a non-composable, high-level [ak.Array](../_auto/ak.Array.html), which includes the length of the array.
   * [ak::UnknownType](classawkward_1_1UnknownType.html): a type that is not known because it is represented by an [ak::EmptyArray](classawkward_1_1EmptyArray.html).
   * [ak::PrimitiveType](classawkward_1_1PrimitiveType.html): a numeric or boolean type.
   * [ak::RegularType](classawkward_1_1RegularType.html): lists of a fixed length; this ``size`` is part of the type description.
   * [ak::ListType](classawkward_1_1ListType.html): lists of unspecified or variable length.
   * [ak::RecordType](classawkward_1_1RecordType.html): records with named fields or tuples with a fixed number of unnamed slots. The fields/slots and their types are part of the type description.
   * [ak::OptionType](classawkward_1_1OptionType.html): data that may be missing.
   * [ak::UnionType](classawkward_1_1UnionType.html): heterogeneous data selected from a short list of possibilities.

All concrete [ak::Type](classawkward_1_1Type.html) subclasses are composable except [ak::ArrayType](classawkward_1_1ArrayType.html).

### Low-level array forms

This is the type of a [ak::Content](classawkward_1_1Content.html) array expressed with low-level granularity (e.g. including the distinction between [ak::ListArrayOf<T>](classawkward_1_1ListArrayOf.html) and [ak::ListOffsetArrayOf<T>](classawkward_1_1ListOffsetArrayOf.html)). There is a one-to-one relationship between [ak::Content](classawkward_1_1Content.html) subclasses and [ak::Form](classawkward_1_1Form.html) subclasses, and each [ak::Form](classawkward_1_1Form.html) maps to only one [ak::Type](classawkward_1_1Type.html).

   * [ak::Form](classawkward_1_1Form.html): the abstract base class.
   * [ak::EmptyForm](classawkward_1_1EmptyForm.html) for [ak::EmptyArray](classawkward_1_1EmptyArray.html).
   * [ak::RawForm](classawkward_1_1RawForm.html) for [ak::RawArrayOf<T>](classawkward_1_1RawArrayOf.html).
   * [ak::NumpyForm](classawkward_1_1NumpyForm.html) for [ak::NumpyArray](classawkward_1_1NumpyArray.html).
   * [ak::RegularForm](classawkward_1_1RegularForm.html) for [ak::RegularArray](classawkward_1_1RegularArray.html).
   * [ak::ListForm](classawkward_1_1ListForm.html) for [ak::ListArrayOf<T>](classawkward_1_1ListArrayOf.html).
   * [ak::ListOffsetForm](classawkward_1_1ListOffsetForm.html) for [ak::ListOffsetArrayOf<T>](classawkward_1_1ListOffsetArrayOf.html).
   * [ak::RecordForm](classawkward_1_1RecordForm.html) for [ak::RecordArray](classawkward_1_1RecordArray.html).
   * [ak::IndexedForm](classawkward_1_1IndexedForm.html) for [ak::IndexedArrayOf<T, ISOPTION>](classawkward_1_1IndexedArrayOf.html) with `ISOPTION = false`.
   * [ak::IndexedOptionForm](classawkward_1_1IndexedOptionForm.html) for [ak::IndexedArrayOf<T, ISOPTION>](classawkward_1_1IndexedArrayOf.html) with `ISOPTION = true`.
   * [ak::ByteMaskedForm](classawkward_1_1ByteMaskedForm.html) for [ak::ByteMaskedArray](classawkward_1_1ByteMaskedArray.html).
   * [ak::BitMaskedForm](classawkward_1_1BitMaskedForm.html) for [ak::BitMaskedArray](classawkward_1_1BitMaskedArray.html).
   * [ak::UnmaskedForm](classawkward_1_1UnmaskedForm.html) for [ak::UnmaskedArray](classawkward_1_1UnmaskedArray.html).
   * [ak::UnionForm](classawkward_1_1UnionForm.html) for [ak::UnionArrayOf<T, I>](classawkward_1_1UnionArrayOf.html).
   * [ak::VirtualForm](classawkward_1_1VirtualForm.html) for [ak::VirtualArray](classawkward_1_1VirtualArray.html).

### ArrayBuilder structure

The [ak.ArrayBuilder](../_auto/ak.ArrayBuilder.html) is an append-only array for generating data backed by [ak.layout.ArrayBuilder](../_auto/ak.layout.ArrayBuilder.html) (layout-level ArrayBuilder) and [ak::ArrayBuilder](classawkward_1_1ArrayBuilder.html) (C++ implementation).

[ak::ArrayBuilder](classawkward_1_1ArrayBuilder.html) is the front-end for a tree of [ak::Builder](classawkward_1_1Builder.html) instances. The structure of this tree indicates the current state of knowledge about the type of the data it's being filled with, and this tree can grow from any node. Types always grow in the direction of more generality, so the tree only gets bigger.

Here is an example that illustrates how knowledge about the type grows.

```python
b = ak.ArrayBuilder()

# fill commands   # as JSON   # current array type
##########################################################################################
b.begin_list()    # [         # 0 * var * unknown     (initially, the type is unknown)
b.integer(1)      #   1,      # 0 * var * int64
b.integer(2)      #   2,      # 0 * var * int64
b.real(3)         #   3.0     # 0 * var * float64     (all the integers have become floats)
b.end_list()      # ],        # 1 * var * float64
b.begin_list()    # [         # 1 * var * float64
b.end_list()      # ],        # 2 * var * float64
b.begin_list()    # [         # 2 * var * float64
b.integer(4)      #   4,      # 2 * var * float64
b.null()          #   null,   # 2 * var * ?float64    (now the floats are nullable)
b.integer(5)      #   5       # 2 * var * ?float64
b.end_list()      # ],        # 3 * var * ?float64
b.begin_list()    # [         # 3 * var * ?float64
b.begin_record()  #   {       # 3 * var * ?union[float64, {}]
b.field("x")      #     "x":  # 3 * var * ?union[float64, {"x": unknown}]
b.integer(1)      #      1,   # 3 * var * ?union[float64, {"x": int64}]
b.field("y")      #      "y": # 3 * var * ?union[float64, {"x": int64, "y": unknown}]
b.begin_list()    #      [    # 3 * var * ?union[float64, {"x": int64, "y": var * unknown}]
b.integer(2)      #        2, # 3 * var * ?union[float64, {"x": int64, "y": var * int64}]
b.integer(3)      #        3  # 3 * var * ?union[float64, {"x": int64, "y": var * int64}]
b.end_list()      #      ]    # 3 * var * ?union[float64, {"x": int64, "y": var * int64}]
b.end_record()    #   }       # 3 * var * ?union[float64, {"x": int64, "y": var * int64}]
b.end_list()      # ]         # 4 * var * ?union[float64, {"x": int64, "y": var * int64}]

ak.to_list(b.snapshot())
# [[1.0, 2.0, 3.0], [], [4.0, None, 5.0], [{'x': 1, 'y': [2, 3]}]]
```

The [ak::Builder](classawkward_1_1Builder.html) instances contain arrays of accumulated data, and thus store both data (in these arrays) and type (in their tree structure). The hierarchy is not exactly the same as [ak::Content](classawkward_1_1Content.html)/[ak::Form](classawkward_1_1Form.html) (which are identical to each other) or [ak::Type](classawkward_1_1Type.html), since it reflects the kinds of data to be encountered in the input data: mostly JSON-like, but with a distinction between records with named fields and tuples with unnamed slots.

   * [ak::Builder](classawkward_1_1Builder.html): the abstract base class.
   * [ak::UnknownBuilder](classawkward_1_1UnknownBuilder.html): the initial builder; a builder for unknown type; generates an [ak::EmptyArray](classawkward_1_1EmptyArray.html) (which is why we have that class).
   * [ak::BoolBuilder](classawkward_1_1BoolBuilder.html): boolean type; generates a [ak::NumpyArray](classawkward_1_1NumpyArray.html).
   * [ak::Int64Builder](classawkward_1_1Int64Builder.html): 64-bit integer type; generates a [ak::NumpyArray](classawkward_1_1NumpyArray.html). Appending integer data to boolean data generates a union; it does not promote the booleans to integers.
   * [ak::Float64Builder](classawkward_1_1Float64Builder.html): 64-bit floating point type; generates a [ak::NumpyArray](classawkward_1_1NumpyArray.html). Appending floating-point data to integer data does not generate a union; it promotes integers to floating-point.
   * [ak::StringBuilder](classawkward_1_1StringBuilder.html): UTF-8 encoded string or raw bytestring type; generates a [ak::ListOffsetArrayOf<T>](classawkward_1_1ListOffsetArrayOf.html) with parameter `"__array__"` equal to `"string"` or `"bytestring"`.
   * [ak::ListBuilder](classawkward_1_1ListBuilder.html): list type; generates a [ak::ListOffsetArrayOf<T>](classawkward_1_1ListOffsetArrayOf.html).
   * [ak::OptionBuilder](classawkward_1_1OptionBuilder.html): option type; generats an [ak::IndexedArrayOf<T, ISOPTION>](classawkward_1_1IndexedArrayOf.html) with `ISOPTION = true`.
   * [ak::RecordBuilder](classawkward_1_1RecordBuilder.html): record type with field names; generates a [ak::RecordArray](classawkward_1_1RecordArray.html) with a non-null [ak::util::RecordLookup](namespaceawkward_1_1util.html#a86f80ddab98f968a2845db832f0c8210).
   * [ak::TupleBuilder](classawkward_1_1TupleBuilder.html): tuple type without field names; generates a [ak::RecordArray](classawkward_1_1RecordArray.html) with a null [ak::util::RecordLookup](namespaceawkward_1_1util.html#a86f80ddab98f968a2845db832f0c8210).
   * [ak::UnionBuilder](classawkward_1_1UnionBuilder.html): union type; generates a [ak::UnionArrayOf<T, I>](classawkward_1_1UnionArrayOf.html).
   * [ak::IndexedBuilder](classawkward_1_1IndexedBuilder.html): indexed [ak::Content](classawkward_1_1Content.html); inserts an existing array node into the new array under construction, referencing its elements with an [ak::IndexedArrayOf<T, ISOPTION>](classawkward_1_1IndexedArrayOf.html). This way, complex structures can be included by reference, rather than by copying.

**Options for building an array:** [ak::ArrayBuilderOptions](classawkward_1_1ArrayBuilderOptions.html) are passed to every [ak::Builder](classawkward_1_1Builder.html) in the tree.

**Buffers for building an array:** [ak::GrowableBuffer<T>](classawkward_1_1GrowableBuffer.html) is a one-dimensional array with append-only semantics, used both for buffers that will become [ak::IndexOf<T>](classawkward_1_1IndexOf.html) and buffers that will become [ak::NumpyArray](classawkward_1_1NumpyArray.html). It works like `std::vector` in that it replaces its underlying storage at logarithmically frequent intervals, but unlike a `std::vector` in that the underlying storage is a `std::shared_ptr<void*>`.

After an [ak::ArrayBuilder](classawkward_1_1ArrayBuilder.html) is turned into a [ak::Content](classawkward_1_1Content.html) with [snapshot](classawkward_1_1ArrayBuilder.html#ac064fd827abb99f81772e59706f1a6a8), the read-only data and its append-only source share buffers (with `std::shared_ptr<void*>`). This makes the [snapshot](classawkward_1_1ArrayBuilder.html#ac064fd827abb99f81772e59706f1a6a8) operation fast and capable of being called frequently, and the `std::shared_ptr` manages the lifetime of buffers that stay in scope because a [ak::Content](classawkward_1_1Content.html) is using it, even if a [ak::GrowableBuffer<T>](classawkward_1_1GrowableBuffer.html) is not (because it reallocated its internal buffer).

Array building is not as efficient as computing with pre-built arrays because the type-discovery makes each access a tree-descent. Array building is also an exception to the rule that C++ implementations do not touch array data (do not dereference pointers in [ak::RawArrayOf<T>](classawkward_1_1RawArrayOf.html), [ak::NumpyArray](classawkward_1_1NumpyArray.html), [ak::IndexOf<T>](classawkward_1_1IndexOf.html), and [ak::IdentitiesOf<T>](classawkward_1_1IdentitiesOf.html)). The [ak::Builder](classawkward_1_1Builder.html) instances append to their [ak::GrowableBuffer<T>](classawkward_1_1GrowableBuffer.html), which are assumed to exist in main memory, not a GPU.

### Reducers

Reducer operations like [ak.sum](../_auto/ak.sum.html) and [ak.max](../_auto/ak.max.html) are implemented as a group for code reuse. The most complex part is how elements from variable-length lists are grouped when `axis != -1` (see [ak::Content::reduce_next](classawkward_1_1Content.html#a4f4a15fd7609dbbc75a44e47291c186d)). The problem of actually computing sums and maxima is comparatively simple.

Each reducer has a class that implements the sum, max, etc.

   * [ak::Reducer](classawkward_1_1Reducer.html): the abstract base class.
   * [ak::ReducerCount](classawkward_1_1ReducerCount.html): for [ak.count](../_auto/ak.count.html).
   * [ak::ReducerCountNonzero](classawkward_1_1ReducerCountNonzero.html): for [ak.count_nonzero](../_auto/ak.count_nonzero.html).
   * [ak::ReducerSum](classawkward_1_1ReducerSum.html): for [ak.sum](../_auto/ak.sum.html).
   * [ak::ReducerProd](classawkward_1_1ReducerProd.html): for [ak.prod](../_auto/ak.prod.html).
   * [ak::ReducerAny](classawkward_1_1ReducerAny.html): for [ak.any](../_auto/ak.any.html).
   * [ak::ReducerAll](classawkward_1_1ReducerAll.html): for [ak.all](../_auto/ak.all.html).
   * [ak::ReducerMin](classawkward_1_1ReducerMin.html): for [ak.min](../_auto/ak.min.html).
   * [ak::ReducerMax](classawkward_1_1ReducerMax.html): for [ak.max](../_auto/ak.max.html).
   * [ak::ReducerArgmin](classawkward_1_1ReducerArgmin.html): for [ak.argmin](../_auto/ak.argmin.html).
   * [ak::ReducerArgmax](classawkward_1_1ReducerArgmax.html): for [ak.argmax](../_auto/ak.argmax.html).

### Slices

Many Python objects can be used as slices; they are each converted into a specialized C++ type before being passed to [ak::Content::getitem](classawkward_1_1Content.html#afe0d7eaa5d9d72290e3211efa4003678).

   * [ak::Slice](classawkward_1_1Slice.html): represents a Python tuple of slice items, which selects multiple dimensions at once. A non-tuple slice in Python is wrapped as a single-item [ak::Slice](classawkward_1_1Slice.html) in C++.
   * [ak::SliceItem](classawkward_1_1SliceItem.html): abstract base class for non-tuple slice items.
   * [ak::SliceAt](classawkward_1_1SliceAt.html): represents a single integer as a slice item, which selects an element.
   * [ak::SliceRange](classawkward_1_1SliceRange.html): represents a Python `slice` object with `start`, `stop`, and `step`, which selects a range of elements.
   * [ak::SliceField](classawkward_1_1SliceField.html): represents a single string as a slice item, which selects a record field or projects across all record fields in an array.
   * [ak::SliceFields](classawkward_1_1SliceFields.html): represents an iterable of strings as a slice item, which selects a set of record fields or projects across a set of all record fields in an array.
   * [ak::SliceNewAxis](classawkward_1_1SliceNewAxis.html): represents `np.newaxis` (a.k.a. `None`), which inserts a new regular dimension of length 1 in the sliced output.
   * [ak::SliceEllipsis](classawkward_1_1SliceEllipsis.html): represents `Elipsis` (a.k.a. `...`), which skips enough dimensions to put the rest of the slice items at the deepest possible level.
   * [ak::SliceArrayOf](classawkward_1_1SliceArrayOf.html): represents an integer or boolean array (boolean arrays are converted into integers early), which do general element-selection (rearrangement, filtering, and duplication).
   * [ak::SliceJaggedOf](classawkward_1_1SliceJaggedOf.html): represents a one level of jaggedness in an array as a slice item.
   * [ak::SliceMissingOf](classawkward_1_1SliceMissingOf.html): represents an array with missing values as a slice item.

Note: [ak::SliceGenerator](classawkward_1_1SliceGenerator.html) is not a [ak::SliceItem](classawkward_1_1SliceItem.html); it's an [ak::ArrayGenerator](classawkward_1_1ArrayGenerator.html). (A case of clashing naming conventions.)

### Conversion to JSON

The following classes hide Awkward Array's dependence on RapidJSON, so that it's a swappable component. Header files do not reference RapidJSON, but the implementation of these classes do: [ak::ToJson](classawkward_1_1ToJson.html), [ak::ToJsonString](classawkward_1_1ToJsonString.html), [ak::ToJsonPrettyString](classawkward_1_1ToJsonPrettyString.html), [ak::ToJsonFile](classawkward_1_1ToJsonFile.html), [ak::ToJsonPrettyFile](classawkward_1_1ToJsonPrettyFile.html).

### CPU kernels and GPU kernels

The kernels library is separated from the C++ codebase by a pure C interface, and thus the kernels could be used by other languages.

The CPU kernels are implemented in these files:

   * [kernels/getitem](getitem_8h.html): kernels to implement [ak::Content::getitem](classawkward_1_1Content.html#afe0d7eaa5d9d72290e3211efa4003678).
   * [kernels/identities](kernels_2identities_8h.html): kernels to implement [ak::IdentitiesOf<T>](classawkward_1_1IdentitiesOf.html).
   * [kernels/reducers](reducers_8h.html): kernels to implement [ak::Content::reduce_next](classawkward_1_1Content.html#a4f4a15fd7609dbbc75a44e47291c186d).
   * [kernels/operations](operations_8h.html): kernels to implement all other operations.

The GPU kernels follow exactly the same interface, though a different implementation.

**FIXME:** kernel function names and arguments should be systematized in some searchable way.
