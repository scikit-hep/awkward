![](https://raw.githubusercontent.com/scikit-hep/awkward-1.0/master/docs-img/logo/logo-300px.png)

Awkward Array is a library for **nested, variable-sized data**, including arbitrary-length lists, records, mixed types, and missing data, using **NumPy-like idioms**.

Arrays are **dynamically typed**, but operations on them are **compiled and fast**. Their behavior coincides with NumPy when array dimensions are regular and generalizes when they’re not.

## Documentation

   * [How-to documentation for data analysts](https://awkward-array.org/how-do-i.html)
   * [How-it-works tutorials for developers](https://awkward-array.org/how-it-works.html)
   * C++ API reference: **this site**
   * [Python API reference](../index.html)

## Navigation

The Awkward Array project is divided into 3 layers with 5 main components.

<img src="awkward-1-0-layers.svg" style="max-width: 500px">

The high-level interface and Numba implementation are described in the [Python API reference](../index.html), though the Numba implementation is considered an internal detail (in the `_connect` submodule, which starts with an underscore and is therefore private in Python).

This reference describes the

   * **C++ classes:** in [namespace awkward](namespaceawkward.html) (often abbreviated as "ak"), which are compiled into **libawkward.so** (or dylib or lib). This library is fully usable from C++, without Python (see [dependent-project](https://github.com/scikit-hep/awkward-1.0/tree/master/dependent-project) for an example).
   * **pybind11 interface:** no namespace, but contained entirely within the [python directory](dir_91f33a3f1dd6262845ebd1570075970c.html), which are compiled into **awkward1._ext** for use in Python.
   * **CPU kernels:** no namespace, but contained entirely within the [cpu-kernels directory](dir_6225843069e7cc68401bbec110a1667f.html), which are compiled into **libawkward-cpu-kernels.so** (or dylib or lib). This library is fully usable from any language that can call functions through [FFI](https://en.wikipedia.org/wiki/Foreign_function_interface).
   * **GPU kernels:** FIXME! (not implemented yet)

### Array layout nodes

Awkward Arrays are a tree of layout nodes nested within each other, some of which carry one-dimensional arrays that get reinterpreted as part of the structure. These descend from the abstract base class [ak::Content](classawkward_1_1Content.html) and are reflected through pybind11 as [ak.layout.Content](../ak.layout.Content.html) and subclasses in Python.

   * [ak::Content](classawkward_1_1Content.html): the abstract superclass.
   * [ak::EmptyArray](classawkward_1_1EmptyArray.html): an array of unknown type with no elements (usually produced by [ak::ArrayBuilder](classawkward_1_1ArrayBuilder.html), which can't determine type at a given level without samples).
   * [ak::NumpyArray](classawkward_1_1NumpyArray.html): any NumPy array (e.g. multidimensional shape, arbitrary dtype), though usually only one-dimensional arrays of numbers.
   * [ak::RegularArray](classawkward_1_1RegularArray.html): splits its nested content into equal-length lists.
   * [ak::ListArrayOf<T>](classawkward_1_1ListArrayOf.html): splits its nested content into variable-length lists with full generality (may use its content non-contiguously, overlapping, or out-of-order).
   * [ak::ListOffsetArrayOf<T>](classawkward_1_1ListOffsetArrayOf.html): splits its nested content into variable-length lists, assuming contiguous, non-overlapping, in-order content.
   * [ak::RecordArray](classawkward_1_1RecordArray.html): represents an array of records in "struct of arrays" form.
   * [ak::Record](classawkward_1_1Record.html): represents a single record (a subclass of [ak::Content](classawkward_1_1Content.html) in C++, but not Python).
   * [ak::IndexedArrayOf<T, ISOPTION>](classawkward_1_1IndexedArrayOf.html): rearranges and/or duplicates its content by lazily applying an integer index.
   * [ak::ByteMaskedArray](classawkward_1_1ByteMaskedArray.html): represents its content with missing values with an 8-bit boolean mask.
   * [ak::BitMaskedArray](classawkward_1_1BitMaskedArray.html): represents its content with missing values with a 1-bit boolean mask.
   * [ak::UnmaskedArray](classawkward_1_1UnmaskedArray.html): specifies that its content can contain missing values in principle, but no mask is supplied because all elements are non-missing.
   * [ak::UnionArrayOf<T, I>](classawkward_1_1UnionArrayOf.html): interleaves a set of arrays as a tagged union, can represent heterogeneous data.
   * [ak::VirtualArray](classawkward_1_1VirtualArray.html): generates an array on demand from an [ak::ArrayGenerator](classawkward_1_1ArrayGenerator.html) or a [ak::SliceGenerator](classawkward_1_1SliceGenerator.html) and optionally caches the generated array in an [ak::ArrayCache](classawkward_1_1ArrayCache.html).
   * [ak::None](classawkward_1_1None.html): represents a missing value that will be converted to `None` in Python (a subclass of [ak::Content](classawkward_1_1Content.html) in C++).

The [ak::Record](classawkward_1_1Record.html), [ak::None](classawkward_1_1None.html), and [ak::NumpyArray](classawkward_1_1NumpyArray.html) with empty [shape](classawkward_1_1NumpyArray.html#ab4eec3bfd0e50bc035c26e62974d209d) are technically [ak::Content](classawkward_1_1Content.html) in C++ even though they represent scalar data, rather than arrays. (This can be checked with the [isscalar](classawkward_1_1Content.html#a878ae38b66c14067b231469863d6d1a1) method.) This is because they are possible return values of methods that would ordinarily return [ak::Contents](classawkward_1_1Content.html), so they are subclasses to simplify the type hierarchy. However, in the [Python layer](dir_91f33a3f1dd6262845ebd1570075970c.html), they are converted directly into Python scalars, such as [ak.layout.Record](../ak.layout.Record.html) (which isn't an [ak.layout.Content](../ak.layout.Content.html) subclass), Python's `None`, or a Python number/bool.

Most layout nodes contain another content node ([ak::RecordArray](classawkward_1_1RecordArray.html) and [ak::UnionArrayOf<T, I>](classawkward_1_1UnionArrayOf.html) can contain more than one), thus forming a tree. Only [ak::EmptyArray](classawkward_1_1EmptyArray.html) and [ak::NumpyArray](classawkward_1_1NumpyArray.html) cannot contain a content, and hence these are leaves of the tree.

Note that [ak::PartitionedArray](classawkward_1_1PartitionedArray.html) and its concrete class, [ak::IrregularlyPartitionedArray](classawkward_1_1IrregularlyPartitionedArray.html), are not [ak::Content](classawkward_1_1Content.html) because they cannot be nested within a tree. Partitioning is only allowed at the root of the tree.

**Iterator for layout nodes:** [ak::Iterator](classawkward_1_1Iterator.html).

**Index for layout nodes:** integer and boolean arrays that define the shape of the data structure, such as boolean masks in [ak::ByteMaskedArray](classawkward_1_1ByteMaskedArray.html), are not [ak::NumpyArray](classawkward_1_1NumpyArray.html) but a more constrained type called [ak::IndexOf<T>](classawkward_1_1IndexOf.html).

**Identities for layout nodes:** [ak::IdentitiesOf<T>](classawkward_1_1IdentitiesOf.html) are an optional surrogate key for certain join operations. (Not yet used.)

### High-level data types

This is the type of data in a high-level [ak.Array](../_auto/ak.Array.html) or [ak.Record](../_auto/ak.Record.html) as reported by [ak.type](../_auto/ak.type.html). It represents as much information as a data analyst needs to know (e.g. the distinction between variable and fixed-length lists, but not the distinction between [ak::ListArrayOf<T>](classawkward_1_1ListArrayOf.html) and [ak::ListOffsetArrayOf<T>](classawkward_1_1ListOffsetArrayOf.html)).

   * [ak::Type](classawkward_1_1Type.html): the abstract superclass.
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

   * [ak::Form](classawkward_1_1Form.html): the abstract subclass.
   * [ak::EmptyForm](classawkward_1_1EmptyForm.html) for [ak::EmptyArray](classawkward_1_1EmptyArray.html)
   * [ak::NumpyForm](classawkward_1_1NumpyForm.html) for [ak::NumpyArray](classawkward_1_1NumpyArray.html)
   * [ak::RegularForm](classawkward_1_1RegularForm.html) for [ak::RegularArray](classawkward_1_1RegularArray.html)
   * [ak::ListForm](classawkward_1_1ListForm.html) for [ak::ListArrayOf<T>](classawkward_1_1ListArrayOf.html)
   * [ak::ListOffsetForm](classawkward_1_1ListOffsetForm.html) for [ak::ListOffsetArrayOf<T>](classawkward_1_1ListOffsetArrayOf.html)
   * [ak::RecordForm](classawkward_1_1RecordForm.html) for [ak::RecordArray](classawkward_1_1RecordArray.html)
   * [ak::IndexedForm](classawkward_1_1IndexedForm.html) for [ak::IndexedArrayOf<T, ISOPTION>](classawkward_1_1IndexedArrayOf.html) with `ISOPTION = false`
   * [ak::IndexedOptionForm](classawkward_1_1IndexedOptionForm.html) for [ak::IndexedArrayOf<T, ISOPTION>](classawkward_1_1IndexedArrayOf.html) with `ISOPTION = true`
   * [ak::ByteMaskedForm](classawkward_1_1ByteMaskedForm.html) for [ak::ByteMaskedArray](classawkward_1_1ByteMaskedArray.html)
   * [ak::BitMaskedForm](classawkward_1_1BitMaskedForm.html) for [ak::BitMaskedArray](classawkward_1_1BitMaskedArray.html)
   * [ak::UnmaskedForm](classawkward_1_1UnmaskedForm.html) for [ak::UnmaskedArray](classawkward_1_1UnmaskedArray.html)
   * [ak::UnionForm](classawkward_1_1UnionForm.html) for [ak::UnionArrayOf<T, I>](classawkward_1_1UnionArrayOf.html)
   * [ak::VirtualForm](classawkward_1_1VirtualForm.html) for [ak::VirtualArray](classawkward_1_1VirtualArray.html)

### ArrayBuilder structure

