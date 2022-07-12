# API Reference

*High-level data types:* {doc}`_auto/awkward._v2.Array` for an array of items (records, numbers, strings, etc.) and {doc}`_auto/awkward._v2.Record` for a single record. Arrays and records are read-only structures, but functions that manipulate them efficiently share data between the input and output.

*Append-only data type:* {doc}`_auto/awkward._v2.ArrayBuilder` discovers its type from the sequence of append operations called on it.

*Adding methods, overloading operators:* {doc}`_auto/awkward._v2.behavior` for a global registry; see also for overloading individual arrays.

*Describing an array:* {doc}`_auto/awkward._v2.is_valid`, {doc}`_auto/awkward._v2.validity_error`, {doc}`_auto/awkward._v2.type`, {doc}`_auto/awkward._v2.parameters`, {doc}`_auto/awkward._v2.fields`.

*Converting from other formats:* {doc}`_auto/awkward._v2.from_numpy`, {doc}`_auto/awkward._v2.from_iter`, {doc}`_auto/awkward._v2.from_json`. Note that the {doc}`_auto/awkward._v2.Array` and {doc}`_auto/awkward._v2.Record` constructors use these functions.

*Converting to other formats:* {doc}`_auto/awkward._v2.to_numpy`, {doc}`_auto/awkward._v2.to_list`, {doc}`_auto/awkward._v2.to_json`.

*Conversion functions used internally:* {doc}`_auto/awkward._v2.to_layout`, {doc}`_auto/awkward._v2.regularize_numpyarray`.

*Alternative to filtering:* {doc}`_auto/awkward._v2.mask`, which is the same as `array.mask[filter]`. Creates an array with missing values instead of removing values.

*Number of elements in each list:* {doc}`_auto/awkward._v2.num` (not to be confused with the reducer {doc}`_auto/awkward._v2.count`).

*Making and breaking arrays of records:* {doc}`_auto/awkward._v2.zip` and {doc}`_auto/awkward._v2.unzip`.

*Manipulating records:* {doc}`_auto/awkward._v2.with_name`, {doc}`_auto/awkward._v2.with_field`.

*Manipulating parameters:* {doc}`_auto/awkward._v2.with_parameter`, {doc}`_auto/awkward._v2.without_parameters`.

*Broadcasting:* {doc}`_auto/awkward._v2.broadcast_arrays` forms an explicit broadcast of a set of arrays, which usually isn't necessary. This page also describes the general broadcasting rules, though.

*Merging arrays:* {doc}`_auto/awkward._v2.concatenate`, {doc}`_auto/awkward._v2.where`.

*Flattening lists and missing values:* {doc}`_auto/awkward._v2.flatten` removes a level of list structure. Empty lists and None at that level disappear. Also useful for eliminating None in the first dimension.

*Inserting, replacing, and checking for missing values:* {doc}`_auto/awkward._v2.pad_none`, {doc}`_auto/awkward._v2.fill_none`, {doc}`_auto/awkward._v2.is_none`.

*Converting missing values to and from empty lists:* {doc}`_auto/awkward._v2.singletons` turns `[1, None, 3]` into `[[1], [], [3]]` and {doc}`_auto/awkward._v2.firsts` turns `[[1], [], [3]]` into `[1, None, 3]`. This can be useful with {doc}`_auto/awkward._v2.argmin` and {doc}`_auto/awkward._v2.argmax`.

*Combinatorics:* {doc}`_auto/awkward._v2.cartesian` produces tuples of *n* items from *n* arrays, usually per-sublist, and {doc}`_auto/awkward._v2.combinations` produces unique tuples of *n* items from the same array. To get integer arrays for selecting these tuples, use {doc}`_auto/awkward._v2.argcartesian` and {doc}`_auto/awkward._v2.argcombinations`.

*NumPy compatibility:* {doc}`_auto/awkward._v2.size`, {doc}`_auto/awkward._v2.atleast_1d`.

*Reducers:* eliminate a dimension by replacing it with a count, sum, logical and/or, etc. over its members. These functions summarize the innermost lists with `axis=-1` and cross lists with other values of `axis`. They never apply to data structures, only numbers at the innermost fields of a structure.

   * {doc}`_auto/awkward._v2.count`: the number of elements (not to be confused with {doc}`_auto/awkward._v2.num`, which interprets `axis` differently from a reducer).
   * {doc}`_auto/awkward._v2.count_nonzero`: the number of elements that are not equal to zero or False.
   * {doc}`_auto/awkward._v2.sum`: adds values with identity 0.
   * {doc}`_auto/awkward._v2.prod`: multiplies values with identity 1.
   * {doc}`_auto/awkward._v2.any`: reduces with logical or, "true if *any* members are non-zero."
   * {doc}`_auto/awkward._v2.all`: reduces with logical and, "true if *all* members are non-zero."
   * {doc}`_auto/awkward._v2.min`: minimum value; empty lists result in None.
   * {doc}`_auto/awkward._v2.max`: maximum value; empty lists result in None.
   * {doc}`_auto/awkward._v2.argmin`: integer position of the minimum value; empty lists result in None.
   * {doc}`_auto/awkward._v2.argmax`: integer position of the maximum value; empty lists result in None.

*Non-reducers:* not technically reducers because they don't obey an associative law (e.g. the mean of means is not the overall mean); these functions nevertheless have the same interface as reducers.

   * {doc}`_auto/awkward._v2.moment`: the "nth" moment of the distribution; `0` for sum, `1` for mean, `2` for variance without subtracting the mean, etc.
   * {doc}`_auto/awkward._v2.mean`: also known as the average.
   * {doc}`_auto/awkward._v2.var`: variance about the mean.
   * {doc}`_auto/awkward._v2.std`: standard deviation about the mean.
   * {doc}`_auto/awkward._v2.covar`: covariance of two datasets.
   * {doc}`_auto/awkward._v2.corr`: correlation of two datasets (covariance normalized to variance).
   * {doc}`_auto/awkward._v2.linear_fit`: linear fits, possibly very many of them.
   * {doc}`_auto/awkward._v2.softmax`: the softmax function of machine learning.

*String behaviors:* defined in the `awkward.behaviors.string` submodule; rarely needed for analysis (strings are a built-in behavior).

*Numba compatibility:* {doc}`_auto/awkward._v2.numba.register` informs Numba about Awkward Array types; rarely needed because this should happen automatically.

*Pandas compatibility:* {doc}`_auto/awkward._v2.to_pandas` turns an Awkward Array into a list of DataFrames or joins them with `pd.merge <https://pandas.pydata.org/pandas-docs/version/1.0.3/reference/api/pandas.merge.html>`__ if necessary.

*NumExpr compatibility:* {doc}`_auto/awkward._v2.numexpr.evaluate` and {doc}`_auto/awkward._v2.numexpr.re_evaluate` are like the NumExpr functions, but with Awkward Array support.

*Autograd compatibility:* {doc}`_auto/awkward._v2.autograd.elementwise_grad` is like the Autograd function, but with Awkward Array support.

*Layout nodes:* the high-level {doc}`_auto/awkward._v2.Array` and {doc}`_auto/awkward._v2.Record` types hide the tree-structure that build the array, but they can be accessed with `awkward.Array.layout <_auto/awkward._v2.Array.html#ak-array-layout>`_. This layout structure is the core of the library, but usually doesn't have to be accessed by data analysts.

   * {doc}`_auto/awkward._v2.contents.Content`: the abstract base class.
   * {doc}`_auto/awkward._v2.contents.EmptyArray`: an array of unknown type with no elements (usually produced by {doc}`_auto/awkward._v2.ArrayBuilder`, which can't determine type at a given level without samples).
   * {doc}`_auto/awkward._v2.contents.NumpyArray`: any NumPy array (e.g. multidimensional shape, arbitrary dtype), though usually only one-dimensional arrays of numbers.
   * {doc}`_auto/awkward._v2.contents.RegularArray`: splits its nested content into equal-length lists.
   * {doc}`_auto/awkward._v2.contents.ListArray`: splits its nested content into variable-length lists with full generality (may use its content non-contiguously, overlapping, or out-of-order).
   * {doc}`_auto/awkward._v2.contents.ListOffsetArray`: splits its nested content into variable-length lists, assuming contiguous, non-overlapping, in-order content.
   * {doc}`_auto/awkward._v2.contents.RecordArray`: represents a logical array of records with a "struct of arrays" layout in memory.
   * {doc}`_auto/awkward._v2.record.Record`: represents a single record (not a subclass of {doc}`_auto/awkward._v2.contents.Content` in Python).
   * {doc}`_auto/awkward._v2.contents.IndexedArray`: rearranges and/or duplicates its content by lazily applying an integer index.
   * {doc}`_auto/awkward._v2.contents.IndexedOptionArray`: same as {doc}`_auto/awkward._v2.contents.IndexedArray` with missing values as negative indexes.
   * {doc}`_auto/awkward._v2.contents.ByteMaskedArray`: represents its content with missing values with an 8-bit boolean mask.
   * {doc}`_auto/awkward._v2.contents.BitMaskedArray`: represents its content with missing values with a 1-bit boolean mask.
   * {doc}`_auto/awkward._v2.contents.UnmaskedArray`: specifies that its content can contain missing values in principle, but no mask is supplied because all elements are non-missing.
   * {doc}`_auto/awkward._v2.contents.UnionArray`: interleaves a set of arrays as a tagged union, can represent heterogeneous data.

Most layout nodes contain another content node ({doc}`_auto/awkward._v2.contents.RecordArray` and {doc}`_auto/awkward._v2.contents.UnionArray` can contain more than one), thus forming a tree. Only {doc}`_auto/awkward._v2.contents.EmptyArray` and {doc}`_auto/awkward._v2.contents.NumpyArray` cannot contain a content, and hence these are leaves of the tree.


*Iterator for layout nodes:* {doc}`_auto/awkward._v2.contents.Iterator` (used internally).

*Layout-level ArrayBuilder:* {doc}`_auto/awkward._v2.contents.ArrayBuilder` (used internally).

*Index for layout nodes:* integer and boolean arrays that define the shape of the data structure, such as boolean masks in {doc}`_auto/awkward._v2.contents.ByteMaskedArray`, are not {doc}`_auto/awkward._v2.contents.NumpyArray` but a more constrained type called {doc}`_auto/awkward._v2.contents.Index`.

*Identities for layout nodes:* {doc}`_auto/awkward._v2.contents.Identities` are an optional surrogate key for certain join operations. (Not yet used.)

*High-level data types:*

This is the type of data in a high-level {doc}`_auto/awkward._v2.Array` or {doc}`_auto/awkward._v2.Record` as reported by {doc}`_auto/awkward._v2.type`. It represents as much information as a data analyst needs to know (e.g. the distinction between variable and fixed-length lists, but not the distinction between {doc}`_auto/awkward._v2.contents.ListArray` and {doc}`_auto/awkward._v2.contents.ListOffsetArray`).

   * {doc}`_auto/awkward._v2.types.Type`: the abstract base class.
   * {doc}`_auto/awkward._v2.types.ArrayType`: type of a non-composable, high-level {doc}`_auto/awkward._v2.Array`, which includes the length of the array.
   * {doc}`_auto/awkward._v2.types.UnknownType`: a type that is not known because it is represented by an {doc}`_auto/awkward._v2.contents.EmptyArray`.
   * {doc}`_auto/awkward._v2.types.PrimitiveType`: a numeric or boolean type.
   * {doc}`_auto/awkward._v2.types.RegularType`: lists of a fixed length; this `size` is part of the type description.
   * {doc}`_auto/awkward._v2.types.ListType`: lists of unspecified or variable length.
   * {doc}`_auto/awkward._v2.types.RecordType`: records with named fields or tuples with a fixed number of unnamed slots. The fields/slots and their types are part of the type description.
   * {doc}`_auto/awkward._v2.types.OptionType`: data that may be missing.
   * {doc}`_auto/awkward._v2.types.UnionType`: heterogeneous data selected from a short list of possibilities.

All concrete {doc}`_auto/awkward._v2.types.Type` subclasses are composable except {doc}`_auto/awkward._v2.types.ArrayType`.

*Low-level array forms:*

This is the type of a {doc}`_auto/awkward._v2.contents.Content` array expressed with low-level granularity (e.g. including the distinction between {doc}`_auto/awkward._v2.contents.ListArray` and {doc}`_auto/awkward._v2.contents.ListOffsetArray`). There is a one-to-one relationship between {doc}`_auto/awkward._v2.contents.Content` subclasses and {doc}`_auto/awkward._v2.forms.Form` subclasses, and each {doc}`_auto/awkward._v2.forms.Form` maps to only one {doc}`_auto/awkward._v2.types.Type`.

   * {doc}`_auto/awkward._v2.forms.Form`: the abstract base class.
   * {doc}`_auto/awkward._v2.forms.EmptyForm` for {doc}`_auto/awkward._v2.contents.EmptyArray`.
   * {doc}`_auto/awkward._v2.forms.NumpyForm` for {doc}`_auto/awkward._v2.contents.NumpyArray`.
   * {doc}`_auto/awkward._v2.forms.RegularForm` for {doc}`_auto/awkward._v2.contents.RegularArray`.
   * {doc}`_auto/awkward._v2.forms.ListForm` for {doc}`_auto/awkward._v2.contents.ListArray`.
   * {doc}`_auto/awkward._v2.forms.ListOffsetForm` for {doc}`_auto/awkward._v2.contents.ListOffsetArray`.
   * {doc}`_auto/awkward._v2.forms.RecordForm` for {doc}`_auto/awkward._v2.contents.RecordArray`.
   * {doc}`_auto/awkward._v2.forms.IndexedForm` for {doc}`_auto/awkward._v2.contents.IndexedArray`.
   * {doc}`_auto/awkward._v2.forms.IndexedOptionForm` for {doc}`_auto/awkward._v2.contents.IndexedOptionArray`.
   * {doc}`_auto/awkward._v2.forms.ByteMaskedForm` for {doc}`_auto/awkward._v2.contents.ByteMaskedArray`.
   * {doc}`_auto/awkward._v2.forms.BitMaskedForm` for {doc}`_auto/awkward._v2.contents.BitMaskedArray`.
   * {doc}`_auto/awkward._v2.forms.UnmaskedForm` for {doc}`_auto/awkward._v2.contents.UnmaskedArray`.
   * {doc}`_auto/awkward._v2.forms.UnionForm` for {doc}`_auto/awkward._v2.contents.UnionArray`.

## Internal implementation

The rest of the classes and functions described here are not part of the public interface. Either the objects or the submodules begin with an underscore, indicating that they can freely change from one version to the next.

## More documentation

The Awkward Array project is divided into 3 layers with 5 main components.

```{image} _static/awkward-1-0-layers.svg
:width: 500px
:align: center
```

The C++ classes, cpu-kernels, and gpu-kernels are described in the [C++ API reference](_static/index.html>).

The kernels (cpu-kernels and cuda-kernels) are documented on the {doc}`_auto/kernels` page, with interfaces and normative Python implementations.
