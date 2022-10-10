# API reference

:::{card} C++ Documentation {fas}`external-link-alt`
:link: ../_static/doxygen/index.html

The C++ classes, cpu-kernels, and gpu-kernels are documented separately. Click here to go to the C++ API reference
:::

**High-level data types:** {func}`ak.Array` for an array of items (records, numbers, strings, etc.) and {func}`ak.Record` for a single record. Arrays and records are read-only structures, but functions that manipulate them efficiently share data between the input and output.

**Append-only data type:** {func}`ak.ArrayBuilder` discovers its type from the sequence of append operations called on it.

**Adding methods, overloading operators:** {doc}`ak.behavior` for a global registry; see also for overloading individual arrays.

**Describing an array:** {func}`ak.is_valid`, {func}`ak.validity_error`, {func}`ak.type`, {func}`ak.parameters`, {func}`ak.fields`.

**Converting from other formats:** {func}`ak.from_numpy`, {func}`ak.from_iter`, {func}`ak.from_json`, {func}`ak.from_arrow`, {func}`ak.from_parquet`. Note that the {func}`ak.Array` and {func}`ak.Record` constructors use these functions.

**Converting to other formats:** {func}`ak.to_numpy`, {func}`ak.to_list`, {func}`ak.to_json`, {func}`ak.to_dataframe`, {func}`ak.to_arrow`, {func}`ak.to_parquet`.

**Conversion functions used internally:** {func}`ak.to_layout`, {func}`ak.regularize_numpyarray`.

**Alternative to filtering:** {func}`ak.mask`, which is the same as `array.mask[filter]`. Creates an array with missing values instead of removing values.

**Number of elements in each list:** {func}`ak.num` (not to be confused with the reducer {func}`ak.count`).

**Making and breaking arrays of records:** {func}`ak.zip` and {func}`ak.unzip`.

**Manipulating records:** {func}`ak.with_name`, {func}`ak.with_field`.

**Manipulating parameters:** {func}`ak.with_parameter`, {func}`ak.without_parameters`.

**Broadcasting:** {func}`ak.broadcast_arrays` forms an explicit broadcast of a set of arrays, which usually isn't necessary. This page also describes the general broadcasting rules, though.

**Merging arrays:** {func}`ak.concatenate`, {func}`ak.where`.

**Flattening lists and missing values:** {func}`ak.flatten` removes a level of list structure. Empty lists and None at that level disappear. Also useful for eliminating None in the first dimension.

**Inserting, replacing, and checking for missing values:** {func}`ak.pad_none`, {func}`ak.fill_none`, {func}`ak.is_none`.

**Converting missing values to and from empty lists:** {func}`ak.singletons` turns `[1, None, 3]` into `[[1], [], [3]]` and {func}`ak.firsts` turns `[[1], [], [3]]` into `[1, None, 3]`. This can be useful with {func}`ak.argmin` and {func}`ak.argmax`.

**Combinatorics:** {func}`ak.cartesian` produces tuples of*n* items from*n* arrays, usually per-sublist, and {func}`ak.combinations` produces unique tuples of*n* items from the same array. To get integer arrays for selecting these tuples, use {func}`ak.argcartesian` and {func}`ak.argcombinations`.

**NumPy compatibility:** {func}`ak.size`, {func}`ak.atleast_1d`.

**Reducers:** eliminate a dimension by replacing it with a count, sum, logical and/or, etc. over its members. These functions summarize the innermost lists with `axis=-1` and cross lists with other values of `axis`. They never apply to data structures, only numbers at the innermost fields of a structure.

* {func}`ak.count`: the number of elements (not to be confused with {func}`ak.num`, which interprets `axis` differently from a reducer).
* {func}`ak.count_nonzero`: the number of elements that are not equal to zero or False.
* {func}`ak.sum`: adds values with identity 0.
* {func}`ak.prod`: multiplies values with identity 1.
* {func}`ak.any`: reduces with logical or, "true if*any* members are non-zero."
* {func}`ak.all`: reduces with logical and, "true if*all* members are non-zero."
* {func}`ak.min`: minimum value; empty lists result in None.
* {func}`ak.max`: maximum value; empty lists result in None.
* {func}`ak.argmin`: integer position of the minimum value; empty lists result in None.
* {func}`ak.argmax`: integer position of the maximum value; empty lists result in None.

**Non-reducers:** not technically reducers because they don't obey an associative law (e.g. the mean of means is not the overall mean); these functions nevertheless have the same interface as reducers.

* {func}`ak.moment`: the "nth" moment of the distribution; `0` for sum, `1` for mean, `2` for variance without subtracting the mean, etc.
* {func}`ak.mean`: also known as the average.
* {func}`ak.var`: variance about the mean.
* {func}`ak.std`: standard deviation about the mean.
* {func}`ak.covar`: covariance of two datasets.
* {func}`ak.corr`: correlation of two datasets (covariance normalized to variance).
* {func}`ak.linear_fit`: linear fits, possibly very many of them.
* {func}`ak.softmax`: the softmax function of machine learning.

**String behaviors:** defined in the {doc}`ak.behaviors.string` submodule; rarely needed for analysis (strings are a built-in behavior).

**Numba compatibility:** {func}`ak.numba.register` informs Numba about Awkward Array types; rarely needed because this should happen automatically.

**Pandas compatibility:** {func}`ak.to_dataframe` turns an Awkward Array into a list of DataFrames or joins them with [pd.merge](https://pandas.pydata.org/pandas-docs/version/1.0.3/reference/api/pandas.merge.html) if necessary.

**NumExpr compatibility:** {func}`ak.numexpr.evaluate` and {func}`ak.numexpr.re_evaluate` are like the NumExpr functions, but with Awkward Array support.

**JAX compatibility:** {func}`ak.jax.register_and_check()` registers Awkward Array integration with JAX.

**Layout nodes:** the high-level {func}`ak.Array` and {func}`ak.Record` types hide the tree-structure that build the array, but they can be accessed with [ak.Array.layout](_auto/ak.Array.html#ak-array-layout). This layout structure is the core of the library, but usually doesn't have to be accessed by data analysts.

* {class}`ak.contents.Content`: the abstract base class.
* {class}`ak.contents.EmptyArray`: an array of unknown type with no elements (usually produced by {func}`ak.ArrayBuilder`, which can't determine type at a given level without samples).
* {class}`ak.contents.NumpyArray`: any NumPy array (e.g. multidimensional shape, arbitrary dtype), though usually only one-dimensional arrays of numbers.
* {class}`ak.contents.RegularArray`: splits its nested content into equal-length lists.
* {class}`ak.contents.ListArray`: splits its nested content into variable-length lists with full generality (may use its content non-contiguously, overlapping, or out-of-order).
* {class}`ak.contents.ListOffsetArray`: splits its nested content into variable-length lists, assuming contiguous, non-overlapping, in-order content.
* {class}`ak.contents.RecordArray`: represents a logical array of records with a "struct of arrays" layout in memory.
* {class}`ak.record.Record`: represents a single record (not a subclass of {class}`ak.contents.Content` in Python).
* {class}`ak.contents.IndexedArray`: rearranges and/or duplicates its content by lazily applying an integer index.
* {class}`ak.contents.IndexedOptionArray`: same as {class}`ak.contents.IndexedArray` with missing values as negative indexes.
* {class}`ak.contents.ByteMaskedArray`: represents its content with missing values with an 8-bit boolean mask.
* {class}`ak.contents.BitMaskedArray`: represents its content with missing values with a 1-bit boolean mask.
* {class}`ak.contents.UnmaskedArray`: specifies that its content can contain missing values in principle, but no mask is supplied because all elements are non-missing.
* {class}`ak.contents.UnionArray`: interleaves a set of arrays as a tagged union, can represent heterogeneous data.

Most layout nodes contain another content node ({class}`ak.contents.RecordArray` and {class}`ak.contents.UnionArray` can contain more than one), thus forming a tree. Only {class}`ak.contents.EmptyArray` and {class}`ak.contents.NumpyArray` cannot contain a content, and hence these are leaves of the tree.

**Iterator for layout nodes:** {class}`ak.layout.Iterator` (used internally).

**Layout-level ArrayBuilder:** {class}`ak.layout.ArrayBuilder` (used internally).

**Index for layout nodes:** integer and boolean arrays that define the shape of the data structure, such as boolean masks in {class}`ak.contents.ByteMaskedArray`, are not {class}`ak.contents.NumpyArray` but a more constrained type called {class}`ak.layout.Index`.

**Identities for layout nodes:** {class}`ak.layout.Identities` are an optional surrogate key for certain join operations. (Not yet used.)

**High-level data types:**

This is the type of data in a high-level {func}`ak.Array` or {func}`ak.Record` as reported by {func}`ak.type`. It represents as much information as a data analyst needs to know (e.g. the distinction between variable and fixed-length lists, but not the distinction between {class}`ak.contents.ListArray` and {class}`ak.contents.ListOffsetArray`).

* {class}`ak.types.Type`: the abstract base class.
* {class}`ak.types.ArrayType`: type of a non-composable, high-level {func}`ak.Array`, which includes the length of the array.
* {class}`ak.types.UnknownType`: a type that is not known because it is represented by an {class}`ak.contents.EmptyArray`.
* {class}`ak.types.PrimitiveType`: a numeric or boolean type.
* {class}`ak.types.RegularType`: lists of a fixed length; this `size` is part of the type description.
* {class}`ak.types.ListType`: lists of unspecified or variable length.
* {class}`ak.types.RecordType`: records with named fields or tuples with a fixed number of unnamed slots. The fields/slots and their types are part of the type description.
* {class}`ak.types.OptionType`: data that may be missing.
* {class}`ak.types.UnionType`: heterogeneous data selected from a short list of possibilities.

All concrete {class}`ak.types.Type` subclasses are composable except {class}`ak.types.ArrayType`.

**Low-level array forms:**

This is the type of a {class}`ak.contents.Content` array expressed with low-level granularity (e.g. including the distinction between {class}`ak.contents.ListArray` and {class}`ak.contents.ListOffsetArray`). There is a one-to-one relationship between {class}`ak.contents.Content` subclasses and {class}`ak.forms.Form` subclasses, and each {class}`ak.forms.Form` maps to only one {class}`ak.types.Type`.

* {class}`ak.forms.Form`: the abstract base class.
* {class}`ak.forms.EmptyForm` for {class}`ak.contents.EmptyArray`.
* {class}`ak.forms.NumpyForm` for {class}`ak.contents.NumpyArray`.
* {class}`ak.forms.RegularForm` for {class}`ak.contents.RegularArray`.
* {class}`ak.forms.ListForm` for {class}`ak.contents.ListArray`.
* {class}`ak.forms.ListOffsetForm` for {class}`ak.contents.ListOffsetArray`.
* {class}`ak.forms.RecordForm` for {class}`ak.contents.RecordArray`.
* {class}`ak.forms.IndexedForm` for {class}`ak.contents.IndexedArray`.
* {class}`ak.forms.IndexedOptionForm` for {class}`ak.contents.IndexedOptionArray`.
* {class}`ak.forms.ByteMaskedForm` for {class}`ak.contents.ByteMaskedArray`.
* {class}`ak.forms.BitMaskedForm` for {class}`ak.contents.BitMaskedArray`.
* {class}`ak.forms.UnmaskedForm` for {class}`ak.contents.UnmaskedArray`.
* {class}`ak.forms.UnionForm` for {class}`ak.contents.UnionArray`.

## Additional documentation

The Awkward Array project is divided into 3 layers with 5 main components.

:::{figure} ../image/awkward-1-0-layers.svg
:align: center

:::
 
% Use HTML link as HTML files are considered assets, not source files

The C++ classes, cpu-kernels, and gpu-kernels are described in the  <a href="../_static/doxygen/index.html">C++ API reference</a>.

The kernels (cpu-kernels and cuda-kernels) are documented on the {doc}`generated/kernels` page, with interfaces and normative Python implementations.

:::{eval-rst}
.. include:: generated/toctree.txt
:::
