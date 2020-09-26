.. |br| raw:: html

    <br/>

.. image:: https://github.com/scikit-hep/awkward-1.0/raw/master/docs-img/logo/logo-300px.png
    :width: 300px
    :alt: Awkward Array

|br| Awkward Array is a library for **nested, variable-sized data**, including arbitrary-length lists, records, mixed types, and missing data, using **NumPy-like idioms**.

Arrays are **dynamically typed**, but operations on them are **compiled and fast**. Their behavior coincides with NumPy when array dimensions are regular and generalizes when they're not.

Documentation
*************

   * `Main page (quickstart) <https://awkward-array.org>`__
   * `C++ API reference <_static/index.html>`__
   * Python API reference: **this site**
   * `GitHub repository <https://github.com/scikit-hep/awkward-1.0>`__

Navigation
**********

**High-level data types:** :doc:`_auto/ak.Array` for an array of items (records, numbers, strings, etc.) and :doc:`_auto/ak.Record` for a single record. Arrays and records are read-only structures, but functions that manipulate them efficiently share data between the input and output.

**Append-only data type:** :doc:`_auto/ak.ArrayBuilder` discovers its type from the sequence of append operations called on it.

**Adding methods, overloading operators:** :doc:`ak.behavior` for a global registry; see also for overloading individual arrays.

**Describing an array:** :doc:`_auto/ak.is_valid`, :doc:`_auto/ak.validity_error`, :doc:`_auto/ak.type`, :doc:`_auto/ak.parameters`, :doc:`_auto/ak.keys`.

**Converting from other formats:** :doc:`_auto/ak.from_numpy`, :doc:`_auto/ak.from_iter`, :doc:`_auto/ak.from_json`, :doc:`_auto/ak.from_awkward0`. Note that the :doc:`_auto/ak.Array` and :doc:`_auto/ak.Record` constructors use these functions.

**Converting to other formats:** :doc:`_auto/ak.to_numpy`, :doc:`_auto/ak.to_list`, :doc:`_auto/ak.to_json`, :doc:`_auto/ak.to_awkward0`.

**Conversion functions used internally:** :doc:`_auto/ak.to_layout`, :doc:`_auto/ak.regularize_numpyarray`.

**Alternative to filtering:** :doc:`_auto/ak.mask`, which is the same as ``array.mask[filter]``. Creates an array with missing values instead of removing values.

**Number of elements in each list:** :doc:`_auto/ak.num` (not to be confused with the reducer :doc:`_auto/ak.count`).

**Making and breaking arrays of records:** :doc:`_auto/ak.zip` and :doc:`_auto/ak.unzip`.

**Manipulating records:** :doc:`_auto/ak.with_name`, :doc:`_auto/ak.with_field`.

**Manipulating parameters:** :doc:`_auto/ak.with_parameter`, :doc:`_auto/ak.without_parameters`.

**Broadcasting:** :doc:`_auto/ak.broadcast_arrays` forms an explicit broadcast of a set of arrays, which usually isn't necessary. This page also describes the general broadcasting rules, though.

**Merging arrays:** :doc:`_auto/ak.concatenate`, :doc:`_auto/ak.where`.

**Flattening lists and missing values:** :doc:`_auto/ak.flatten` removes a level of list structure. Empty lists and None at that level disappear. Also useful for eliminating None in the first dimension.

**Inserting, replacing, and checking for missing values:** :doc:`_auto/ak.pad_none`, :doc:`_auto/ak.fill_none`, :doc:`_auto/ak.is_none`.

**Converting missing values to and from empty lists:** :doc:`_auto/ak.singletons` turns ``[1, None, 3]`` into ``[[1], [], [3]]`` and :doc:`_auto/ak.firsts` turns ``[[1], [], [3]]`` into ``[1, None, 3]``. This can be useful with :doc:`_auto/ak.argmin` and :doc:`_auto/ak.argmax`.

**Combinatorics:** :doc:`_auto/ak.cartesian` produces tuples of *n* items from *n* arrays, usually per-sublist, and :doc:`_auto/ak.combinations` produces unique tuples of *n* items from the same array. To get integer arrays for selecting these tuples, use :doc:`_auto/ak.argcartesian` and :doc:`_auto/ak.argcombinations`.

**Partitioned arrays:** :doc:`_auto/ak.partitions` reveals how an array is internally partitioned (if at all) and :doc:`_auto/ak.partitioned`, :doc:`_auto/ak.repartition` create or change the partitioning.

**Virtual arrays:** :doc:`_auto/ak.virtual` creates an array that will be generated on demand and :doc:`_auto/ak.with_cache` assigns a new cache to all virtual arrays in a structure.

**NumPy compatibility:** :doc:`_auto/ak.size`, :doc:`_auto/ak.atleast_1d`.

**Reducers:** eliminate a dimension by replacing it with a count, sum, logical and/or, etc. over its members. These functions summarize the innermost lists with ``axis=-1`` and cross lists with other values of ``axis``. They never apply to data structures, only numbers at the innermost fields of a structure.

   * :doc:`_auto/ak.count`: the number of elements (not to be confused with :doc:`_auto/ak.num`, which interprets ``axis`` differently from a reducer).
   * :doc:`_auto/ak.count_nonzero`: the number of elements that are not equal to zero or False.
   * :doc:`_auto/ak.sum`: adds values with identity 0.
   * :doc:`_auto/ak.prod`: multiplies values with identity 1.
   * :doc:`_auto/ak.any`: reduces with logical or, "true if *any* members are non-zero."
   * :doc:`_auto/ak.all`: reduces with logical and, "true if *all* members are non-zero."
   * :doc:`_auto/ak.min`: minimum value; empty lists result in None.
   * :doc:`_auto/ak.max`: maximum value; empty lists result in None.
   * :doc:`_auto/ak.argmin`: integer position of the minimum value; empty lists result in None.
   * :doc:`_auto/ak.argmax`: integer position of the maximum value; empty lists result in None.

**Non-reducers:** not technically reducers because they don't obey an associative law (e.g. the mean of means is not the overall mean); these functions nevertheless have the same interface as reducers.

   * :doc:`_auto/ak.moment`: the "nth" moment of the distribution; ``0`` for sum, ``1`` for mean, ``2`` for variance without subtracting the mean, etc.
   * :doc:`_auto/ak.mean`: also known as the average.
   * :doc:`_auto/ak.var`: variance about the mean.
   * :doc:`_auto/ak.std`: standard deviation about the mean.
   * :doc:`_auto/ak.covar`: covariance of two datasets.
   * :doc:`_auto/ak.corr`: correlation of two datasets (covariance normalized to variance).
   * :doc:`_auto/ak.linear_fit`: linear fits, possibly very many of them.
   * :doc:`_auto/ak.softmax`: the softmax function of machine learning.

**String behaviors:** defined in the ``ak.behaviors.string`` submodule; rarely needed for analysis (strings are a built-in behavior).

**Partition functions:** defined in the ``ak.partition`` submodule; rarely needed for analysis: use :doc:`_auto/ak.partitions`, :doc:`_auto/ak.partitioned`, :doc:`_auto/ak.repartition`.

**Numba compatibility:** :doc:`ak.numba.register` informs Numba about Awkward Array types; rarely needed because this should happen automatically.

**Pandas compatibility:** :doc:`ak.to_pandas` turns an Awkward Array into a list of DataFrames or joins them with `pd.merge <https://pandas.pydata.org/pandas-docs/version/1.0.3/reference/api/pandas.merge.html>`__ if necessary.

**NumExpr compatibility:** :doc:`ak.numexpr.evaluate` and :doc:`ak.numexpr.re_evaluate` are like the NumExpr functions, but with Awkward Array support.

**Autograd compatibility:** :doc:`ak.autograd.elementwise_grad` is like the Autograd function, but with Awkward Array support.

**Layout nodes:** the high-level :doc:`_auto/ak.Array` and :doc:`_auto/ak.Record` types hide the tree-structure that build the array, but they can be accessed with `ak.Array.layout <_auto/ak.Array.html#ak-array-layout>`_. This layout structure is the core of the library, but usually doesn't have to be accessed by data analysts.

   * :doc:`ak.layout.Content`: the abstract base class.
   * :doc:`ak.layout.EmptyArray`: an array of unknown type with no elements (usually produced by :doc:`_auto/ak.ArrayBuilder`, which can't determine type at a given level without samples).
   * :doc:`ak.layout.NumpyArray`: any NumPy array (e.g. multidimensional shape, arbitrary dtype), though usually only one-dimensional arrays of numbers.
   * :doc:`ak.layout.RegularArray`: splits its nested content into equal-length lists.
   * :doc:`ak.layout.ListArray`: splits its nested content into variable-length lists with full generality (may use its content non-contiguously, overlapping, or out-of-order).
   * :doc:`ak.layout.ListOffsetArray`: splits its nested content into variable-length lists, assuming contiguous, non-overlapping, in-order content.
   * :doc:`ak.layout.RecordArray`: represents an array of records in "struct of arrays" form.
   * :doc:`ak.layout.Record`: represents a single record (not a subclass of :doc:`ak.layout.Content` in Python).
   * :doc:`ak.layout.IndexedArray`: rearranges and/or duplicates its content by lazily applying an integer index.
   * :doc:`ak.layout.IndexedOptionArray`: same as :doc:`ak.layout.IndexedArray` with missing values as negative indexes.
   * :doc:`ak.layout.ByteMaskedArray`: represents its content with missing values with an 8-bit boolean mask.
   * :doc:`ak.layout.BitMaskedArray`: represents its content with missing values with a 1-bit boolean mask.
   * :doc:`ak.layout.UnmaskedArray`: specifies that its content can contain missing values in principle, but no mask is supplied because all elements are non-missing.
   * :doc:`ak.layout.UnionArray`: interleaves a set of arrays as a tagged union, can represent heterogeneous data.
   * :doc:`ak.layout.VirtualArray`: generates an array on demand from an :doc:`ak.layout.ArrayGenerator` or a :doc:`ak.layout.SliceGenerator` and optionally caches the generated array in an :doc:`ak.layout.ArrayCache`.

Most layout nodes contain another content node (:doc:`ak.layout.RecordArray` and :doc:`ak.layout.UnionArray` can contain more than one), thus forming a tree. Only :doc:`ak.layout.EmptyArray` and :doc:`ak.layout.NumpyArray` cannot contain a content, and hence these are leaves of the tree.

Note that :doc:`_auto/ak.partition.PartitionedArray` and its concrete class,  :doc:`_auto/ak.partition.IrregularlyPartitionedArray`, are not :doc:`ak.layout.Content` because they cannot be nested within a tree. Partitioning is only allowed at the root of the tree.

**Iterator for layout nodes:** :doc:`ak.layout.Iterator` (used internally).

**Layout-level ArrayBuilder:** :doc:`ak.layout.ArrayBuilder` (used internally).

**Index for layout nodes:** integer and boolean arrays that define the shape of the data structure, such as boolean masks in :doc:`ak.layout.ByteMaskedArray`, are not :doc:`ak.layout.NumpyArray` but a more constrained type called :doc:`ak.layout.Index`.

**Identities for layout nodes:** :doc:`ak.layout.Identities` are an optional surrogate key for certain join operations. (Not yet used.)

**High-level data types:**

This is the type of data in a high-level :doc:`_auto/ak.Array` or :doc:`_auto/ak.Record` as reported by :doc:`_auto/ak.type`. It represents as much information as a data analyst needs to know (e.g. the distinction between variable and fixed-length lists, but not the distinction between :doc:`ak.layout.ListArray` and :doc:`ak.layout.ListOffsetArray`).

   * :doc:`ak.types.Type`: the abstract base class.
   * :doc:`ak.types.ArrayType`: type of a non-composable, high-level :doc:`_auto/ak.Array`, which includes the length of the array.
   * :doc:`ak.types.UnknownType`: a type that is not known because it is represented by an :doc:`ak.layout.EmptyArray`.
   * :doc:`ak.types.PrimitiveType`: a numeric or boolean type.
   * :doc:`ak.types.RegularType`: lists of a fixed length; this ``size`` is part of the type description.
   * :doc:`ak.types.ListType`: lists of unspecified or variable length.
   * :doc:`ak.types.RecordType`: records with named fields or tuples with a fixed number of unnamed slots. The fields/slots and their types are part of the type description.
   * :doc:`ak.types.OptionType`: data that may be missing.
   * :doc:`ak.types.UnionType`: heterogeneous data selected from a short list of possibilities.

All concrete :doc:`ak.types.Type` subclasses are composable except :doc:`ak.types.ArrayType`.

**Low-level array forms:**

This is the type of a :doc:`ak.layout.Content` array expressed with low-level granularity (e.g. including the distinction between :doc:`ak.layout.ListArray` and :doc:`ak.layout.ListOffsetArray`). There is a one-to-one relationship between :doc:`ak.layout.Content` subclasses and :doc:`ak.forms.Form` subclasses, and each :doc:`ak.forms.Form` maps to only one :doc:`ak.types.Type`.

   * :doc:`ak.forms.Form`: the abstract base class.
   * :doc:`ak.forms.EmptyForm` for :doc:`ak.layout.EmptyArray`.
   * :doc:`ak.forms.NumpyForm` for :doc:`ak.layout.NumpyArray`.
   * :doc:`ak.forms.RegularForm` for :doc:`ak.layout.RegularArray`.
   * :doc:`ak.forms.ListForm` for :doc:`ak.layout.ListArray`.
   * :doc:`ak.forms.ListOffsetForm` for :doc:`ak.layout.ListOffsetArray`.
   * :doc:`ak.forms.RecordForm` for :doc:`ak.layout.RecordArray`.
   * :doc:`ak.forms.IndexedForm` for :doc:`ak.layout.IndexedArray`.
   * :doc:`ak.forms.IndexedOptionForm` for :doc:`ak.layout.IndexedOptionArray`.
   * :doc:`ak.forms.ByteMaskedForm` for :doc:`ak.layout.ByteMaskedArray`.
   * :doc:`ak.forms.BitMaskedForm` for :doc:`ak.layout.BitMaskedArray`.
   * :doc:`ak.forms.UnmaskedForm` for :doc:`ak.layout.UnmaskedArray`.
   * :doc:`ak.forms.UnionForm` for :doc:`ak.layout.UnionArray`.
   * :doc:`ak.forms.VirtualForm` for :doc:`ak.layout.VirtualArray`.

Internal implementation
"""""""""""""""""""""""

The rest of the classes and functions described here are not part of the public interface. Either the objects or the submodules begin with an underscore, indicating that they can freely change from one version to the next.

More documentation
""""""""""""""""""

The Awkward Array project is divided into 3 layers with 5 main components.

.. raw:: html

    <img src="_static/awkward-1-0-layers.svg" style="max-width: 500px; margin-left: auto; margin-right: auto;">

The C++ classes, cpu-kernels, and gpu-kernels are described in the `C++ API reference <_static/index.html>`__.

The kernels (cpu-kernels and cuda-kernels) are documented on the :doc:`_auto/kernels` page, with interfaces and normative Python implementations.

.. include:: _auto/toctree.txt
