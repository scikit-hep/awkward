.. |br| raw:: html

    <br/>

.. role:: raw-html(raw)
    :format: html

.. image:: https://github.com/scikit-hep/awkward-1.0/raw/main/docs-img/logo/logo-300px.png
    :width: 300px
    :alt: Awkward Array
    :target: https://github.com/scikit-hep/awkward-1.0

:raw-html:`<p>`

.. image:: https://badge.fury.io/py/awkward.svg
   :alt: PyPI version
   :target: https://pypi.org/project/awkward

.. image:: https://img.shields.io/conda/vn/conda-forge/awkward
   :alt: Conda-Forge
   :target: https://github.com/conda-forge/awkward-feedstock

.. image:: https://img.shields.io/badge/python-3.7%E2%80%923.10-blue
   :alt: Python 3.7‒3.10
   :target: https://www.python.org

.. image:: https://img.shields.io/badge/license-BSD%203--Clause-blue.svg
   :alt: BSD-3 Clause License
   :target: https://opensource.org/licenses/BSD-3-Clause

.. image:: https://github.com/scikit-hep/awkward/actions/workflows/build-test.yml/badge.svg?branch=main
   :alt: Continuous integration tests
   :target: https://github.com/scikit-hep/awkward/actions/workflows/build-test.yml

:raw-html:`</p><p>`

.. image:: https://scikit-hep.org/assets/images/Scikit--HEP-Project-blue.svg
   :alt: Scikit-HEP
   :target: https://scikit-hep.org/

.. image:: https://img.shields.io/badge/NSF-1836650-blue.svg
   :alt: NSF-1836650
   :target: https://nsf.gov/awardsearch/showAward?AWD_ID=1836650

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4341376.svg
   :alt: DOI
   :target: https://doi.org/10.5281/zenodo.4341376

.. image:: https://img.shields.io/badge/docs-online-success
   :alt: Documentation
   :target: https://awkward-array.readthedocs.io

.. image:: https://img.shields.io/badge/chat-online-success
   :alt: Gitter
   :target: https://gitter.im/Scikit-HEP/awkward-array

:raw-html:`</p>`

|br| Awkward Array is a library for **nested, variable-sized data**, including arbitrary-length lists, records, mixed types, and missing data, using **NumPy-like idioms**.

Arrays are **dynamically typed**, but operations on them are **compiled and fast**. Their behavior coincides with NumPy when array dimensions are regular and generalizes when they're not.

API Reference
*************

   * `Main page (quickstart) <https://awkward-array.readthedocs.io>`__
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

**Numba compatibility:** :doc:`ak.numba.register` informs Numba about Awkward Array types; rarely needed because this should happen automatically.

**Pandas compatibility:** :doc:`ak.to_pandas` turns an Awkward Array into a list of DataFrames or joins them with `pd.merge <https://pandas.pydata.org/pandas-docs/version/1.0.3/reference/api/pandas.merge.html>`__ if necessary.

**NumExpr compatibility:** :doc:`ak.numexpr.evaluate` and :doc:`ak.numexpr.re_evaluate` are like the NumExpr functions, but with Awkward Array support.

**Autograd compatibility:** :doc:`ak.autograd.elementwise_grad` is like the Autograd function, but with Awkward Array support.

**Layout nodes:** the high-level :doc:`_auto/ak.Array` and :doc:`_auto/ak.Record` types hide the tree-structure that build the array, but they can be accessed with `ak.Array.layout <_auto/ak.Array.html#ak-array-layout>`_. This layout structure is the core of the library, but usually doesn't have to be accessed by data analysts.

   * :class:`ak.contents.Content`: the abstract base class.
   * :class:`ak.contents.EmptyArray`: an array of unknown type with no elements (usually produced by :doc:`_auto/ak.ArrayBuilder`, which can't determine type at a given level without samples).
   * :class:`ak.contents.NumpyArray`: any NumPy array (e.g. multidimensional shape, arbitrary dtype), though usually only one-dimensional arrays of numbers.
   * :class:`ak.contents.RegularArray`: splits its nested content into equal-length lists.
   * :class:`ak.contents.ListArray`: splits its nested content into variable-length lists with full generality (may use its content non-contiguously, overlapping, or out-of-order).
   * :class:`ak.contents.ListOffsetArray`: splits its nested content into variable-length lists, assuming contiguous, non-overlapping, in-order content.
   * :class:`ak.contents.RecordArray`: represents a logical array of records with a "struct of arrays" layout in memory.
   * :class:`ak.record.Record`: represents a single record (not a subclass of :class:`ak.contents.Content` in Python).
   * :class:`ak.contents.IndexedArray`: rearranges and/or duplicates its content by lazily applying an integer index.
   * :class:`ak.contents.IndexedOptionArray`: same as :class:`ak.contents.IndexedArray` with missing values as negative indexes.
   * :class:`ak.contents.ByteMaskedArray`: represents its content with missing values with an 8-bit boolean mask.
   * :class:`ak.contents.BitMaskedArray`: represents its content with missing values with a 1-bit boolean mask.
   * :class:`ak.contents.UnmaskedArray`: specifies that its content can contain missing values in principle, but no mask is supplied because all elements are non-missing.
   * :class:`ak.contents.UnionArray`: interleaves a set of arrays as a tagged union, can represent heterogeneous data.

Most layout nodes contain another content node (:class:`ak.contents.RecordArray` and :class:`ak.contents.UnionArray` can contain more than one), thus forming a tree. Only :class:`ak.contents.EmptyArray` and :class:`ak.contents.NumpyArray` cannot contain a content, and hence these are leaves of the tree.

**Index for layout nodes:** integer and boolean arrays that define the shape of the data structure, such as boolean masks in :class:`ak.contents.ByteMaskedArray`, are not :class:`ak.contents.NumpyArray` but a more constrained type called :class:`ak.index.Index`.

**Identifiers for layout nodes:** :class:`ak.identifier.Identifier` is an optional surrogate key for certain join operations. (Not yet used.)

**High-level data types:**

This is the type of data in a high-level :doc:`_auto/ak.Array` or :doc:`_auto/ak.Record` as reported by :doc:`_auto/ak.type`. It represents as much information as a data analyst needs to know (e.g. the distinction between variable and fixed-length lists, but not the distinction between :class:`ak.contents.ListArray` and :class:`ak.contents.ListOffsetArray`).

   * :class:`ak.types.Type`: the abstract base class.
   * :class:`ak.types.ArrayType`: type of a non-composable, high-level :doc:`_auto/ak.Array`, which includes the length of the array.
   * :class:`ak.types.UnknownType`: a type that is not known because it is represented by an :class:`ak.contents.EmptyArray`.
   * :class:`ak.types.PrimitiveType`: a numeric or boolean type.
   * :class:`ak.types.RegularType`: lists of a fixed length; this ``size`` is part of the type description.
   * :class:`ak.types.ListType`: lists of unspecified or variable length.
   * :class:`ak.types.RecordType`: records with named fields or tuples with a fixed number of unnamed slots. The fields/slots and their types are part of the type description.
   * :class:`ak.types.OptionType`: data that may be missing.
   * :class:`ak.types.UnionType`: heterogeneous data selected from a short list of possibilities.

All concrete :class:`ak.types.Type` subclasses are composable except :class:`ak.types.ArrayType`.

**Low-level array forms:**

This is the type of a :class:`ak.contents.Content` array expressed with low-level granularity (e.g. including the distinction between :class:`ak.contents.ListArray` and :class:`ak.contents.ListOffsetArray`). There is a one-to-one relationship between :class:`ak.contents.Content` subclasses and :class:`ak.forms.Form` subclasses, and each :class:`ak.forms.Form` maps to only one :class:`ak.types.Type`.

   * :class:`ak.forms.Form`: the abstract base class.
   * :class:`ak.forms.EmptyForm` for :class:`ak.contents.EmptyArray`.
   * :class:`ak.forms.NumpyForm` for :class:`ak.contents.NumpyArray`.
   * :class:`ak.forms.RegularForm` for :class:`ak.contents.RegularArray`.
   * :class:`ak.forms.ListForm` for :class:`ak.contents.ListArray`.
   * :class:`ak.forms.ListOffsetForm` for :class:`ak.contents.ListOffsetArray`.
   * :class:`ak.forms.RecordForm` for :class:`ak.contents.RecordArray`.
   * :class:`ak.forms.IndexedForm` for :class:`ak.contents.IndexedArray`.
   * :class:`ak.forms.IndexedOptionForm` for :class:`ak.contents.IndexedOptionArray`.
   * :class:`ak.forms.ByteMaskedForm` for :class:`ak.contents.ByteMaskedArray`.
   * :class:`ak.forms.BitMaskedForm` for :class:`ak.contents.BitMaskedArray`.
   * :class:`ak.forms.UnmaskedForm` for :class:`ak.contents.UnmaskedArray`.
   * :class:`ak.forms.UnionForm` for :class:`ak.contents.UnionArray`.

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
