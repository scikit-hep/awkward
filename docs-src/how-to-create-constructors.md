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

```{code-cell} ipython3
import awkward as ak

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

```{code-cell} ipython3
array.tolist()
```

gets all of the items in the first list separately from the items in the second or third lists, and both fields of each record (_x_ and _y_) are expressed near each other in the flow of the code and in the times when they're appended.

By contrast, the physical data are laid out in columns, with all _x_ values next to each other, regardless of which records or lists they're in, and all the _y_ values next to each other in another buffer.

```{code-cell} ipython3
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

EmptyArray
----------

[ak.layout.EmptyArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.EmptyArray.html)

+++

NumpyArray
----------

[ak.layout.NumpyArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.NumpyArray.html)

+++

RegularArray
------------

[ak.layout.RegularArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.RegularArray.html)

+++

ListArray
---------

[ak.layout.ListArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.ListArray.html)

+++

ListOffsetArray
---------------

[ak.layout.ListOffsetArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.ListOffsetArray.html)

+++

RecordArray and Record
----------------------

[ak.layout.RecordArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.RecordArray.html)

+++

IndexedArray
------------

[ak.layout.IndexedArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.IndexedArray.html)

+++

IndexedOptionArray
------------------

[ak.layout.IndexedOptionArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.IndexedOptionArray.html)

+++

ByteMaskedArray
---------------

[ak.layout.ByteMaskedArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.ByteMaskedArray.html)

+++

BitMaskedArray
--------------

[ak.layout.BitMaskedArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.BitMaskedArray.html)

+++

UnmaskedArray
-------------

[ak.layout.UnmaskedArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.UnmaskedArray.html)

+++

UnionArray
----------

[ak.layout.UnionArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.UnionArray.html)

+++

VirtualArray
------------

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

```{code-cell} ipython3
array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
array.layout
```

```{code-cell} ipython3
# Abbreviated JSON representation
array.layout.form
```

```{code-cell} ipython3
# Full JSON representation
print(array.layout.form.tojson(pretty=True))
```

In this way, you can figure out how to generate Forms corresponding to the Content nodes you want [ak.from_buffers](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_buffers.html) to make.
