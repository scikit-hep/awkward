[![](https://raw.githubusercontent.com/scikit-hep/awkward-1.0/main/docs-img/logo/logo-300px.png)](https://github.com/scikit-hep/awkward-1.0)

Awkward Array is a library for **nested, variable-sized data**, including arbitrary-length lists, records, mixed types, and missing data, using **NumPy-like idioms**.

Arrays are **dynamically typed**, but operations on them are **compiled and fast**. Their behavior coincides with NumPy when array dimensions are regular and generalizes when they’re not.

## Documentation

   * C++ API reference: **this site**
   * [Main reference](https://awkward-array.readthedocs.io/en/latest/)
   * [GitHub repository](https://github.com/scikit-hep/awkward-1.0)

## Navigation

The Awkward Array project is divided into three layers:

   * **High-level** array objects and operations (Python)
   * **Low-level** layout objects (Python, C++)
   * **Primitive** kernel operations (C++, CuPy, JAX)

This reference describes the

   * **C++ classes:** in [namespace awkward](namespaceawkward.html) (often abbreviated as "ak"), which are compiled into **libawkward.so** (or dylib or lib).
   * **pybind11 interface:** no namespace, but contained entirely within the [python directory](dir_91f33a3f1dd6262845ebd1570075970c.html), which are compiled into **awkward._ext** for use in Python.
   * **CPU kernels:** no namespace, but contained entirely within the [kernels directory](dir_6225843069e7cc68401bbec110a1667f.html), which are compiled into **libawkward-cpu-kernels.so** (or dylib or lib). This library is fully usable from any language that can call functions through [FFI](https://en.wikipedia.org/wiki/Foreign_function_interface).
   * **GPU kernels:** FIXME! (not implemented yet)

### ArrayBuilder structure

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

The [ak::Builder](classawkward_1_1Builder.html) instances contain arrays of accumulated data, and thus store both data (in these arrays) and type (in their tree structure). The hierarchy is not exactly the same as `ak.Content`/`ak.Form` (which are identical to each other) or `ak.Type`, since it reflects the kinds of data to be encountered in the input data: mostly JSON-like, but with a distinction between records with named fields and tuples with unnamed slots.

   * [ak::Builder](classawkward_1_1Builder.html): the abstract base class.
   * [ak::UnknownBuilder](classawkward_1_1UnknownBuilder.html): the initial builder; a builder for unknown type; generates an `ak.forms.EmptyForm` (which is why we have that class).
   * [ak::BoolBuilder](classawkward_1_1BoolBuilder.html): boolean type; generates a `ak.forms.NumpyForm` and associated data buffers.
   * [ak::Int64Builder](classawkward_1_1Int64Builder.html): 64-bit integer type; generates a `ak.forms.NumpyForm` and associated data buffers. Appending integer data to boolean data generates a union; it does not promote the booleans to integers.
   * [ak::Float64Builder](classawkward_1_1Float64Builder.html): 64-bit floating point type; generates a `ak.forms.NumpyForm` and associated data buffers. Appending floating-point data to integer data does not generate a union; it promotes integers to floating-point.
   * [ak::StringBuilder](classawkward_1_1StringBuilder.html): UTF-8 encoded string or raw bytestring type; generates a `ak.forms.ListOfsetForm` and associated data buffers, with parameter `"__array__"` equal to `"string"` or `"bytestring"`.
   * [ak::ListBuilder](classawkward_1_1ListBuilder.html): list type; generates a `ak.forms.ListOffsetForm` and associated data buffers.
   * [ak::OptionBuilder](classawkward_1_1OptionBuilder.html): option type; generates an `ak.forms.IndexedOptionForm` and associated data buffers.
   * [ak::RecordBuilder](classawkward_1_1RecordBuilder.html): record type with field names; generates a `ak.forms.RecordForm` and associated data buffers with a non-null `fields`.
   * [ak::TupleBuilder](classawkward_1_1TupleBuilder.html): tuple type without field names; generates a `ak.forms.RecordForm` and associated data buffers with null `fields`.
   * [ak::UnionBuilder](classawkward_1_1UnionBuilder.html): union type; generates a `ak.forms.UnionForm` and associated data buffers.
   * [ak::IndexedBuilder](classawkward_1_1IndexedBuilder.html): indexed type; generates an `ak.forms.IndexedForm` and associated data buffers; inserts an existing array node into the new array under construction, referencing its elements with an `ak.forms.IndexedForm`. This way, complex structures can be included by reference, rather than by copying.

**Options for building an array:** [ak::ArrayBuilderOptions](classawkward_1_1ArrayBuilderOptions.html) are passed to every [ak::Builder](classawkward_1_1Builder.html) in the tree.

**Buffers for building an array:** [ak::GrowableBuffer<T>](classawkward_1_1GrowableBuffer.html) is a one-dimensional array with append-only semantics, used both for buffers that will become [ak::IndexOf<T>](classawkward_1_1IndexOf.html) and buffers that will become [ak::NumpyArray](classawkward_1_1NumpyArray.html). It works like `std::vector` in that it replaces its underlying storage at logarithmically frequent intervals, but unlike a `std::vector` in that the underlying storage is a `std::shared_ptr<void*>`.

Upon calling the [to_buffers](classawkward_1_1Builder.html#a24ef6967f0648462def1c03e00e8a610) method of [`ak::Builder`](classawkward_1_1Builder.html), the potentially non-contiguous data are copied into the given [`ak::BuffersContainer`](classawkward_1_1BuffersContainer.html). This ensures that the Python runtime manages the memory associated with the builder.

Array building is not as efficient as computing with pre-built arrays because the type-discovery makes each access a tree-descent. Additionally, the [ak::Builder](classawkward_1_1Builder.html) instances append to their [ak::GrowableBuffer<T>](classawkward_1_1GrowableBuffer.html), which are assumed to exist in main memory, not a GPU.


### Conversion from JSON

The [`ak::fromjsonobject`](namespaceawkward.html#a8f042641c01a0ec3206b2f169d3a396b) and [`ak::FromJsonObjectSchema`](classawkward_1_1FromJsonObjectSchema.html) symbols are used to facilitate fast construction of Awkward Arrays from JSON data using [ak::ArrayBuilder](classawkward_1_1ArrayBuilder.html).

### CPU kernels and GPU kernels

The kernels library is separated from the C++ codebase by a pure C interface, and thus the kernels could be used by other languages.

The GPU kernels follow exactly the same interface, though a different implementation.

**FIXME:** kernel function names and arguments should be systematized in some searchable way.
