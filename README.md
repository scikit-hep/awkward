# awkward-1.0

Development of Awkward 1.0, to replace [scikit-hep/awkward-array](https://github.com/scikit-hep/awkward-array) in 2020.

   * The [original motivations document](https://docs.google.com/document/d/1lj8ARTKV1_hqGTh0W_f01S6SsmpzZAXz9qqqWnEB3j4/edit?usp=sharing), now out-of-date.
   * My [PyHEP talk](https://indico.cern.ch/event/833895/contributions/3577882) on October 17, 2019.
   * My [CHEP talk](https://indico.cern.ch/event/773049/contributions/3473258) on November 7, 2019.

## Motivation for a new Awkward Array

Awkward Array has proven to be a useful way to analyze variable-length and tree-like data in Python, by extending Numpy's idioms from flat arrays to arrays of data structures. For over a year, physicists have been using Awkward Array both in and out of uproot; it is already one of the most popular Python packages in particle physics.

<p align="center"><img src="docs/img/awkward-0-popularity.png" width="60%"></p>

However, the pure-NumPy implementation is hard to extend (finding for-loop-free implementations of operations on nested data is _hard_) and maintain (most bugs are NumPy corner cases). Also, the feedback users have given me through [GitHub](https://github.com/scikit-hep/awkward-array/issues), [StackOverflow](https://stackoverflow.com/questions/tagged/awkward-array), and [in-person tutorials](https://github.com/jpivarski/2019-07-29-dpf-python#readme) have pointed out some design mistakes. A backward-incompatible release will allow us to fix design mistakes while providing freedom to make deep changes in the implementation.

The Awkward 1.0 project is a major investment, a six-month sprint from late August 2019 to late February 2020. The time spent on a clean, robust Awkward Array is justified by the widespread adoption of Awkward 0.x: its usefulness to the community has been demonstrated.

## Main goals of Awkward 1.0

   * Full access to create and manipulate Awkward Arrays in C++ with no Python dependencies. This is so that C++ libraries can produce and share data with Python front-ends.
   * Easy installation with `pip install` and `conda install` for most users (Mac, Windows, and [most Linux](https://github.com/pypa/manylinux)).
   * Imperative (for-loop-style) access to Awkward Arrays in [Numba](https://numba.pydata.org/), a just-in-time compiler for Python. This is so that physicists can write critical loops in straightforward Python without a performance penalty.
   * A single `awkward.Array` class that hides the details of how columnar data is built, with a suite of operations that apply to all internal types.
   * Conformance to NumPy, where Awkward and NumPy overlap.
   * Better control over "behavioral mix-ins," such as `LorentzVector` (i.e. adding methods like `pt()` to arrays of records with `px` and `py` fields). In Awkward 0.x, this was achieved with multiple inheritance, but that was brittle.
   * Support for set operations and database-style joins, which can be put to use in a [declarative analysis language](https://github.com/jpivarski/PartiQL#readme), but requires database-style accounting of an index (like a Pandas index).
   * Better interoperability with Pandas, NumExpr, and Dask, while maintaining interoperability with ROOT, Arrow, and Parquet.
   * Ability to add GPU implementations of array operations in the future.
   * Better error messages.

## Architecture of Awkward 1.0






## Old

Awkward-array 1.0 will be rewritten in C++ using simple, single-pass algorithms, and exposed to Python through pybind11 and Numba. It will consist of four layers:

   1. A single `Array` class in Python that can be lowered in Numba, representing a sequence of abstract objects according to a high-level [datashape](https://datashape.readthedocs.io/en/latest/).
   2. Its columnar implementation in terms of nested `ListArray`, `RecordArray`, `MaskedArray`, etc., exposed to Python through pybind11. The modularity of this "layout" is very useful for engineering, but it has some features that have tripped up users, such as different physical arrays representing the same logical objects.
   3. Memory management of this layout in C++ classes, and again as [Numba extensions](https://numba.pydata.org/numba-doc/dev/extending/index.html). As a side-effect, `libawkward.so` would be a library for creating and manipulating awkward-arrays purely in C++, with no pybind11 dependencies.
   4. Implementations of the actual algorithms for (a) CPUs and (b) GPUs. This layer performs no memory management (arrays must be allocated and owned by the caller), but they perform all loops over data, reading old arrays, filling new arrays.

Below is a diagram of how each component uses those at a lower level of abstraction:

<p align="center"><img src="docs/img/awkward-1-0-layers.png" width="60%"></p>

Layer 4 is a dynamically linked library named `{lib,}awkward-cpu-kernels.{so,dylib,dll}` with an unmangled C FFI interface. The layer 3 C++ library is `{lib,}awkward.{so,dylib,dll}` and layer 3 Numba extensions are in the `numbaext` submodule of the `awkward1` Python library. Layer 2 is the `layout` submodule, linked via pybind11, and layer 1 is the Python code in `awkward1`.

The original `awkward` library will continue to be maintained while `awkward1` is in development, and the two will be swapped, becoming `awkward0` and `awkward`, respectively, in a gradual deprecation process in 2020. [Uproot 3.x](https://github.com/scikit-hep/uproot) will continue to depend on `awkward0`, but uproot 4.x will make the new `awkward` an optional but highly recommended dependency. Many users are only directly aware of uproot, so a major change in version from 3 to 4 will alert them to changes in interface.

## What will not change

The following features of awkward 0.x will be features of awkward 1.x.

   * The efficient, columnar data representation built out of "layout" classes, such as `ListArray` (`JaggedArray`), `RecordArray` (`Table`), and `MaskedArray` (same name). This was the key idea in the transition from OAMap to awkward-array, and it has paid off. If anything, we will be increasing the composability, with a larger number of classes that take on smaller roles (e.g. separate `ListArray` from `ListOffsetArray` for the starts/stops vs offsets cases).
   * Numpy-like interface: awkward 1.x will be more consistent with Numpy's API, not less.
   * Interoperability with ROOT, Arrow, Parquet and Pandas, as well as planned interoperability with Numba and Dask.
   * The goals of zero-copy and shallow manipulations, such as slicing an array of objects without necessarily loading all of their attributes.
   * The ability to attach domain-specific methods to arrays, such as Lorentz transformations for arrays of Lorentz vectors.
   * `VirtualArrays`, and therefore lazy-loading, will be supported.

## What will change

   * Awkward 0.x's single specification, many implementations model: awkward 1.x will have at most two implementations of each algorithm, one for CPUs and one for GPUs. All the goals of precompiled kernels, Numba interface, Pandas interoperability, etc. will be accomplished through the four-layer system described above instead of completely separate implementations. Each of the four layers will be fully specified by documentation, though.
   * Native access to the layout in C++ and the ability to develop an implementation in any language that supports C FFI (i.e. all of them).
   * Data analysts will see a single `Array` class that hides details about the layout. This is an API-breaking change for data analysis scripts, but one that would help new users.
   * All manipulations of this `Array` class will be through functions in the `awkward` namespace, such as `awkward.cross(a, b)` instead of `a.cross(b)` to mean a cross-join of `a` and `b` per element. The namespace on the arrays is therefore free for data fields like `a.x`, `a.y`, `a.z` and domain-specific methods like `a.cross(b)` meaning 3-D cross product. This is an API-breaking change for data analysis scripts, but one that would help new users.
   * Arrays will pass through optional Pandas-style indexes for high-level operations [like these](https://github.com/jpivarski/PartiQL#readme).

## Roadmap

The rough estimate for development time to a minimally usable library for physics was six months, starting in late August (i.e. finishing in late February). **Progress is currently on track.**

### Approximate order of implementation

Completed items are ☑check-marked. See [closed PRs](https://github.com/scikit-hep/awkward-1.0/pulls?q=is%3Apr+is%3Aclosed) for more details.

   * [X] Cross-platform, cross-Python version build and deploy process. Regularly deploying [30 wheels](https://pypi.org/project/awkward1/#files) after closing each PR.
   * [X] Basic `NumpyArray`, `ListArray`, and `ListOffsetArray` with `__getitem__` for int/slice and `__iter__` in C++/pybind11 to establish structure and ensure proper reference counting.
   * [X] Introduce `Identity` as a Pandas-style index to pass through `__getitem__`.
   * [X] Reproduce all of the above as Numba extensions (make `NumpyArray`, `ListArray`, and `ListOffsetArray` usable in Numba-compiled functions).
   * [X] Error messages with location-of-failure information if the array has an `Identity` (except in Numba).
   * [X] Fully implement `__getitem__` for int/slice/intarray/boolarray/tuple (placeholders for newaxis/ellipsis), with perfect agreement with [Numpy basic/advanced indexing](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html), to all levels of depth.
   * [X] Appendable arrays (a distinct phase from readable arrays, when the type is still in flux) to implement `awkward.fromiter` in C++.
      * [X] Implemented all types but records; tested all primitives and lists.
      * [X] Expose appendable arrays to Numba.
      * [X] Implement appendable records.
      * [X] Test all (tested in mock [studies/fillable.py](tree/master/studies/fillable.py)).
   * [X] JSON → Awkward via header-only [RapidJSON](https://rapidjson.org) and `awkward.fromiter`.
   * [ ] Explicit broadcasting functions for jagged and non-jagged arrays and scalars.
   * [ ] Extend `__getitem__` to take jagged arrays of integers and booleans (same behavior as old).
   * [ ] Full suite of array types:
      * [X] `EmptyArray`: 1-dimensional array with length 0 and unknown type (result of `UnknownFillable`, compatible with all types of arrays).
      * [X] `RawArray`: flat, 1-dimensional array type for pure C++ (header-only).
      * [X] `NumpyArray`: rectilinear, N-dimensional array type without Python/pybind11 dependencies, but intended for Numpy.
      * [X] `ListArray`: the new `JaggedArray`, based on `starts` and `stops` (i.e. fully general).
      * [X] `ListOffsetArray`: the `JaggedArray` case with no unreachable data between reachable data (gaps).
      * [X] `RegularArray`: for building rectilinear, N-dimensional arrays of arbitrary contents, e.g. putting jagged dimensions inside fixed dimensions.
      * [X] `RecordArray`: the new `Table` _without_ lazy-slicing.
         * [X] Implement it in Numba as well.
      * [ ] `MaskedArray`, `BitMaskedArray`, `IndexedMaskedArray`: same as the old versions.
      * [ ] `UnionArray`: same as the old version; `SparseUnionArray`: the additional case found in Apache Arrow.
      * [ ] `IndexedArray`: same as the old version.
      * [ ] `RedirectArray`: an explicit weak-reference to another part of the structure (no hard-linked cycles). Often used with an `IndexedArray`.
      * [ ] `SlicedArray`: lazy-slicing (from old `Table`) that can be applied to any type.
      * [ ] `SparseArray`: same as the old version.
      * [ ] `ChunkedArray`: same as the old version, except that the type is a union if chunks conflict, not an error, and knowledge of all chunk sizes is always required. (Maybe `AmorphousChunkedArray` would fill that role.)
      * [ ] `RegularChunkedArray`: like a `ChunkedArray`, but all chunks are known to have the same size.
      * [ ] `VirtualArray`: same as the old version, including caching, but taking C++11 lambda functions for materialization, get-cache, and put-cache. The pybind11 layer will connect this to Python callables.
      * [ ] `PyVirtualArray`: takes a Python lambda (which gets carried into `VirtualArray`).
      * [ ] `PyObjectArray`: same as the old version.
   * [X] Describe high-level types using [datashape](https://datashape.readthedocs.io/en/latest/) and possibly also an in-house schema. (Emit datashape _strings_ from C++.)
   * [ ] Describe mid-level "persistence types" with no lengths, somewhat minimal JSON, optional dtypes/compression.
   * [ ] Describe low-level layouts independently of filled arrays (JSON or something)?
   * [X] Layer 1 interface `Array`:
      * [ ] Pass through to the layout classes in Python and Numba.
      * [ ] Pass through Numpy ufuncs using [NEP 13](https://www.numpy.org/neps/nep-0013-ufunc-overrides.html) (as before).
      * [ ] Pass through other Numpy functions using [NEP 18](https://www.numpy.org/neps/nep-0018-array-function-protocol.html) (this would be new).
      * [ ] `RecordArray` fields (not called "columns" anymore) through Layer 1 `__getattr__`.
      * [ ] Special Layer 1 `Record` type for `RecordArray` elements, supporting some methods and a visual representation based on `Identity` if available, all fields if `recordtype == "tuple"`, or the first field otherwise.
      * [X] Mechanism for adding user-defined `Methods` like `LorentzVector`, as before, but only on Layer 1.
         * [X] High-level classes for characters and strings.
      * [ ] Inerhit from Pandas so that all Layer 1 arrays can be DataFrame columns.
   * [ ] Full suite of operations:
      * [X] `awkward.tolist`: same as before.
      * [X] `awkward.fromiter`: same as before.
      * [X] `awkward.typeof`: reports the high-level type (accepting some non-awkward objects).
      * [ ] `awkward.tonumpy`: to force conversion to Numpy, if possible. Neither Layer 1 nor Layer 2 will have an `__array__` method; in the Numpy sense, they are not "array-like" or "array-compatible."
      * [ ] `awkward.topandas`: flattening jaggedness into `MultiIndex` rows and nested records into `MultiIndex` columns. This is distinct from the arrays' inheritance from Pandas, distinct from the natural ability to use any one of them as DataFrame columns.
      * [ ] `awkward.flatten`: same as old with an `axis` parameter.
      * [ ] Reducers, such as `awkward.sum`, `awkward.max`, etc., supporting an `axis` method.
      * [ ] The non-reducers: `awkward.moment`, `awkward.mean`, `awkward.var`, `awkward.std`.
      * [ ] `awkward.argmin`, `awkward.argmax`, `awkward.argsort`, and `awkward.sort`: same as old.
      * [ ] `awkward.where`: like `numpy.where`; old doesn't have this yet, but we'll need it.
      * [ ] `awkward.concatenate`: same as old, but supporting `axis` at any depth.
      * [ ] `awkward.zip`: makes jagged tables; this is a naive version of `awkward.join` below.
      * [ ] `awkward.pad`: same as old, but without the `clip` option (use slicing instead).
      * [ ] `awkward.fillna`: same as old.
      * [ ] `awkward.cross` (and `awkward.argcross`): to make combinations by cross-joining multiple arrays; option to use `Identity` index.
      * [ ] `awkward.choose` (and `awkward.argchoose`): to make combinations by choosing a fixed number from a single array; option to use `Identity` index and an option to include same-object combinations.
      * [ ] `awkward.join`: performs an inner join of multiple arrays; requires `Identity`. Because the `Identity` is a surrogate index, this is effectively a per-event intersection, zipping all fields.
      * [ ] `awkward.union`: performs an outer join of multiple arrays; requires `Identity`. Because the `Identity` is a surrogate index, this is effectively a per-event union, zipping fields where possible.

### Soon after (possibly within) the six-month timeframe

   * [ ] Update [hepvector](https://github.com/henryiii/hepvector#readme) to be Derived classes, replacing the `TLorentzVectorArray` in uproot-methods.
   * [ ] Update uproot (on a branch) to use Awkward 1.0.
   * [ ] Start the `awkward → awkward0`, `awkward1 → awkward` transition.
   * [ ] Translation to and from Apache Arrow and Parquet in C++.
   * [ ] Persistence to any medium that stores named binary blobs, as before, but accessible via C++ (especially for writing). The persistence format might differ slightly from the existing one (break backward compatibility, if needed).
   * [ ] Universal `array.get[...]` as a softer form of `array[...]` that inserts `None` for non-existent indexes, rather than raising errors.
   * [ ] Explicit interface with [NumExpr](https://numexpr.readthedocs.io/en/latest/index.html).

### At some point in the future

   * [ ] Demonstrate Awkward 1.0 as a C++ wrapping library with [FastJet](http://fastjet.fr/).
   * [ ] GPU implementations of the cpu-kernels in Layer 4, with the Layer 3 C++ passing a "device" variable at every level of the layout to indicate whether the data pointers refer to main memory or a particular GPU.
   * [ ] CPU-acceleration of the cpu-kernels using vectorization and other tricks.
   * [ ] Explicit interface with [Dask](https://dask.org).
