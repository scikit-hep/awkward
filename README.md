# awkward-1.0

Development of awkward 1.0, to replace [scikit-hep/awkward-array](https://github.com/scikit-hep/awkward-array) in 2020.

   * [Motivation and requirements](https://docs.google.com/document/d/1lj8ARTKV1_hqGTh0W_f01S6SsmpzZAXz9qqqWnEB3j4/edit?usp=sharing) as a Google Doc (for comments).

## Short summary

Awkward-array has proven to be a useful way to analyze variable-length and tree-like data in Python, by extending Numpy's idioms from flat arrays to arrays of data structures. Unlike previous iterations of this idea ([Femtocode](https://github.com/diana-hep/femtocode), [OAMap](https://github.com/diana-hep/oamap)), awkward-array is used in data analysis, with feedback from users.

The original awkward-array is written in pure Python + Numpy, and it incorporates some clever tricks to do nested structure manipulations with only Numpy calls (e.g. Jaydeep Nandi's [cross-join](https://gitlab.com/Jayd_1234/GSoC_vectorized_proof_of_concepts/blob/master/argproduct.md) and [self-join without replacement](https://gitlab.com/Jayd_1234/GSoC_vectorized_proof_of_concepts/blob/master/argpairs.md), Nick Smith's [N choose k for k < 5](https://github.com/scikit-hep/awkward-array/pull/102), Jonas Rembser's [JaggedArray concatenation for axis=1](https://github.com/scikit-hep/awkward-array/pull/117), and my [deep JaggedArray get-item](https://github.com/scikit-hep/awkward-array/blob/423ca484e17fb0d0b3938e2ec0a0dcd8ef26c735/awkward/array/jagged.py#L597-L775)). However, cleverness is hard to scale: JaggedArray get-item raises `NotImplementedError` for some triply nested or deeper cases. Also, most bugs reported by users have been related to Numpy special cases, such as Numpy's raising an error for the min or max of an empty array. And finally, these implementations make multiple passes over the data, whereas a straightforward implementation (where we're allowed to use for loops, because it's compiled code) would be single-pass and therefore more cache-friendly.

Awkward-array 1.0 will be rewritten in C++ using simple, single-pass algorithms, and exposed to Python through pybind11 and Numba. It will consist of four layers:

   1. A single `Array` class in Python that can be lowered in Numba, representing a sequence of abstract objects according to a high-level [datashape](https://datashape.readthedocs.io/en/latest/).
   2. Its columnar implementation in terms of nested `JaggedArray`, `Table`, `MaskedArray`, etc., exposed to Python through pybind11. The modularity of this "layout" is very useful for engineering, but it has some features that have tripped up users, such as different physical arrays representing the same logical objects.
   3. Memory management of this layout in C++ classes, and again as [Numba extensions](https://numba.pydata.org/numba-doc/dev/extending/index.html). As a side-effect, `libawkward.so` would be a library for creating and manipulating awkward-arrays purely in C++, with no pybind11 dependencies.
   4. Implementations of the actual algorithms for (a) CPUs and (b) GPUs. This layer performs no memory management (arrays must be allocated and owned by the caller), but they perform all loops over data, reading old arrays, filling new arrays.

Below is a diagram of how each component uses those at a lower level of abstraction:

<p align="center"><img src="docs/img/awkward-1-0-layers.png" width="60%"></p>

Level 4 is a dynamically linked library named `{lib,}awkward-cpu-kernels.{so,dylib,dll}` with an unmangled C FFI interface. The level 3 C++ library is `{lib,}awkward.{so,dylib,dll}` and layer 3 Numba extensions are in the `numbaext` submodule of the `awkward1` Python library. Level 2 is the `layout` submodule, linked via pybind11, and level 1 is the Python code in `awkward1`.

The original `awkward` library will continue to be maintained while `awkward1` is in development, and the two will be swapped, becoming `awkward0` and `awkward`, respectively, in a gradual deprecation process in 2020. [Uproot 3.x](https://github.com/scikit-hep/uproot) will continue to depend on `awkward0`, but uproot 4.x will make the new `awkward` an optional but highly recommended dependency. Many users are only directly aware of uproot, so a major change in version from 3 to 4 will alert them to changes in interface.

## What will not change

The following features of awkward 0.x will be features of awkward 1.x.

   * The efficient, columnar data representation built out of "layout" classes, such as `JaggedArray`, `Table`, and `MaskedArray`. This was the key idea in the transition from OAMap to awkward-array, and it has paid off. If anything, we will be increasing the composability, with a larger number of classes that take on smaller roles (e.g. separate the lazy-slicing of `Table` into its own layout class).
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
   * Arrays will be efficiently appendable at all levels by allocating in blocks, and some or all layouts will support in-place assignment.
   * Arrays will pass through optional indexes for high-level operations [described here](https://github.com/jpivarski/PartiQL#readme).

## Status

   * 2019-08-17: set up a build process for the four layers with continuous deployment to Linux, MacOS, and Windows wheels.
   * 2019-08-22 (PR [#2](../../pull/2)): created a basic `NumpyArray` and `ListOffsetArray` in C++, exposed to Python with pybind11, and ensured correct memory management between Python's reference counts and C++'s `std::shared_ptr`.
   * 2019-08-26 (PR [#3](../../pull/3)): extended Numba so that `NumpyArray` and `ListOffsetArray` can be used in Numba-compiled functions, ensuring no memory leaks/double frees.
   * 2019-08-27 (PR [#4](../../pull/4)): introduced `Identity`, an optional surrogate key whose use is illustrated in [PartiQL](https://github.com/jpivarski/PartiQL#readme).
   * 2019-08-29 (PR [#5](../../pull/5)): extended Numba to use `Identity` as well, ensuring no memory leaks/double frees.
   * 2019-08-30 (PR [#6](../../pull/6)): added iteration to both C++ and Numba, as well as the first "operation," `awkward1.tolist`, which turns an awkward array into Python lists (and eventually dicts, etc.).
   * 2019-09-02 (PR [#7](../../pull/7)): refactored `Index`, `Identity`, and `ListOffsetArray` (and any other array types with `Index`, which is nearly all of them) to have a 32-bit and a 64-bit version. My original plan to only support 64-bit in "chunked arrays" with 32-bit everywhere else is hereby scrapped—both bit widths will be supported on all indexes. Non-native endian, non-trivial strides, and multidimensional `Index`/`Identity` are not supported, though all of these features are allowed for `NumpyArray` (which is _content_, not an _index_). The only limitation on `NumpyArray` is that data must be C-ordered, not Fortran-ordered.
   * 2019-09-21 (PR [#8](../../pull/8)): C++ NumpyArray::getitem is done, setting the pattern for other classes (external C functions). The Numba and Identity extensions are not done, which would be necessary to fully set the pattern. This involved a lot of investigation (see [studies/getitem.py](https://github.com/jpivarski/awkward-1.0/blob/master/studies/getitem.py)).

## Roadmap

**TODO.** Rough estimate: it will be in a testable state later this year, possibly the beginning of October, with the `awkward/awkward1` → `awkward0/awkward` transition early in 2020.
