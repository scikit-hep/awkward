[![](https://raw.githubusercontent.com/scikit-hep/awkward/main/docs-img/logo/logo-300px.png)](https://github.com/scikit-hep/awkward)

Awkward Array is a library for **nested, variable-sized data**, including arbitrary-length lists, records, mixed types, and missing data, using **NumPy-like idioms**.

Arrays are **dynamically typed**, but operations on them are **compiled and fast**. Their behavior coincides with NumPy when array dimensions are regular and generalizes when they’re not.

The primary documentation can be found at [awkward-array.org](https://awkward-array.org/). This sub-site documents the C++ classes in the `awkward-cpp` library, which `awkward` uses to accelerate some key algorithms.

The main parts of the C++ codebase are:

   * `awkward-cpp/src/cpu-kernels`: processes that iterate over all elements in an array buffer. Any Awkward Array algorithms that can't be expressed in terms of NumPy/CuPy/etc. array manipulations are implemented as Awkward kernels. Individual kernels are [specified in YAML](https://github.com/scikit-hep/awkward/blob/main/kernel-specification.yml) and [documented here](../../reference/generated/kernels.html).
   * `awkward-cpp/src/libawkward`: other algorithms that must be implemented in C++. Currently, these are the [ArrayBuilder](../../reference/generated/ak.ArrayBuilder.html) and [AwkwardForth language](../../reference/awkwardforth.html).
   * `awkward-cpp/src/python`: pybind11 bindings for `libawkward` classes and functions.
   * `header-only/awkward`: header-only files intended for inclusion in other C++ projects that create Awkward Arrays. See the [code in GitHub](https://github.com/scikit-hep/awkward/tree/main/header-only/awkward).
