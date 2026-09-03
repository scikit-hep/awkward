# Getting started

## Installation

:::::{card} If you use pip, rip, pixi, or uv to install packages,

```bash
pip install awkward
```
:::::
:::::{card} If you use conda or mamba, it's in the conda-forge channel.

```bash
conda install -c conda-forge awkward
```
:::::

If you're installing as a developer or testing updates that haven't been released in a package manager yet, see the [developer installation instructions](https://github.com/scikit-hep/awkward/blob/main/CONTRIBUTING.md#building-and-testing-locally) in the [Contributor guide](https://github.com/scikit-hep/awkward/blob/main/CONTRIBUTING.md).

## Tutorials

See the left side-bar (or bring it into view by clicking on the upper-left `≡`) for tutorials that illustrate the purpose and main concepts behind Awkward Arrays.

## Frequently asked questions

You can test any examples in a new window/tab by clicking on [![Try It! ⭷](https://img.shields.io/badge/-Try%20It%21%20%E2%86%97-orange?style=for-the-badge)](https://awkward-array.org/doc/main/_static/try-it.html).

::::::::{grid} 1

:::::::{grid-item} 
::::::{dropdown} What is Awkward Array for? How does it compare to other libraries?

Python's builtin lists, dicts, and classes can be used to analyze arbitrary data structures, but at a cost in speed and memory. Therefore, they can't be used (easily) with large datasets.

[Pandas](https://pandas.pydata.org/) DataFrames (as well as [Polars](https://pola.rs/), [cuDF](https://docs.rapids.ai/api/cudf/stable/), and [Dask DataFrame](https://docs.dask.org/en/stable/dataframe.html)) are well-suited to tabular data, including tables with relational indexes, but not arbitrary data structures. If a DataFrame is filled with Python's builtin types, then it offers no speed or memory advantage over Python itself.

[NumPy](https://numpy.org/) is ideal for rectangular arrays of numbers, but not arbitrary data structures. If a NumPy array is filled with Python's builtin types, then it offers no speed or memory advantage over Python itself.

[Apache Arrow](https://arrow.apache.org/) ([pyarrow](https://arrow.apache.org/docs/python/)) manages arrays of arbitrary data structures (including those in [Polars](https://pola.rs/), [cuDF](https://docs.rapids.ai/api/cudf/stable/), and to some extent, [Pandas](https://pandas.pydata.org/)), with great language interoperability and interprocess communication, but without manipulation functions oriented toward data analysts.

Awkward Array is a data analyst-friendly extension of NumPy-like idioms for arbitrary data structures. It is intended to be used interchangeably with NumPy and share data with Arrow and DataFrames. Like NumPy, it simplifies and accelerates computations that transform arrays into arrays—all computations over elements in an array are compiled. Also like NumPy, imperative-style computations can be accelerated with [Numba](https://numba.pydata.org/).

Note that there is also a [ragged](https://github.com/scikit-hep/ragged) array library with simpler (but still non-rectangular) data types that more closely adheres to [array APIs](https://data-apis.org/array-api/latest/API_specification).

::::::
:::::::

:::::::{grid-item} 
::::::{dropdown} Where is an Awkward Array's `shape` and `dtype`?

Since Awkward Arrays can contain arbitrary data structures, their type can't be separated into a `shape` and a `dtype`, the way a NumPy array can.

For an array of records like

```python
import awkward as ak

example = ak.Array([
    [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}],
    [],
    [{"x": 4.4, "y": [1, 2, 3, 4]}, {"x": 5.5, "y": [1, 2, 3, 4, 5]}]
])
```

the `x` field contains floating point numbers and the `y` field contains lists of integers. They would have different `dtypes`, as well as different numbers of dimensions. This array also can't be separated into `x` and `y` columns with different `dtypes`, as in a DataFrame, since both fields are inside of records in a variable-length list.

Instead, Awkward Arrays have a `type`, which looks like

```python
3 * var * {x: float64, y: var * int64}
```

for the above. This combines `shape` and `dtype` information in the following way: the length of the array is `3`, the first dimension has `var` or variable length, it contains records with `x` and `y` field names in `{` `}`, the `x` field has `float64` primitive type and the `y` field is a `var` variable length list of `int64`. You can `print(array.type)` or `array.type.show()` to see the type of any `array`. (For more, see the [DataShape language](https://datashape.readthedocs.io/).)

See the [ragged](https://github.com/scikit-hep/ragged) array library for variable-length dimensions that are nevertheless separable into a `shape` and `dtype`, like a conventional array.

::::::
:::::::

:::::::{grid-item} 
::::::{dropdown} How do I get Awkward Arrays, or read or write files of them?

After importing Awkward Array with

```python
import awkward as ak
```

the `ak.Array` constructor takes [NumPy arrays](https://numpy.org/), [CuPy arrays](https://cupy.dev/), [pyarrow arrays](https://arrow.apache.org/docs/python/), or an iterable of Python builtin lists and dicts, such as

```python
example = ak.Array([
    [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}],
    [],
    [{"x": 4.4, "y": [1, 2, 3, 4]}, {"x": 5.5, "y": [1, 2, 3, 4, 5]}]
])
```

This is a shorthand for functions such as {func}`ak.from_numpy`, {func}`ak.from_cupy`, {func}`ak.from_arrow`, and {func}`ak.from_iter`, which you can call explicitly for more control. Similarly, functions like {func}`ak.to_numpy`, {func}`ak.to_cupy`, {func}`ak.to_arrow`, and {func}`ak.to_list` convert Awkward Arrays into other types of arrays, or Python lists.

Several file formats have `ak.from_*` and `ak.to_*` functions, such as JSON, Parquet, and Feather. To read and write ROOT files, see [Uproot](https://uproot.readthedocs.io/).

In addition, there are low-level routines, {func}`ak.from_buffers` and {func}`ak.to_buffers`, to build new file or line protocol interfaces.

::::::
:::::::

:::::::{grid-item} 
::::::{dropdown} How do I slice Awkward Arrays?

Like NumPy: all [NumPy slicing rules](https://numpy.org/doc/stable/user/basics.indexing.html) are supported, with generalizations to support more data types, as well as slicing rules that have no analog in rectangular arrays; see {func}`ak.Array.__getitem__`.

Some common examples using

```python
import awkward as ak

example = ak.Array([
    [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}],
    [],
    [{"x": 4.4, "y": [1, 2, 3, 4]}, {"x": 5.5, "y": [1, 2, 3, 4, 5]}]
])
```

are

* `example[0:-1]` to select a range in the first dimension,
* `example[:, 1:]` to keep all elements of the first dimension but drop the first element of each nested list,
* `example["x"]` or `array.x` to select the `x` field of all records,
* `example[array.x > 3]` to select all records in which the field `x` is greater than `3`,
* `example.y[:, :, [0, -1]]` to select field `y` and take the first (`0`) and last (`-1`) element of each list of field `y`,
* and so on.

::::::
:::::::

:::::::{grid-item} 
::::::{dropdown} How do I use NumPy functions with Awkward Arrays?

All NumPy [universal functions](https://numpy.org/doc/stable/reference/ufuncs.html) can be applied to Awkward Arrays that do not contain record structures, as well as any other functions that have Awkward equivalents, such as {func}`ak.sum`, {func}`ak.argmax`, {func}`ak.mean`, {func}`ak.sort`, and {func}`ak.concatenate`.

For example, with

```python
import awkward as ak
import numpy as np

y = ak.Array([
    [[1], [1, 2], [1, 2, 3]],
    [],
    [[1, 2, 3, 4], [1, 2, 3, 4, 5]]
])
```

you can call

* `np.sqrt(y)` to get an array of lists of lists of the square roots of the numbers above,
* `y * 2` or `y + y` to multiply every value by 2 (which calls `np.multiply` or `np.add`, which are NumPy ufuncs),
* `np.sum(y)` to get the sum of all values,
* `np.argmax(y, axis=-1)` to get the position of the maximum value of each inner list,
* `np.mean(y, axis=0)` to get the mean of the first elements of each list, the second elements, and so on,
* `np.sort(y)` to sort lists,
* `np.concatenate((y, y))` to concatenate them,
* and so on.

::::::
:::::::

:::::::{grid-item} 
::::::{dropdown} How do I flatten a ragged array for plotting?

{func}`ak.flatten` eliminates one level of nested lists, and {func}`ak.ravel` eliminates them all. {func}`ak.flatten` also removes missing values (`None`), which plotting libraries might not recognize.

Depending on what you're trying to plot, selecting the first element of each list or computing the {func}`ak.sum` or {func}`ak.mean` of each list might be more meaningful.

::::::
:::::::

:::::::{grid-item} 
::::::{dropdown} How do I make ragged dimensions regular for (ML) algorithms that require it?

The {func}`ak.to_regular` function changes the _data type_ from variable-length (`var`) to fixed-length _if_ all lists in that dimension happen to have the same length anyway.

If you need to _change_ the data to make it conform to a rectangular shape, you can

* slice it to the minimum {func}`ak.num` (and then use {func}`ak.to_regular` to formalize the data type as being regular)
* perform a reduction over a ragged dimension, such as {func}`ak.sum` or {func}`ak.mean`,
* {func}`ak.pad_none` to pad the lists to the maximum {func}`ak.num` and then use {func}`ak.fill_none` to replace the missing values with a value of your choice,
* use {func}`ak.pad_none` with `clip=True` to pad and clip in one step.

::::::
:::::::

:::::::{grid-item} 
::::::{dropdown} How can I make or break records in arrays?

Record (struct/class) data structures may come from JSON objects, Arrow Tables, Parquet columns, etc. This small Python dataset produces an array of lists of records:

```python
import awkward as ak

example = ak.Array([
    [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}],
    [],
    [{"x": 4.4, "y": [1, 2, 3, 4]}, {"x": 5.5, "y": [1, 2, 3, 4, 5]}]
])
```

Individual fields can be extracted by slicing it: `example["x"]` (`example.x`) and `example["y"]` (`example.y`), and all fields can be extracted at once with {func}`ak.unzip`:

```python
x, y = ak.unzip(example)
```

The following is a particularly useful idiom, for turning an array of records into a Python dict of arrays, using both {func}`ak.fields` and {func}`ak.unzip`:

```python
dict_of_arrays = dict(zip(ak.fields(example), ak.unzip(example)))
```

The opposite, {func}`ak.zip`, takes a Python dict of arrays and makes a record array:

```python
ak.zip(dict_of_arrays)
```

When a set of Awkward Arrays are zipped together, it's not clear which level of nested lists should be populated with records; {func}`ak.zip` attempts to create records at the deepest level, inside of all nested lists (which might not even be possible, if the Awkward Arrays don't have the same list lengths at all levels). The `depth_limit` argument of {func}`ak.zip` controls this:

```python
ak.zip(dict_of_arrays, depth_limit=2)
```

reproduces the original `example`, in which the `y` field has one more dimension than the `x` field (scalar `x` values sit beside `y` values that are lists).

::::::
:::::::

:::::::{grid-item} 
::::::{dropdown} How do I add a field to an existing record array?

As a shorthand for {func}`ak.unzip`, add a field, and {func}`ak.zip` (see the question above), new fields can be assigned  with {func}`ak.Array.__setitem__`.

For example, with

```python
import awkward as ak

example = ak.Array([
    [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}],
    [],
    [{"x": 4.4, "y": [1, 2, 3, 4]}, {"x": 5.5, "y": [1, 2, 3, 4, 5]}]
])
```

you can add a third field, `z` to the record with

```python
example["z"] = example.x * 10
```

Note that for assignment, the left-hand side must be expressed with square brackets, not a dot. This is to support assignment into records nested within records.

::::::
:::::::

:::::::{grid-item} 
::::::{dropdown} Why can't I assign numerical values in an array?

Awkward Arrays are immutable, and almost all operations on them view parts of a data structure and only replace the parts that have changed. Therefore, with an array like

```python
import awkward as ak

example = ak.Array([
    [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}],
    [],
    [{"x": 4.4, "y": [1, 2, 3, 4]}, {"x": 5.5, "y": [1, 2, 3, 4, 5]}]
])
```

attempting to assign

```python
example[0, "x", 0] = 999
```

results in an error. (If it were allowed, it could have unpredictable consequences.) Immutability is not enforced at a very low level, so if you know what you're doing, you can deconstruct the array, view it in NumPy, Arrow, or as raw memory buffers, and change it.

Problems that would be solved by assigning values in place can usually be solved by {func}`ak.where`.

The only kind of assignment that _is_ allowed is to add new fields, such as

```python
example["z"] = 999
```

(see the question above). This kind of assignment won't cause values in another array to change unpredictably.

::::::
:::::::

:::::::{grid-item} 
::::::{dropdown} How do I get rid of missing values (`None`)?

Some functions, such as {func}`ak.min` or {func}`ak.max` on empty lists, produce missing values. For example, with

```python
import awkward as ak

x = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
```

`ak.max(x, axis=1)` returns

```
<Array [3.3, None, 5.5] type='3 * ?float64'>
```

`None` represents a missing value (distinct from floating-point `nan`), and the `?` or `option[float64]` in the type means that values _could_ be missing. Such an array can be used in numerical calculations—missing values pass through most functions as missing values in the output—but third-party libraries might not recognize them.

* {func}`ak.drop_none` simply removes the missing values, changing the lengths of lists and the data type to reflect the fact that no values are missing.
* {func}`ak.flatten` removes missing values in the process of flattening nested lists (it treats `None` like `[]`).
* {func}`ak.fill_none` lets you replace missing values with a specified value.
* {func}`ak.firsts` and {func}`ak.singletons` convert between representing option-type data as `option[T]` and `var * T`. In the latter, a missing value is an empty list and a non-missing value is a length-1 list.

::::::
:::::::

:::::::{grid-item} 
::::::{dropdown} Why am I getting ValueError or IndexError in mathematical operations?

Most likely, your arrays don't line up at every level of nested lists. This is a generalization of a `shape` mismatch in rectangular arrays.

For example, with

```python
import awkward as ak

x = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
y = ak.Array([[1.1, 2.2, 3.3, 999], [], [4.4, 5.5]])
```

an attempt to add `x + y` would fail because even though `x` and `y` have the same array length (3), the length of the first list differs (3 versus 4).

This type of error is often more subtle than the example above. It won't happen if two arrays are derived from the same array with shape-preserving operations, but if, for instance, you remove outlier data from from one array and not another, they may fail to line up somewhere in the middle of a large dataset.

One way to avoid that is to introduce missing values (`None`) instead of removing outliers. Whereas

```python
x[x > 2]
```

makes an array without values smaller than 2,

```
<Array [[2.2, 3.3], [], [4.4, 5.5]] type='3 * var * float64'>
```

a mask,

```python
x.mask[x > 2]
```

replaces the values smaller than 2 with `None`:

```
<Array [[None, 2.2, 3.3], [], [4.4, 5.5]] type='3 * var * ?float64'>
```

This preserves the shape of the array so that it can continue to be used in mathematical expressions. For instance, `x + x.mask[x > 2]` returns

```
<Array [[None, 4.4, 6.6], [], [8.8, 11]] type='3 * var * ?float64'>
```

(the missing value propagates through to the output).

Missing values can be dropped, using {func}`ak.drop_none`, or replaced, using {func}`ak.fill_none`, as described in the question above.

::::::
:::::::

:::::::{grid-item} 
::::::{dropdown} How do I use Awkward Array with Numba?

Awkward Arrays can be passed into and out of functions that have been JIT-compiled with [Numba](https://numba.pydata.org/). For example, with

```python
import awkward as ak
import numpy as np
import numba as nb

example = ak.Array([
    [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}],
    [],
    [{"x": 4.4, "y": [1, 2, 3, 4]}, {"x": 5.5, "y": [1, 2, 3, 4, 5]}]
])
```

A function that sums `x` in each entry (like {func}`ak.sum`) can be written in JIT-compiled imperative Python like this:

```python
@nb.jit
def sum_over_x(array):
    output = np.zeros(len(array))
    for i, list_of_records in enumerate(array):
        for record in list_of_records:
            output[i] += record.x
    return output

sum_over_x(example)
```

Since Numba JIT-compiled the function, it doesn't suffer the usual slow-down of iterating in Python. On the other hand, all variables in the function must have fixed data type and adhere to Numba's set of [supported Python features](https://numba.readthedocs.io/en/stable/reference/pysupported.html) and [supported NumPy features](https://numba.readthedocs.io/en/stable/reference/numpysupported.html) to be compiled. None of Awkward Array's `ak.*` functions can be used—only iteration over values.

A JIT-compiled function can also return a part of the input Awkward Array:

```python
@nb.jit
def record_in_which_y_sums_to_10(array):
    for list_of_records in array:
        for record in list_of_records:
            if np.asarray(record.y).sum() == 10:
                return record

record_in_which_y_sums_to_10(example)
```

returns

```
<Record {x: 4.4, y: [1, ..., 4]} type='{x: float64, y: var * int64}'>
```

which is the `record` that has `np.asarray(record.y).sum() == 10`. (One-dimensional Awkward Arrays may be cast as NumPy arrays, to take advantage of NumPy functions.)

Awkward Arrays are immutable inside of JIT-compiled functions, just as they are outside. To create new Awkward Arrays with Numba, use {obj}`ak.ArrayBuilder`.

Awkward Arrays with {func}`ak.backend` equal to `"cuda"` can be passed to Numba functions on GPUs, compiled with `@nb.cuda.jit`. See {doc}`../user-guide/how-to-use-in-numba-cuda` for more.

The choice between computing outside of a Numba JIT-compiled function and outside of one is an either/or choice between imperative style in Numba (only iteration is allowed, no `ak.*` functions or fancy slices) and array-oriented style outside (iteration is slow in Python; `ak.*` functions are encouraged).

::::::
:::::::

:::::::{grid-item} 
::::::{dropdown} How do I perform computations on arrays of spatial or momentum vectors?

For 2-D, 3-D, and 4-D space and space-time vectors, see the [Vector](https://vector.readthedocs.io/) library. These can be used as momentum vectors for physics with a variety of coordinate transformations, including special relativity. As Awkward Arrays, these each vector is a record whose fields are coordinates, such as `x`, `y`, `z` or `rho`, `phi`, `theta`.

To enable Awkward Arrays of vectors, import Vector as

```python
import vector
vector.register_awkward()
```

and now any Awkward records with an appropriate name, such as

```python
example = ak.zip({
    "x": ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]),
    "y": ak.Array([[  1,   2,   3], [], [  4,   5]]),
    "z": ak.Array([[ 10,  20,  30], [], [ 40,  50]])
}, with_name="Momentum3D")
```

is recognized as an array of vectors with methods like

```python
example.phi
```

and

```python
(2 * example).is_parallel(example)
```

These methods also work in Numba JIT-compiled functions. (See the above question.)

Several array-constructing functions accept a `with_name` argument, including the {obj}`ak.Array` constructor and {func}`ak.zip`. There's also a {func}`ak.with_name` function to add a name after an array has already been created.

::::::
:::::::

:::::::{grid-item} 
::::::{dropdown} How would I write my own suite of functions, like Vector?

Add new classes or functions to {obj}`ak.behavior`, which links record names to Python code.

Names are strings that can be saved in files or transferred across networks, but Python code is not always serializable.

::::::
:::::::

:::::::{grid-item} 
::::::{dropdown} How can I delay or distribute a computation?

Use [Dask](https://www.dask.org/). The [dask-awkward](https://dask-awkward.readthedocs.io/) library provides a new high-level collection for Awkward Arrays, similar to `dask.array` and `dask.dataframe`.

For example, with

```python
import awkward as ak
import dask_awkward as dak

example = ak.Array([
    [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}],
    [],
    [{"x": 4.4, "y": [1, 2, 3, 4]}, {"x": 5.5, "y": [1, 2, 3, 4, 5]}]
])
```

you can make delayed data with

```python
dak.from_awkward(example, npartitions=1)
```

although it's more common to use [dak.from_parquet](https://dask-awkward.readthedocs.io/en/stable/api/generated/dask_awkward.from_parquet.html) or [uproot.dask](https://uproot.readthedocs.io/en/latest/uproot._dask.dask.html).

Any operations on this delayed array are collected as a Directed Acyclic Graph (DAG) that is computed when you call [dask.compute](https://distributed.dask.org/en/stable/manage-computation.html). The computation may be [distributed](https://distributed.dask.org/) across multiple CPUs on one computer or across multiple computers in a network.

The [dask-awkward project](https://github.com/dask-contrib/dask-awkward) intends to cover the same interface as Awkward Array, though there may be some functions implemented in `ak.*` that aren't in `dak.*` yet. See [dask-awkward's GitHub Issues](https://github.com/dask-contrib/dask-awkward/issues) for Dask-specific issues.

::::::
:::::::

:::::::{grid-item} 
::::::{dropdown} How do I use Awkward Array with ROOT?

[Uproot](https://uproot.readthedocs.io/) can read and write ROOT files, and works with Awkward Arrays by default.

Also, {func}`ak.to_rdataframe` and {func}`ak.from_rdataframe` converts Awkward Arrays in memory to and from ROOT's [RDataFrame](https://root.cern/doc/master/classROOT_1_1RDataFrame.html) for computations. See {doc}`../user-guide/how-to-convert-rdataframe` for details.

::::::
:::::::

:::::::{grid-item} 
::::::{dropdown} How do I use Awkward Array with C++?

One method is to convert Awkward Arrays to or from ROOT's [RDataFrame](https://root.cern/doc/master/classROOT_1_1RDataFrame.html) using {func}`ak.to_rdataframe` and {func}`ak.from_rdataframe`. RDataFrame supports computation in JIT-compiled C++.

Another method is to pass Awkward Arrays into JIT-compiled C++ functions defined with [cppyy](https://cppyy.readthedocs.io/)'s [cppdef](https://cppyy.readthedocs.io/en/latest/toplevel.html#loading-c). This interface is similar to Numba, in that the JIT-compiled functions have arbitrary arguments and return values, rather than fitting into a pipeline like RDataFrame, but it also means that you need to set up the loop over entries manually and inside the compiled block. See {doc}`../user-guide/how-to-use-in-cpp-cppyy` for details.

If you are a library developer wishing to produce and/or consume Awkward Arrays in ahead-of-time compiled code (not JIT), like [fastjet](https://github.com/scikit-hep/fastjet), you'll want to use {doc}`LayoutBuilder <../user-guide/how-to-use-header-only-layoutbuilder>`, {func}`ak.from_buffers`/{func}`ak.to_buffers`, or both. LayoutBuilder constructs an append-only array object like {obj}`ak.ArrayBuilder`, but with statically typed array type in header-only C++ that can be integrated with CMake.

::::::
:::::::

:::::::{grid-item} 
::::::{dropdown} How do I use Awkward Array with Julia?

[AwkwardArray.jl](https://github.com/JuliaHEP/AwkwardArray.jl) is a [Julia](https://julialang.org/) implementation of Awkward Array, sharing the same memory layout. It can therefore be used as a JIT-compilation target like Numba and C++ (see questions above), but with more flexibility: a single array data type can be used as an {obj}`ak.Array` and as an {obj}`ak.ArrayBuilder`. Whereas Pythonic Awkward Arrays are only _borrowed_ by JIT-compiled Numba or C++ (Python continues to own the memory and decide when it will be deleted), Julia's JIT-compiled environment is the entire environment, so such decisions don't need to be made. Julia can act as a producer and/or a consumer of Python Awkward Arrays.

See the [AwkwardArray.jl documentation](https://juliahep.github.io/AwkwardArray.jl/dev/) for details.

::::::
:::::::

:::::::{grid-item} 
::::::{dropdown} How do I emulate nested for-loops in Awkward Array (combinatorics)?

In the simplest cases, imperative code like

```python
output = []
for x in awkward_array:
    output.append(compute(x))
```

can be replaced with

```python
output = compute(awkward_array)
```

But some problems would be solved by imperative code like

```python
output = []
for x in awkward_array1:
    for y in awkward_array2:
        output.append(compute(x, y))
```

or even

```python
output = []
for i, x in enumerate(awkward_array):
    for j in range(i + 1, len(awkward_array)):  # avoid repeating x
        y = awkward_array[j]
        output.append(compute(x, y))
```

These cases involve combinatorics: a Cartesian product and sampling without replacement. To perform such operations at compiled speeds on Awkward Arrays, you may either

* JIT-compile these for loops with Numba, C++, or Julia (as in the questions above),
* use Awkward Array's combinatorics primitives.

{func}`ak.cartesian` is Awkward Array's primitive for Cartesian products: it makes an array of all pairs drawn from two (or more) provided arrays. It emulates nested, unrestricted for loops.

{func}`ak.combinations` is Awkward Array's primitive for sampling without replacement: it makes an array of all pairs drawn from an array and itself without duplicates. It emulates nested for loops that avoid repeating the same element.

These pairs (or triples, etc.) are tuples, which are records without field names. Often, `nested=True` is a useful argument to avoid flattening the output.

::::::
:::::::

:::::::{grid-item} 
::::::{dropdown} Why don't my arrays broadcast as in NumPy?

See the last section of {doc}`../user-guide/how-to-math-broadcasting`.

::::::
:::::::

:::::::{grid-item} 
::::::{dropdown} What if I need more help? What if I think I've found a bug?

After checking the tutorials on the left-bar of this Getting started guide, the User guide, and the API reference, you can ask questions about how to use a feature or solve a problem on Awkward Array's [GitHub Discussions](https://github.com/scikit-hep/awkward/discussions).

If the behavior you're seeing looks like a bug, an error in Awkward Array itself, post some simplified code to reproduce it on Awkward Array's [GitHub Issues](https://github.com/scikit-hep/awkward/issues).

::::::
:::::::

::::::::

<br><br><br><br><br>
