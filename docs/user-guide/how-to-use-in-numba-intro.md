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

Using Awkward Array with Numba
==============================

## Why Numba?

The array-oriented (NumPy-like) interface that Awkward Array provides is often more convenient than imperative code and it's always faster than pure Python. But sometimes it's less convenient than imperative code and it's always slower than C, C++, Julia, Rust, or other compiled code.

* The matching problem described in {doc}`how-to-combinatorics-best-match` is already rather complex—if a problem is more intricate than that, you may want to consider doing it in imperative code, so that you or anyone reading your code don't get lost in indices.
* Although all iterations over arrays in Awkward Array are precompiled, most operations involve several passes over the data, which are not cache-friendly and might exceed your working memory budget.

For this reason, Awkward Arrays were made to be interchangeable with [Numba](https://numba.pydata.org/), a JIT-compiler for Python. Recently, JIT-compiled C++ and Julia have been added as well. Our intention is not to make you choose upfront whether to use array-oriented syntax or JIT-compiled code, but to mix them in the most convenient ways for each task.

## Small example

```{code-cell} ipython3
import awkward as ak
import numpy as np
import numba as nb
```

```{code-cell} ipython3
array = ak.Array([
    [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}],
    [],
    [{"x": 4.4, "y": [1, 2, 3, 4]}, {"x": 5.5, "y": [1, 2, 3, 4, 5]}],
    [{"x": 6.6, "y": [1, 2, 3, 4, 5, 6]}],
])[np.tile([0, 1, 2, 3], 250000)]
array
```

Suppose we want to compute the sum of all `y` values in each of the million entries above. We can do that with a simple Awkward expression,

```{code-cell} ipython3
ak.sum(ak.sum(array.y, axis=-1), axis=-1)
```

Although it's faster than iterating over pure Python loops, it makes intermediate arrays that aren't necessary for the final result. Allocating them and iterating over all of them slows down the Awkward Array expression relative to compiled code.

```{code-cell} ipython3
%%timeit

ak.sum(ak.sum(array.y, axis=-1), axis=-1)
```

```{code-cell} ipython3
@nb.jit
def sum_of_y(array):
    out = np.zeros(len(array), dtype=np.int64)

    for i, list_of_records in enumerate(array):
        for record in list_of_records:
            for y in record.y:
                out[i] += y

    return out
```

```{code-cell} ipython3
ak.Array(sum_of_y(array))
```

The JIT-compiled function is faster.

```{code-cell} ipython3
%%timeit

ak.Array(sum_of_y(array))
```

## Combining features of Awkward Array and Numba

Even on a per-task level, Awkward Array's array-oriented functions and Numba's JIT-compilation don't need to be exclusive. Numba can be used to prepare steps of an array-oriented process, such as generating boolean or integer-valued arrays to use as slices for an Awkward Array.

```{code-cell} ipython3
@nb.jit
def sum_of_y_is_more_than_10(array):
    out = np.zeros(len(array), dtype=np.bool_)

    for i, list_of_records in enumerate(array):
        total = 0
        for record in list_of_records:
            for y in record.y:
                total += y
        if total > 10:
            out[i] = True

    return out
```

```{code-cell} ipython3
array[sum_of_y_is_more_than_10(array)]
```

## Relative strengths and weaknesses

Awkward Array's array oriented interface is

* good for reading and writing data to and from columnar file formats like Parquet,
* good for interactive exploration in Jupyter, applying a sequence of simple operations to a whole dataset and observing its effects after each operation,
* good for speed and memory use, relative to pure Python,
* bad for very intricate calculations with many indices,
* bad for large intermediate arrays,
* bad for speed and memory use, relative to custom-compiled code.

Numba's JIT-compilation is

* good for writing understandable algorithms with many moving parts,
* good for speed and memory use, on par with other compiled languages,
* bad for interactive exploration of data and iterative data analysis, since you have to write whole functions,
* bad for working through type errors, as you would have in any compiled language (unlike pure Python),
* bad for unboxing and boxing large non-array data when entering and exiting a compiled function.

The {doc}`next section <how-to-use-in-numba-features>` lists what you can and can't do with Awkward Arrays in Numba-compiled code.
