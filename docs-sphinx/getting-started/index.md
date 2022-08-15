# Getting started

## Installation

::::{grid} 1 1 2 2
:::{grid-item-card} Working with Conda?

Awkward Array can be installed from the conda-forge channel.
```bash
conda install -c conda-forge awkward
```

:::
:::{grid-item-card}  Prefer pip?

Binary wheels for Awkward Array are available on PyPI.
```bash
pip install awkward
```
:::
::::

## Overview
::::::{grid} 1

:::::{grid-item} 
::::{dropdown} What kind of data does Awkward Array handle?

Awkward Array is designed to make working with [_ragged_ arrays](https://en.wikipedia.org/wiki/Jagged_array) as trivial as manipulating regular (non-ragged) N-dimensional arrays in NumPy. It understands variable-length lists and missing ({data}`None`) values.

:::{figure} ../image/example-array.svg
:::
::::
:::::

:::::{grid-item} 
::::{dropdown} How do I read and write ragged arrays?

Awkward Array provides a suite of high-level IO functions (`ak.to_*` and `ak.from_*`), such as {func}`ak.to_parquet` and {func}`ak.from_parquet` that make it simple to serialise Awkward Arrays to disk, or read ragged arrays from other formats. 

It is straightforward to implement new IO functions using {func}`ak.to_buffers` and {func}`ak.from_buffers`, which convert/load Awkward Arrays from a metadata object and a collection of 1D arrays.
::::
:::::

:::::{grid-item} 
::::{dropdown} How do I see the type and shape of an array?

Awkward Array extends the [DataShape](https://datashape.readthedocs.io/en/latest/) layout language to support union types.

E.g., the following ragged array with three rows, and a variable number of columns of records with "x" and "y" fields
```python3
ak.Array([[{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}],
          [],
          [{"x": 3.3, "y": [3, 3, 3]}]])
```
has the type:
```python3
3 * var * {"x": float64, "y": var * int64}
``` 
::::
:::::

:::::{grid-item} 
::::{dropdown} How do I select a subset of an array?

Awkward Array extends the rich indexing syntax used by NumPy to support named fields and ragged indexing. The {attr}`ak.Array.mask` interface makes it easy to select a subset of an array whilst preserving its datashape.
::::
:::::


:::::{grid-item} 
::::{dropdown} How do I reshape ragged arrays to change their dimensions?

Ragged arrays do not have shapes that can be described by a collection of integers. Instead, the {func}`ak.flatten` and {func}`ak.ravel` functions can be used to remove surplus dimensions from Awkward Arrays. New, regular, dimensions can be added using {data}`numpy.newaxis`, whilst {func}`ak.unflatten` can be used to introduce a new _ragged_ axis.
::::
:::::


:::::{grid-item} 
::::{dropdown} How do I compute reductions or summary statistics?

Awkward Array supports NumPy's {np:doc}`reference/ufuncs` mechanism, and many of the high-level NumPy reducers (e.g. {func}`numpy.sum`).

:::{figure} ../image/example-reduction-sum-only.svg

::::
:::::

::::::

