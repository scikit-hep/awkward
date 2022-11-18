# Getting started

## Installation

::::::{grid} 1 1 2 2
:::::{grid-item-card} Working with Conda?

Awkward Array can be installed from the conda-forge channel.
```bash
conda install -c conda-forge awkward
```

:::::
:::::{grid-item-card}  Prefer pip?

Binary wheels for Awkward Array are available on PyPI.
```bash
pip install awkward
```
:::::
::::::

## Overview
::::::::{grid} 1

:::::::{grid-item} 
::::::{dropdown} What kind of data does Awkward Array handle?

Awkward Array is designed to make working with [_ragged_ arrays](https://en.wikipedia.org/wiki/Jagged_array) as trivial as manipulating regular (non-ragged) N-dimensional arrays in NumPy. It understands data with variable-length lists,
```pycon
>>> ak.Array([
...     [1, 2, 3],
...     [4]
... ])
<Array [[1, 2, 3], [4]] type='2 * var * int64'>
```
missing ({data}`None`) values,
```pycon
>>> ak.Array([1, None])
<Array [1, None] type='2 * ?int64]'>
```
record structures,
```pycon
>>> ak.Array([{'x': 1, 'y': 2}])
<Array [{x: 1, y: 2}] type='1 * {x: int64, y: int64}'>
```
and even union-types!
```pycon
>>> ak.Array([1, "hi", None])
<Array [1, 'hi', None] type='3 * union[?int64, ?string]'>
```
::::::
:::::::

:::::::{grid-item} 
::::::{dropdown} How do I read and write ragged arrays?

Awkward Array provides a suite of high-level IO functions (`ak.to_*` and `ak.from_*`), such as {func}`ak.to_parquet` and {func}`ak.from_parquet` that make it simple to serialise Awkward Arrays to disk, or read ragged arrays from other formats. 

In addition to specialised IO reading and writing routines, Awkward Arrays can also be serialised to/from a set of one dimensional buffers with the {func}`ak.to_buffers`/{func}`ak.from_buffers` functions. These buffers can then be written to/read from a wide range of existing array serialisation formats that understand NumPy arrays, e.g. {func}`numpy.savez`. 
::::::
:::::::

:::::::{grid-item} 
::::::{dropdown} How do I see the type and shape of an array?

Ragged arrays do not have shapes that can be described by a collection of integers. Instead, Awkward Array uses an extended version of the [DataShape](https://datashape.readthedocs.io/en/latest/) layout language to describe the structure and type of an Array. The {attr}`ak.Array.type` attribute of an array reveals its DataShape:
```pycon
>>> array = ak.Array([[{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}],
...           [],
...           [{"x": 3.3, "y": [3, 3, 3]}]])
>>> array.type
3 * var * {"x": float64, "y": var * int64}
```
::::::
:::::::

:::::::{grid-item} 
::::::{dropdown} How do I select a subset of an array?

Awkward Array extends the rich indexing syntax used by NumPy to support named fields and ragged indexing:
```pycon
>>> array = ak.Array([
...     [1, 2, 3], 
...     [6, 7, 8, 9]
... ])
>>> is_even = (array % 2) == 0
>>> array[is_even].to_list()
<Array [[2], [6, 8]] type='2 * var * int64'>
``` 

Meanwhile, the {attr}`ak.Array.mask` interface makes it easy to select a subset of an array whilst preserving its structure:
```pycon
>>> array.mask[is_even].to_list()
[[None, 2, None], [4, None], [6, None, 8, None]]
``` 
::::::
:::::::


:::::::{grid-item} 
::::::{dropdown} How do I reshape ragged arrays to change their dimensions?
New, regular, dimensions can be added using {data}`numpy.newaxis`, whilst {func}`ak.unflatten` can be used to introduce a new _ragged_ axis.
```pycon
>>> array = ak.Array([1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> array[:, np.newaxis]
<Array [[1], [2], [3], [4], [5], [6], [7], [8], [9]] type='9 * 1 * int64'>
>>> ak.unflatten(array, [3, 2, 4])
<Array [[0, 1, 2], [3, 4], [5, 6, 7, 8]] type='3 * var * int64'>
``` 
The {func}`ak.flatten` and {func}`ak.ravel` functions can be used to remove surplus (or all) dimensions from Awkward Arrays. 
```pycon
>>> array = ak.Array([
...     [1, 2, 3], 
...     [6, 7, 8, 9]
... ])
>>> ak.flatten(array, axis=1)
<Array [1, 2, 3, 6, 7, 8, 9] type='7 * int64'>
>>> ak.ravel(array)
<Array [1, 2, 3, 6, 7, 8, 9] type='7 * int64'>
``` 
::::::
:::::::


:::::::{grid-item} 
::::::{dropdown} How do I compute reductions or summary statistics?

Awkward Array supports NumPy's {np:doc}`reference/ufuncs` mechanism, and many of the high-level NumPy reducers (e.g. {func}`numpy.sum`).

:::::{grid} 1 1 2 2

::::{grid-item}
```pycon
>>> array = ak.Array([
...     [1,    2,    4], 
...     [             ],
...     [None, 8      ],
...     [16           ]
... ])
>>> ak.sum(array, axis=0)
<Array [17, 10, 4] type='3 * int64'>
>>> ak.sum(array, axis=1)
<Array [7, 0, 8, 16] type='4 * int64'>
``` 
::::

::::{grid-item}
:::{figure} ../image/example-reduction-sum-only.svg
:::
::::

:::::
::::::
:::::::

::::::::

