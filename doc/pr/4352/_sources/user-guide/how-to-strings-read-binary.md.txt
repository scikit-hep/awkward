---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Read strings from a binary stream

Awkward Array implements support for ragged strings as ragged lists of [code-units](https://en.wikipedia.org/wiki/UTF-8). As such, successive strings are closely packed in memory, leading to high-performance operations.

+++

Let's imagine that we want to read some logging output that is stored in a text file. For example, [a subset of logs from the Android Application framework](https://zenodo.org/record/8196385).

```{code-cell} ipython3
import gzip
import itertools
import pathlib

# Preview logs
log_path = pathlib.Path("..", "samples", "Android.head.log.gz")
with gzip.open(log_path, "rt") as f:
    for line in itertools.islice(f, 8):
        print(line, end="")
```

To begin with, we can read the decompressed log-files as an array of {data}`np.uint8` dtype using NumPy, and convert the resulting array to an Awkward Array

```{code-cell} ipython3
import awkward as ak
import numpy as np

with gzip.open(log_path, "rb") as f:
    # `gzip.open` doesn't return a true file descriptor that NumPy can ingest
    # So, instead we read into memory.
    arr = np.frombuffer(f.read(), dtype=np.uint8)

raw_bytes = ak.from_numpy(arr)
raw_bytes.type.show()
```

Awkward Array doesn't support scalar values, so we can't treat these characters as a single-string. Instead we need at least one dimension. Let's unflatten our array of characters, to form a length-1 array of characters.

```{code-cell} ipython3
array_of_chars = ak.unflatten(raw_bytes, len(raw_bytes))
array_of_chars
```

We can then ask Awkward Array to treat this array of lists of characters as an array of strings, using {func}`ak.enforce_type`

```{code-cell} ipython3
string = ak.enforce_type(array_of_chars, "string")
string.type.show()
```

The underlying mechanism for implementing strings as lists of code-units can be seen if we inspect the low-level layout that builds the array

```{code-cell} ipython3
string.layout
```

The `__array__` parameter is special. It is reserved by Awkward Array, and signals that the layout is a special pre-undertood built-in type. In this case, that type of the outer {class}`ak.contents.ListOffsetArray` is "string". It can also be seen that the inner {class}`ak.contents.NumpyArray` also has an `__array__` parameter, this time with a value of `char`. In Awkward Array, an array of strings *must* look like this layout; a list with the `__array__="string"` parameter wrapping a {class}`ak.contents.NumpyArray` with the `__array__="char"` parameter.

+++

A single (very long) string isn't much use. Let's split this string at the line boundaries

```{code-cell} ipython3
split_at_newlines = ak.str.split_pattern(string, "\n")
split_at_newlines
```

Now we can remove the temporary length-1 outer dimension that was required to treat the data as a string

```{code-cell} ipython3
lines = split_at_newlines[0]
lines
```

In the low-level layout, we can see that these lines are still just variable-length lists

```{code-cell} ipython3
lines.layout
```

## Bytestrings vs strings

+++

In general, whilst strings can fundamentally be described as lists of bytes (code-units), many string operations do not operate at the byte-level. The {mod}`ak.str` submodule provides a suite of vectorised operations that operate at the code-point (*not* code-unit) level, such as computing the string length. Consider the following simple string

```{code-cell} ipython3
large_code_point = ak.Array(["Å"])
```

In Awkward Array, strings are UTF-8 encoded, meaning that a single code-point may comprise up to four code-units (bytes). Although it looks like this is a single character, if we look at the layout it's clear that the number of code-units is in-fact two

```{code-cell} ipython3
large_code_point.layout
```

This is reflected in the {func}`ak.num` function

```{code-cell} ipython3
ak.num(large_code_point)
```

The {mod}`ak.str` module provides a function for computing the length of a string

```{code-cell} ipython3
ak.str.length(large_code_point)
```

Clearly _this_ function is code-point aware.

+++

If one wants to drop the UTF-8 string abstraction, and instead deal with strings as raw byte arrays, there is the `bytes` type

```{code-cell} ipython3
large_code_point_bytes = ak.enforce_type(large_code_point, "bytes")
large_code_point_bytes
```

The layout of this array has different `"bytestring"` and `"byte"` parameters

```{code-cell} ipython3
large_code_point_bytes.layout
```

Many of the functions in the {mod}`ak.str` module treat bytestrings and strings differently; in the latter case, strings are often manipulated in terms of code-points instead of code-units. Consider {func}`ak.str.length` for this array

```{code-cell} ipython3
ak.str.length(large_code_point_bytes)
```

This is clearly counting the bytes (code-units), not code-points.
