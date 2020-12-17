---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.10'
    jupytext_version: 1.5.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

Generic buffers
===============

Most of the conversion functions target a particular library: NumPy, Arrow, Pandas, or Python itself. As a catch-all for other storage formats, Awkward Arrays can be converted to and from sets of named buffers. The buffers are not (usually) intelligible on their own; the length of the array and a JSON document are needed to reconstitute the original structure. This section will demonstrate how an array-set can be used to store an Awkward Array in an HDF5 file, which ordinarily wouldn't be able to represent nested, irregular data structures.

```{code-cell} ipython3
import awkward as ak
import numpy as np
import h5py
import json
```

From Awkward to buffers
-----------------------

Consider the following complex array:

```{code-cell} ipython3
ak_array = ak.Array([
    [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}],
    [],
    [{"x": 4.4, "y": [1, 2, 3, 4]}, {"x": 5.5, "y": [1, 2, 3, 4, 5]}],
])
ak_array
```

The [ak.to_buffers](https://awkward-array.readthedocs.io/en/latest/_auto/ak.to_buffers.html) function decomposes it into a set of one-dimensional arrays (a zero-copy operation).

```{code-cell} ipython3
form, length, container = ak.to_buffers(ak_array)
```

The pieces needed to reconstitute this array are:

   * the [Form](https://awkward-array.readthedocs.io/en/latest/ak.forms.Form.html), which defines how structure is built from one-dimensional arrays,
   * the length of the original array or lengths of all of its partitions ([ak.partitions](https://awkward-array.readthedocs.io/en/latest/_auto/ak.partitions.html)),
   * the one-dimensional arrays in the `container` (a [MutableMapping](https://docs.python.org/3/library/collections.abc.html#collections-abstract-base-classes)).

The [Form](https://awkward-array.readthedocs.io/en/latest/ak.forms.Form.html) is like an Awkward [Type](https://awkward-array.readthedocs.io/en/latest/ak.types.Type.html) in that it describes how the data are structured, but with more detail: it includes distinctions such as the difference between [ListArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.ListArray.html) and [ListOffsetArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.ListOffsetArray.html), as well as the integer types of structural [Indexes](https://awkward-array.readthedocs.io/en/latest/ak.layout.Index.html).

It is usually presented as JSON, and has a compact JSON format (when [Form.tojson](https://awkward-array.readthedocs.io/en/latest/ak.forms.Form.html#ak-forms-form-tojson) is invoked).

```{code-cell} ipython3
form
```

In this case, the `length` is just an integer. It would be a list of integers if `ak_array` was partitioned.

```{code-cell} ipython3
length
```

This `container` is a new dict, but it could have been a user-specified [MutableMapping](https://docs.python.org/3/library/collections.abc.html#collections-abstract-base-classes) if passed into [ak.to_buffers](https://awkward-array.readthedocs.io/en/latest/_auto/ak.to_buffers.html) as an argument.

```{code-cell} ipython3
container
```

From buffers to Awkward
-----------------------

The function that reverses [ak.to_buffers](https://awkward-array.readthedocs.io/en/latest/_auto/ak.to_buffers.html) is [ak.from_buffers](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_buffers.html). Its first three arguments are `form`, `length`, and `container`.

```{code-cell} ipython3
ak.from_buffers(form, length, container)
```

Saving Awkward Arrays to HDF5
-----------------------------

The [h5py](https://www.h5py.org/) library presents each group in an HDF5 file as a [MutableMapping](https://docs.python.org/3/library/collections.abc.html#collections-abstract-base-classes), which we can use as a container for an array-set. We must also save the `form` and `length` as metadata for the array to be retrievable.

```{code-cell} ipython3
file = h5py.File("/tmp/example.hdf5", "w")
group = file.create_group("awkward")
group
```

We can fill this `group` as a `container` by passing it in to [ak.to_buffers](https://awkward-array.readthedocs.io/en/latest/_auto/ak.to_buffers.html).

```{code-cell} ipython3
form, length, container = ak.to_buffers(ak_array, container=group)
```

```{code-cell} ipython3
container
```

Now the HDF5 group has been filled with array pieces.

```{code-cell} ipython3
container.keys()
```

Here's one.

```{code-cell} ipython3
np.asarray(container["part0-node0-offsets"])
```

Now we need to add the other information to the group as metadata. Since HDF5 accepts string-valued metadata, we can put it all in as JSON or numbers.

```{code-cell} ipython3
group.attrs["form"] = form.tojson()
group.attrs["form"]
```

```{code-cell} ipython3
group.attrs["length"] = json.dumps(length)   # JSON-encode it because it might be a list
group.attrs["length"]
```

Reading Awkward Arrays from HDF5
--------------------------------

With that, we can reconstitute the array by supplying [ak.from_buffers](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_buffers.html) the right arguments from the group and metadata.

The group can't be used as a `container` as-is, since subscripting it returns `h5py.Dataset` objects, rather than arrays.

```{code-cell} ipython3
reconstituted = ak.from_buffers(
    ak.forms.Form.fromjson(group.attrs["form"]),
    json.loads(group.attrs["length"]),
    {k: np.asarray(v) for k, v in group.items()},
)
reconstituted
```

Like [ak.from_parquet](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_parquet.html), [ak.from_buffers](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_buffers.html) has the option to read lazily, only accessing record fields and partitions that are accessed.

```{code-cell} ipython3
class LazyGet:
    def __init__(self, group):
        self.group = group
    
    def __getitem__(self, key):
        print(key)
        return np.asarray(self.group[key])

lazy = ak.from_buffers(
    ak.forms.Form.fromjson(group.attrs["form"]),
    json.loads(group.attrs["length"]),
    LazyGet(group),
    lazy=True,
)
```

The `LazyGet` class prints out any keys that actually get read from the HDF5 file, when they get read. Nothing has been printed yet.

```{code-cell} ipython3
lazy.x
```

Now that we have looked at the `"x"` field, `"node0-offsets"` (for the outer list structure) and `"node2"` (the `"x"` values) have been read.

```{code-cell} ipython3
lazy.x
```

They were only read once.

```{code-cell} ipython3
lazy.y
```

Looking at the `"y"` field causes the `"node3-offsets"` (inner list structure) and `"node4"` (`"y"` values) to be read.

```{code-cell} ipython3
lazy.y
```

Only once.
