---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.0
kernelspec:
  display_name: Python 3 (ipykernel)
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
ak_array = ak.Array(
    [
        [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}],
        [],
        [{"x": 4.4, "y": [1, 2, 3, 4]}, {"x": 5.5, "y": [1, 2, 3, 4, 5]}],
    ]
)
ak_array
```

The {func}`ak.to_buffers` function decomposes it into a set of one-dimensional arrays (a zero-copy operation).

```{code-cell} ipython3
form, length, container = ak.to_buffers(ak_array)
```

The pieces needed to reconstitute this array are:

   * the {class}`ak.forms.Form`, which defines how structure is built from one-dimensional arrays,
   * the length of the original array,
   * the one-dimensional arrays in the `container` (a {class}`collections.abc.MutableMapping`).

The {class}`ak.forms.Form` is like an Awkward {class}`ak.types.Type` in that it describes how the data are structured, but with more detail: it includes distinctions such as the difference between {class}`ak.contents.ListArray` and {class}`ak.contents.ListOffsetArray`, as well as the integer types of structural {class}`ak.index.Index`.

It is usually presented as JSON, and has a compact JSON format (when {meth}`ak.forms.Form.tojson` is invoked).

```{code-cell} ipython3
form
```

In this case, the `length` is just an integer. It would be a list of integers if `ak_array` was partitioned.

```{code-cell} ipython3
length
```

This `container` is a new dict, but it could have been a user-specified {class}`collections.abc.MutableMapping` if passed into {func}`ak.to_buffers` as an argument.

```{code-cell} ipython3
container
```

From buffers to Awkward
-----------------------

The function that reverses {func}`ak.to_buffers` is {func}`ak.from_buffers`. Its first three arguments are `form`, `length`, and `container`.

```{code-cell} ipython3
ak.from_buffers(form, length, container)
```

Minimizing the size of the output buffers
-----------------------------------------

The {func}`ak.to_buffers`/{func}`ak.from_buffers` functions exactly preserve an array, warts and all. Often, you'll want to only write {func}`ak.to_packed` arrays. "Packing" replaces an array structure with an equivalent structure that has no unreachable elements—data that you can't see as part of the array, and therefore probably don't want to write.

Here is an example of an array in need of packing:

```{code-cell} ipython3
unpacked = ak.Array(
    ak.contents.ListArray(
        ak.index.Index64(np.array([4, 10, 1])),
        ak.index.Index64(np.array([7, 10, 3])),
        ak.contents.NumpyArray(np.array([999, 4.4, 5.5, 999, 1.1, 2.2, 3.3, 999])),
    )
)
unpacked
```

This {class}`ak.contents.ListArray` is in a strange order and the `999` values are unreachable. (Also, using `starts[1] == stops[1] == 10` to represent an empty list is a little odd, though allowed by the specification.)

The {func}`ak.to_buffers` function dutifully writes the `999` values into the output, even though they're not visible in the array.

```{code-cell} ipython3
ak.to_buffers(unpacked)
```

If the intended purpose of calling {func}`ak.to_buffers` is to write to a file or send data over a network, this is wasted space. It can be trimmed by calling the {func}`ak.to_packed` function.

```{code-cell} ipython3
packed = ak.to_packed(unpacked)
packed
```

At high-level, the array appears to be the same, but its low-level structure is quite different:

```{code-cell} ipython3
unpacked.layout
```

```{code-cell} ipython3
packed.layout
```

This version of the array is more concise when written with {func}`ak.to_buffers`:

```{code-cell} ipython3
ak.to_buffers(packed)
```

Saving Awkward Arrays to HDF5
-----------------------------

The [h5py](https://www.h5py.org/) library presents each group in an HDF5 file as a {class}`collections.abc.MutableMapping`, which we can use as a container for an array-set. We must also save the `form` and `length` as metadata for the array to be retrievable.

```{code-cell} ipython3
file = h5py.File("/tmp/example.hdf5", "w")
group = file.create_group("awkward")
group
```

We can fill this `group` as a `container` by passing it in to {func}`ak.to_buffers`. (See the previous section for more on {func}`ak.to_packed`.)

```{code-cell} ipython3
form, length, container = ak.to_buffers(ak.to_packed(ak_array), container=group)
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
np.asarray(container["node0-offsets"])
```

Now we need to add the other information to the group as metadata. Since HDF5 accepts string-valued metadata, we can put it all in as JSON or numbers.

```{code-cell} ipython3
group.attrs["form"] = form.to_json()
group.attrs["form"]
```

```{code-cell} ipython3
group.attrs["length"] = json.dumps(length)  # JSON-encode it because it might be a list
group.attrs["length"]
```

Reading Awkward Arrays from HDF5
--------------------------------

With that, we can reconstitute the array by supplying {func}`ak.from_buffers` the right arguments from the group and metadata.

The group can't be used as a `container` as-is, since subscripting it returns {class}`h5py.Dataset` objects, rather than arrays.

```{code-cell} ipython3
reconstituted = ak.from_buffers(
    ak.forms.from_json(group.attrs["form"]),
    json.loads(group.attrs["length"]),
    {k: np.asarray(v) for k, v in group.items()},
)
reconstituted
```
