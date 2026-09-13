# Generic buffers

Most of the conversion functions target a particular library: NumPy, Arrow, Pandas, or Python itself. As a catch-all for other storage formats, Awkward Arrays can be converted to and from sets of named buffers. The buffers are not (usually) intelligible on their own; the length of the array and a JSON document are needed to reconstitute the original structure. This section will demonstrate how an array-set can be used to store an Awkward Array in an HDF5 file, which ordinarily wouldn’t be able to represent nested, irregular data structures.

```ipython3
import awkward as ak
import numpy as np
import h5py
import json
```

## From Awkward to buffers

Consider the following complex array:

```ipython3
ak_array = ak.Array(
    [
        [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}],
        [],
        [{"x": 4.4, "y": [1, 2, 3, 4]}, {"x": 5.5, "y": [1, 2, 3, 4, 5]}],
    ]
)
ak_array
```

The [`ak.to_buffers()`](sphinx-llm:0f37278395f642a289396d6393d4d837#ak.to_buffers) function decomposes it into a set of one-dimensional arrays (a zero-copy operation).

```ipython3
form, length, container = ak.to_buffers(ak_array)
```

The pieces needed to reconstitute this array are:

* the [`ak.forms.Form`](sphinx-llm:6295086f75dd4713a4cbcfec3594d1a1#ak.forms.Form), which defines how structure is built from one-dimensional arrays,
* the length of the original array,
* the one-dimensional arrays in the `container` (a [`collections.abc.MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)).

The [`ak.forms.Form`](sphinx-llm:6295086f75dd4713a4cbcfec3594d1a1#ak.forms.Form) is like an Awkward [`ak.types.Type`](sphinx-llm:fc995ece0ffa4f68adbe74567c7b2ea1#ak.types.Type) in that it describes how the data are structured, but with more detail: it includes distinctions such as the difference between [`ak.contents.ListArray`](sphinx-llm:e2e74f5b6c644d858b90eaa78cd4dedc#ak.contents.ListArray) and [`ak.contents.ListOffsetArray`](sphinx-llm:00659f65177e4a1db8f94cec2f25984d#ak.contents.ListOffsetArray), as well as the integer types of structural [`ak.index.Index`](sphinx-llm:c5dc164fb63f49c0946c0b5dfb338435#ak.index.Index).

It is usually presented as JSON, and has a compact JSON format (when `ak.forms.Form.tojson()` is invoked).

```ipython3
form
```

In this case, the `length` is just an integer. It would be a list of integers if `ak_array` was partitioned.

```ipython3
length
```

This `container` is a new dict, but it could have been a user-specified [`collections.abc.MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping) if passed into [`ak.to_buffers()`](sphinx-llm:0f37278395f642a289396d6393d4d837#ak.to_buffers) as an argument.

```ipython3
container
```

## From buffers to Awkward

The function that reverses [`ak.to_buffers()`](sphinx-llm:0f37278395f642a289396d6393d4d837#ak.to_buffers) is [`ak.from_buffers()`](sphinx-llm:fba1faa82f7649e28003bf8ebdda86f8#ak.from_buffers). Its first three arguments are `form`, `length`, and `container`.

```ipython3
ak.from_buffers(form, length, container)
```

## Minimizing the size of the output buffers

The [`ak.to_buffers()`](sphinx-llm:0f37278395f642a289396d6393d4d837#ak.to_buffers)/[`ak.from_buffers()`](sphinx-llm:fba1faa82f7649e28003bf8ebdda86f8#ak.from_buffers) functions exactly preserve an array, warts and all. Often, you’ll want to only write [`ak.to_packed()`](sphinx-llm:e05e976068e942e983b7102df31bdca8#ak.to_packed) arrays. “Packing” replaces an array structure with an equivalent structure that has no unreachable elements—data that you can’t see as part of the array, and therefore probably don’t want to write.

Here is an example of an array in need of packing:

```ipython3
unpacked = ak.Array(
    ak.contents.ListArray(
        ak.index.Index64(np.array([4, 10, 1])),
        ak.index.Index64(np.array([7, 10, 3])),
        ak.contents.NumpyArray(np.array([999, 4.4, 5.5, 999, 1.1, 2.2, 3.3, 999])),
    )
)
unpacked
```

This [`ak.contents.ListArray`](sphinx-llm:e2e74f5b6c644d858b90eaa78cd4dedc#ak.contents.ListArray) is in a strange order and the `999` values are unreachable. (Also, using `starts[1] == stops[1] == 10` to represent an empty list is a little odd, though allowed by the specification.)

The [`ak.to_buffers()`](sphinx-llm:0f37278395f642a289396d6393d4d837#ak.to_buffers) function dutifully writes the `999` values into the output, even though they’re not visible in the array.

```ipython3
ak.to_buffers(unpacked)
```

If the intended purpose of calling [`ak.to_buffers()`](sphinx-llm:0f37278395f642a289396d6393d4d837#ak.to_buffers) is to write to a file or send data over a network, this is wasted space. It can be trimmed by calling the [`ak.to_packed()`](sphinx-llm:e05e976068e942e983b7102df31bdca8#ak.to_packed) function.

```ipython3
packed = ak.to_packed(unpacked)
packed
```

At high-level, the array appears to be the same, but its low-level structure is quite different:

```ipython3
unpacked.layout
```

```ipython3
packed.layout
```

This version of the array is more concise when written with [`ak.to_buffers()`](sphinx-llm:0f37278395f642a289396d6393d4d837#ak.to_buffers):

```ipython3
ak.to_buffers(packed)
```

## Saving Awkward Arrays to HDF5

The [h5py](https://www.h5py.org/) library presents each group in an HDF5 file as a [`collections.abc.MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping), which we can use as a container for an array-set. We must also save the `form` and `length` as metadata for the array to be retrievable.

```ipython3
file = h5py.File("/tmp/example.hdf5", "w")
group = file.create_group("awkward")
group
```

We can fill this `group` as a `container` by passing it in to [`ak.to_buffers()`](sphinx-llm:0f37278395f642a289396d6393d4d837#ak.to_buffers). (See the previous section for more on [`ak.to_packed()`](sphinx-llm:e05e976068e942e983b7102df31bdca8#ak.to_packed).)

```ipython3
form, length, container = ak.to_buffers(ak.to_packed(ak_array), container=group)
```

```ipython3
container
```

Now the HDF5 group has been filled with array pieces.

```ipython3
container.keys()
```

Here’s one.

```ipython3
np.asarray(container["node0-offsets"])
```

Now we need to add the other information to the group as metadata. Since HDF5 accepts string-valued metadata, we can put it all in as JSON or numbers.

```ipython3
group.attrs["form"] = form.to_json()
group.attrs["form"]
```

```ipython3
group.attrs["length"] = length
group.attrs["length"]
```

## Reading Awkward Arrays from HDF5

With that, we can reconstitute the array by supplying [`ak.from_buffers()`](sphinx-llm:fba1faa82f7649e28003bf8ebdda86f8#ak.from_buffers) the right arguments from the group and metadata.

The group can’t be used as a `container` as-is, since subscripting it returns `h5py.Dataset` objects, rather than arrays.

```ipython3
reconstituted = ak.from_buffers(
    ak.forms.from_json(group.attrs["form"]),
    group.attrs["length"],
    {k: np.asarray(v) for k, v in group.items()},
)
reconstituted
```
