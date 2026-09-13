# ak.to_buffers

Defined in [awkward.operations.ak_to_buffers](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_to_buffers.py) on [line 15](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_to_buffers.py#L15).

#### ak.to_buffers(array, container=None, buffer_key='{form_key}-{attribute}', form_key='node{id}', \*, id_start=0, backend=None, byteorder=ak._util.native_byteorder)

Decomposes an Awkward Array into a Form, length, and memory buffers.

This function returns a 3-tuple:

```default
(form, length, container)
```

where the `form` is a [`ak.forms.Form`](sphinx-llm:6295086f75dd4713a4cbcfec3594d1a1#ak.forms.Form) (whose string representation is JSON),
the `length` is an integer (`len(array)`), and the `container` is either
the MutableMapping you passed in or a new dict containing the buffers (as
NumPy arrays).

These are also the first three arguments of [`ak.from_buffers`](sphinx-llm:fba1faa82f7649e28003bf8ebdda86f8#ak.from_buffers), so a full
round-trip is

```pycon
>>> reconstituted = ak.from_buffers(*ak.to_buffers(original))
```

The `container` argument lets you specify your own MutableMapping, which
might be an interface to some storage format or device (e.g. h5py). It’s
okay if the `container` drops NumPy’s `dtype` and `shape` information,
leaving raw bytes, since `dtype` and `shape` can be reconstituted from
the [`ak.forms.NumpyForm`](sphinx-llm:e462c43e131e42ddbc3ae998cd272d1f#ak.forms.NumpyForm).

The `buffer_key` and `form_key` arguments let you configure the names of the
buffers added to the `container` and string labels on each Form node, so that
the two can be uniquely matched later. `buffer_key` and `form_key` are distinct
arguments to allow for more indirection (buffer keys can differ from Form keys,
as long as there’s a way to map them to each other) and because some Form nodes,
such as [`ak.forms.ListForm`](sphinx-llm:bdc97ffbe3e241ae950070fee719260d#ak.forms.ListForm) and [`ak.forms.UnionForm`](sphinx-llm:d62266cf70594c33be3bb7c38a607847#ak.forms.UnionForm), have more than one attribute
(`starts` and `stops` for [`ak.forms.ListForm`](sphinx-llm:bdc97ffbe3e241ae950070fee719260d#ak.forms.ListForm) and `tags` and `index` for
[`ak.forms.UnionForm`](sphinx-llm:d62266cf70594c33be3bb7c38a607847#ak.forms.UnionForm)).

Awkward 1.x also included partition numbers (`"part0-"`, `"part1-"`, …) in
the buffer keys. In version 2.x onward, partitioning is handled externally by
Dask, but partition numbers can be emulated by prepending a fixed `"partN-"`
string to the `buffer_key`. The `array` represents exactly one partition.

If you intend to use this function for saving data, you may want to pack it
first with [`ak.to_packed`](sphinx-llm:e05e976068e942e983b7102df31bdca8#ak.to_packed).

See also [`ak.from_buffers`](sphinx-llm:fba1faa82f7649e28003bf8ebdda86f8#ak.from_buffers) and [`ak.to_packed`](sphinx-llm:e05e976068e942e983b7102df31bdca8#ak.to_packed).

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **container** (*None* *or* *MutableMapping*) – The str → NumPy arrays (or
    Python buffers) that represent the decomposed Awkward Array. This
    `container` is only assumed to have a `__setitem__` method that
    accepts strings as keys.
  * **buffer_key** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *or* *callable*) – Python format string containing
    `"{form_key}"` and/or `"{attribute}"` or a function that takes these
    (and/or `layout`) as keyword arguments and returns a string to use
    as a key for a buffer in the `container`. The `form_key` is the result
    of applying `form_key` (below), and the `attribute` is a hard-coded
    string representing the buffer’s function (e.g. `"data"`, `"offsets"`,
    `"index"`).
  * **form_key** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *callable*) – Python format string containing
    `"{id}"` or a function that takes this (and/or `layout`) as a keyword
    argument and returns a string to use as a key for a Form node.
    Together, the `buffer_key` and `form_key` links attributes of each Form
    node to data in the `container`.
  * **id_start** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Starting `id` to use in `form_key` and hence `buffer_key`.
    This integer increases in a depth-first walk over the `array` nodes and
    can be used to generate unique keys for each Form.
  * **backend** (`"cpu"`, `"cuda"`, `"jax"`, None) – Backend to use to
    generate values that are put into the `container`. The default,
    `"cpu"`, makes NumPy arrays, which are in main memory
    (e.g. not GPU) and satisfy Python’s Buffer protocol. If all the
    buffers in `array` have the same `backend` as this, they won’t be
    copied. If the backend is None, then the backend of the layout
    will be used to generate the buffers.
  * **byteorder** (`"<"`, `">"`) – Endianness of buffers written to `container`.
    If the byteorder does not match the current system byteorder, the
    arrays will be copied.
* **Returns:**
  A 3-tuple `(form, length, container)` decomposed from `array`,
  so that data can be losslessly written to file formats and storage devices
  that only map names to binary blobs (such as a filesystem directory).

### Examples

Here is a simple example:

```pycon
>>> original = ak.Array([[1, 2, 3], [], [4, 5]])
>>> form, length, container = ak.to_buffers(original)
>>> print(form)
{
    "class": "ListOffsetArray",
    "offsets": "i64",
    "content": {
        "class": "NumpyArray",
        "primitive": "int64",
        "form_key": "node1"
    },
    "form_key": "node0"
}
>>> length
3
>>> container
{'node0-offsets': array([0, 3, 3, 5]), 'node1-data': array([1, 2, 3, 4, 5])}
```

which may be read back with

```pycon
>>> ak.from_buffers(form, length, container)
<Array [[1, 2, 3], [], [4, 5]] type='3 * var * int64'>
```
