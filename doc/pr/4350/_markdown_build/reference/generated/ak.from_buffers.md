# ak.from_buffers

Defined in [awkward.operations.ak_from_buffers](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_from_buffers.py) on [line 29](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_from_buffers.py#L29).

#### ak.from_buffers(form, length, container, buffer_key='{form_key}-{attribute}', \*, backend='cpu', byteorder=ak._util.native_byteorder, allow_noncanonical_form=False, enable_virtualarray_caching=True, highlevel=True, behavior=None, attrs=None)

Reconstitutes an Awkward Array from a Form, length, and memory buffers.

The first three arguments of this function are the return values of
[`ak.to_buffers`](sphinx-llm:0f37278395f642a289396d6393d4d837#ak.to_buffers), so a full round-trip is

```pycon
>>> reconstituted = ak.from_buffers(*ak.to_buffers(original))
```

The `container` argument lets you specify your own Mapping, which might be
an interface to some storage format or device (e.g. h5py). It’s okay if
the `container` dropped NumPy’s `dtype` and `shape` information, leaving
raw bytes, since `dtype` and `shape` can be reconstituted from the
[`ak.forms.NumpyForm`](sphinx-llm:e462c43e131e42ddbc3ae998cd272d1f#ak.forms.NumpyForm).
If the values of `container` are recognised as arrays by the given backend,
a view over their existing data will be used, where possible.
The `container` values are allowed to be callables with no arguments.
If that’s the case, they will be turned into `VirtualNDArray` buffers whose generator
function is the callable and is used to materialize the buffer when required.

The `buffer_key` should be the same as the one used in [`ak.to_buffers`](sphinx-llm:0f37278395f642a289396d6393d4d837#ak.to_buffers).

When `allow_noncanonical_form` is set to True, this function readily accepts
non-simplified forms, i.e. forms which will be simplified by Awkward Array
into “canonical” representations, e.g. `option[option[...]]` → `option[...]`.
Such forms can be produced by the low-level ArrayBuilder `snapshot()` method.
Given that Awkward Arrays must have canonical layouts, it follows that
invoking this function with `allow_noncanonical_form` may produce arrays
whose forms differ to the input form.

In order for a non-simplified form to be considered valid, it should be one
that the [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) layout classes could produce iff. the
simplification rules were removed.

See [`ak.to_buffers`](sphinx-llm:0f37278395f642a289396d6393d4d837#ak.to_buffers) for examples.

* **Parameters:**
  * **form** ([`ak.forms.Form`](sphinx-llm:6295086f75dd4713a4cbcfec3594d1a1#ak.forms.Form) or str/dict equivalent) – The form of the Awkward
    Array to reconstitute from named buffers.
  * **length** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Length of the array. (The output of this function is always
    single-partition.)
  * **container** (*Mapping* *,* *such as dict*) – The str → Python buffers that
    represent the decomposed Awkward Array. This `container` is only
    assumed to have a `__getitem__` method that accepts strings as keys.
  * **buffer_key** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *or* *callable*) – Python format string containing
    `"{form_key}"` and/or `"{attribute}"` or a function that takes these
    as keyword arguments and returns a string to use as a key for a buffer
    in the `container`.
  * **backend** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Library to use to generate values that are
    put into the new array. The default, cpu, makes NumPy
    arrays, which are in main memory (e.g. not GPU). If all the values in
    `container` have the same `backend` as this, they won’t be copied.
  * **byteorder** (`"<"`, `">"`) – Endianness of buffers read from `container`.
    If the byteorder does not match the current system byteorder, the
    arrays will be copied.
  * **allow_noncanonical_form** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, non-canonical forms will be
    simplified to produce arrays with canonical layouts; otherwise,
    an exception will be thrown for such forms.
  * **enable_virtualarray_caching** ([*bool*](https://docs.python.org/3/library/functions.html#bool) *or* *callable*) – If True (the default),
    all VirtualNDArray buffers that get created will cache their
    materialized buffers on themselves when they get materialized.
    If a callable is given, it must accept two arguments, `form_key` and `attribute`,
    and return a boolean indicating whether caching should be enabled for the
    given buffer. The `form_key` and `attribute` are the same as those passed to
    the `buffer_key` function. If False, all VirtualNDArrays will not cache
    their materialized buffers on themselves.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) reconstituted from a Form, length, and a collection of memory
  buffers, so that data can be losslessly read from file formats and storage
  devices that only map names to binary blobs (such as a filesystem directory).
