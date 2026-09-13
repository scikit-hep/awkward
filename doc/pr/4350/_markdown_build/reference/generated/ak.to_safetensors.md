# ak.to_safetensors

Defined in [awkward.operations.ak_to_safetensors](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_to_safetensors.py) on [line 14](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_to_safetensors.py#L14).

#### ak.to_safetensors(array, destination, \*, storage_options=None, container=None, buffer_key='{form_key}-{attribute}', form_key='node{id}', id_start=0, backend=None, byteorder=ak._util.native_byteorder)

Writes an Awkward Array to a safetensors file.

If `container` is provided, it is populated with the raw buffer bytes.

Ref: [https://huggingface.co/docs/safetensors/](https://huggingface.co/docs/safetensors/).

This function converts the provided Awkward Array (or array-like object) into raw
buffers via `ak.to_buffers` and stores them in the safetensors format. Buffer names
are generated from `buffer_key` and `form_key` templates, allowing downstream
compatibility or layout reuse.
The resulting safetensors file includes metadata containing the Awkward `form` and
array `length`, which are required for `ak.from_safetensors` to reconstruct the array.

See also [`ak.from_safetensors`](sphinx-llm:0666a66217314390b12f2c6b3d896845#ak.from_safetensors).

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **destination** (*path-like*) – Name of the output file, file path, or
    remote URL passed to [fsspec.core.url_to_fs](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.core.url_to_fs)
    for remote writing.
  * **storage_options** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Any additional options to pass to
    [fsspec.core.url_to_fs](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.core.url_to_fs)
    to open a remote file for writing.
  * **container** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict) *,* *optional*) – Optional mapping to receive the generated buffer
    bytes. If None (default), a temporary container is used and discarded
    after writing.
  * **buffer_key** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – Format string for naming buffers. May include
    `{form_key}` and `{attribute}` placeholders. Defaults to
    `"{form_key}-{attribute}"`.
  * **form_key** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – Format string for node forms when generating buffer
    keys. Typically includes `"{id}"`. Defaults to `"node{id}"`.
  * **id_start** ([*int*](https://docs.python.org/3/library/functions.html#int) *,* *optional*) – Starting index for node numbering. Defaults to `0`.
  * **backend** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *|* [*object*](https://docs.python.org/3/library/functions.html#object) *,* *optional*) – Backend used to convert array data into
    buffers. If None, the default backend is used.
  * **byteorder** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – Byte order for numeric buffers. Defaults to the
    system’s native byte order.
* **Returns:**
  None. The contents of `array` are written to `destination` in the
  safetensors format.

### Examples

```pycon
>>> import awkward as ak
>>> arr = ak.Array([[1, 2, 3], [], [4]])
>>> ak.to_safetensors(arr, "out.safetensors")
```
