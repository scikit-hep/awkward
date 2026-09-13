# ak.from_safetensors

Defined in [awkward.operations.ak_from_safetensors](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_from_safetensors.py) on [line 13](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_from_safetensors.py#L13).

#### ak.from_safetensors(source, \*, storage_options=None, virtual=False, buffer_key='{form_key}-{attribute}', backend='cpu', byteorder='<', allow_noncanonical_form=False, highlevel=True, behavior=None, attrs=None)

Reads a safetensors file as an Awkward Array.

Ref: [https://huggingface.co/docs/safetensors/](https://huggingface.co/docs/safetensors/).

This function reads data serialized in the safetensors format and
reconstructs an Awkward Array (or low-level layout) from it. Buffers in the
safetensors file are mapped to Awkward buffers according to the
`buffer_key` template, and optional behavior or attributes can be attached
to the returned array.

The safetensors file **must contain** `form` and `length` entries in its
metadata, which define the structure and length of the reconstructed array.

* **Parameters:**
  * **source** (*path-like*) – Name of the input file, file path, or
    remote URL passed to [fsspec.core.url_to_fs](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.core.url_to_fs)
    for remote reading.
  * **storage_options** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Any additional options to pass to
    [fsspec.core.url_to_fs](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.core.url_to_fs)
    to open a remote file for reading.
  * **virtual** ([*bool*](https://docs.python.org/3/library/functions.html#bool) *,* *optional*) – If True, create a virtual (lazy) Awkward Array
    that references buffers without materializing them. Defaults to False.
  * **buffer_key** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – Template for buffer names, with placeholders
    `{form_key}` and `{attribute}`. Defaults to “{form_key}-{attribute}”.
  * **backend** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – Backend identifier (e.g., “cpu”). Defaults to “cpu”.
  * **byteorder** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – Byte order, “<” (little-endian, default) or “>”.
  * **allow_noncanonical_form** ([*bool*](https://docs.python.org/3/library/functions.html#bool) *,* *optional*) – 

    If True, normalize
    : safetensors forms that do not directly match Awkward. Defaults to False.

    highlevel (bool, optional): If True, return a high-level ak.Array. If False,
    : return the low-level layout. Defaults to True.

    behavior (Mapping | None, optional): Optional Awkward behavior mapping.
    attrs (Mapping | None, optional): Optional metadata to attach to the array.
* **Returns:**
  An Awkward Array (or layout) reconstructed
  from the safetensors buffers.
* **Return type:**
  [ak.Array](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) or ak.layout.Content

### Examples

```pycon
>>> import awkward as ak
>>> arr = ak.from_safetensors("out.safetensors")
>>> arr  
<Array [[1, 2, 3], [], [4]] type='3 * var * int64'>
```

Create a virtual (lazy) array that references buffers without materializing them:

```pycon
>>> virtual_arr = ak.from_safetensors("out.safetensors", virtual=True)
>>> virtual_arr  
<Array [??, ??, ??] type='3 * var * int64'>
```

See also [`ak.to_safetensors`](sphinx-llm:45b754adee5d45e79a1013eeaf627232#ak.to_safetensors).
