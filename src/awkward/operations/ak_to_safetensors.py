# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import json

import awkward as ak
from awkward._dispatch import high_level_function

__all__ = ("to_safetensors",)


@high_level_function()
def to_safetensors(
    array,
    destination,
    *,
    # ak.to_buffers kwargs
    container=None,
    buffer_key="{form_key}-{attribute}",
    form_key="node{id}",
    id_start=0,
    backend=None,
    byteorder=ak._util.native_byteorder,
):
    """Serialize an Awkward Array to the safetensors format and write it to `destination`.

    Ref: https://huggingface.co/docs/safetensors/.

    This function converts the provided Awkward Array (or array-like object) into raw
    buffers via `ak.to_buffers` and stores them in the safetensors format. Buffer names
    are generated from `buffer_key` and `form_key` templates, allowing downstream
    compatibility or layout reuse.
     The resulting safetensors file includes metadata containing the Awkward `form` and
     array `length`, which are required for `ak.from_safetensors` to reconstruct the array.
    Args:
        array: An Awkward Array or array-like object to serialize.
        destination (str | pathlib.Path | file-like): Path or writable binary stream
            where the safetensors file will be written.
        container (dict, optional): Optional mapping to receive the generated buffer
            bytes. If None (default), a temporary container is used and discarded
            after writing.
        buffer_key (str, optional): Format string for naming buffers. May include
            `{form_key}` and `{attribute}` placeholders. Defaults to
            `"{form_key}-{attribute}"`.
        form_key (str, optional): Format string for node forms when generating buffer
            keys. Typically includes `"{id}"`. Defaults to `"node{id}"`.
        id_start (int, optional): Starting index for node numbering. Defaults to `0`.
        backend (str | object, optional): Backend used to convert array data into
            buffers. If None, the default backend is used.
        byteorder (str, optional): Byte order for numeric buffers. Defaults to the
            system’s native byte order.

    Returns:
        None
            This function writes the safetensors file to `destination`. If
        `container` is provided, it will be populated with the raw buffer bytes.

    Raises:
        ValueError: If `destination` is not writable or otherwise invalid.
        TypeError: If `array` is not an Awkward Array or an object convertible to one.
        RuntimeError: If an error occurs while serializing buffers or writing the file.

    Example:

        >>> import awkward as ak
        >>> arr = ak.Array([[1, 2, 3], [], [4]])
        >>> ak.to_safetensors(arr, "out.safetensors")


    See also:
        ak.from_safetensors
    """
    # Implementation
    return _impl(
        array,
        destination,
        container,
        buffer_key,
        form_key,
        id_start,
        backend,
        byteorder,
    )


def _impl(
    array,
    destination,
    container,
    buffer_key,
    form_key,
    id_start,
    backend,
    byteorder,
):
    try:
        from safetensors.numpy import save_file
    except ImportError as err:
        raise ImportError(
            """to use ak.to_safetensors, you must install the 'safetensors' package with:

        pip install safetensors
or
        conda install -c huggingface safetensors"""
        ) from err

    form, length, buffers = ak.ak_to_buffers._impl(
        array,
        container=container,
        buffer_key=buffer_key,
        form_key=form_key,
        id_start=id_start,
        backend=backend,
        byteorder=byteorder,
    )

    metadata = {
        "form": form.to_json(),
        "length": str(length),
    }
    # save
    try:
      save_file(buffers, destination, metadata)
    except Exception as err:
      raise RuntimeError(
          f"Failed to write safetensors file to '{destination}': {err}"
      ) from err
