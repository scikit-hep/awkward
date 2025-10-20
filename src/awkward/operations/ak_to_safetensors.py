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
    buffers (via ak.to_buffers) and stores them in the safetensors layout. Buffer
    names are generated from `buffer_key` and `form_key` and can be controlled to
    match downstream consumers or to allow reuse of existing serialization layouts.
     The resulting safetensors file includes metadata containing the Awkward `form` and
     array `length`, which are required for `ak.from_safetensors` to reconstruct the array.
    Args:
        array: An Awkward Array or array-like object to serialize.
        destination (str or pathlib.Path or file-like): Path to write the resulting
            safetensors file, or a writable binary file-like object.
        container (dict, optional): A mapping to receive the raw buffers produced by
            ak.to_buffers. If provided, the function will populate this dict with
            {buffer_key: bytes} entries. If None (the default), a temporary container
            is used for writing and discarded after the file is created.
        buffer_key (str, optional): A format string used to name each buffer in the
            safetensors container. May include placeholders such as
            "{form_key}" and "{attribute}" (default: "{form_key}-{attribute}").
        form_key (str, optional): A format string used to name node forms when
            generating buffer keys; typically contains an "{id}" placeholder
            (default: "node{id}").
        id_start (int, optional): Starting index for node numbering used by `form_key`
            (default: 0).
        backend (str or object, optional): Backend selection passed through to
            ak.to_buffers for converting array data to raw buffers (e.g., numpy
            backend or an Awkward array backend object). If None, the library's
            default backend is used.
        byteorder (str, optional): Byte order to use when encoding numeric buffers.
            Defaults to the system native byteorder.

    Returns:
        None: The function writes the safetensors file to `destination`. If a
        `container` was provided, it will be populated with the generated buffer
        bytes and can be inspected or reused by the caller.

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
        "length": json.dumps(length),
    }
    # save
    save_file(buffers, destination, metadata)
