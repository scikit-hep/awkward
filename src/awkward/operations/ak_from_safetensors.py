# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import fsspec

import awkward as ak
from awkward._dispatch import high_level_function

__all__ = ("from_safetensors",)


@high_level_function()
def from_safetensors(
    source,
    *,
    storage_options=None,
    virtual=False,
    # ak.from_buffers kwargs
    buffer_key="{form_key}-{attribute}",
    backend="cpu",
    byteorder="<",
    allow_noncanonical_form=False,
    highlevel=True,
    behavior=None,
    attrs=None,
):
    """
    Args:
        source (path-like): Name of the input file, file path, or
            remote URL passed to [fsspec.core.url_to_fs](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.core.url_to_fs)
            for remote reading.
        storage_options (None or dict): Any additional options to pass to
            [fsspec.core.url_to_fs](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.core.url_to_fs)
            to open a remote file for reading.
        virtual (bool, optional): If True, create a virtual (lazy) Awkward Array
           that references buffers without materializing them. Defaults to False.
        buffer_key (str, optional): Template for buffer names, with placeholders
           `{form_key}` and `{attribute}`. Defaults to "{form_key}-{attribute}".
        backend (str, optional): Backend identifier (e.g., "cpu"). Defaults to "cpu".
        byteorder (str, optional): Byte order, "<" (little-endian, default) or ">".
        allow_noncanonical_form (bool, optional): If True, normalize
            safetensors forms that do not directly match Awkward. Defaults to False.
         highlevel (bool, optional): If True, return a high-level ak.Array. If False,
             return the low-level layout. Defaults to True.
         behavior (Mapping | None, optional): Optional Awkward behavior mapping.
         attrs (Mapping | None, optional): Optional metadata to attach to the array.

    Returns:
        ak.Array or ak.layout.Content: An Awkward Array (or layout) reconstructed
        from the safetensors buffers.

    Load a safetensors file as an Awkward Array.

    Ref: https://huggingface.co/docs/safetensors/.

    This function reads data serialized in the safetensors format and reconstructs
    an Awkward Array (or low-level layout) from it. Buffers in the safetensors file
    are mapped to Awkward buffers according to the `buffer_key` template, and
    optional behavior or attributes can be attached to the returned array.

    The safetensors file **must contain** `form` and `length` entries in its
    metadata, which define the structure and length of the reconstructed array.

    Example:

        >>> import awkward as ak
        >>> arr = ak.from_safetensors("out.safetensors")
        >>> arr  # doctest: +SKIP
        <Array [[1, 2, 3], [], [4]] type='3 * var * int64'>

        Create a virtual (lazy) array that references buffers without materializing them:

        >>> virtual_arr = ak.from_safetensors("out.safetensors", virtual=True)
        >>> virtual_arr  # doctest: +SKIP
        <Array [??, ??, ??] type='3 * var * int64'>


    See also #ak.to_safetensors.
    """
    # Implementation
    return _impl(
        source,
        storage_options,
        virtual,
        buffer_key,
        backend,
        byteorder,
        allow_noncanonical_form,
        highlevel,
        behavior,
        attrs,
    )


def _impl(
    source,
    storage_options,
    virtual,
    buffer_key,
    backend,
    byteorder,
    allow_noncanonical_form,
    highlevel,
    behavior,
    attrs,
):
    try:
        from safetensors import _safe_open_handle
    except ImportError as err:
        raise ImportError(
            """to use ak.from_tensorflow, you must install the 'safetensors' package with:

        pip install safetensors
or
        conda install -c huggingface safetensors"""
        ) from err

    fs, source = fsspec.core.url_to_fs(source, **(storage_options or {}))

    buffers = {}

    def maybe_virtualize(x):
        return (lambda: x) if virtual else x

    with fs.open(source, "rb") as f:
        with _safe_open_handle(f, framework="np") as g:
            metadata = g.metadata()
            for k in g.offset_keys():
                buffers[k] = maybe_virtualize(g.get_tensor(k))

    if "form" not in metadata or "length" not in metadata:
        raise RuntimeError(
            "Missing required metadata in safetensors file: 'form' and 'length' are required."
        )
    form = ak.forms.from_json(metadata["form"])
    length = int(metadata["length"])

    # reconstruct array
    return ak.ak_from_buffers._impl(
        form,
        length,
        buffers,
        buffer_key=buffer_key,
        backend=backend,
        byteorder=byteorder,
        simplify=allow_noncanonical_form,
        enable_virtualarray_caching=True,
        highlevel=highlevel,
        behavior=behavior,
        attrs=attrs,
    )
