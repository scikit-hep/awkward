# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function

__all__ = ("from_safetensors",)


@high_level_function()
def from_safetensors(
    source,
    *,
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
    """Load a safetensors file as an Awkward Array.

    Ref: https://huggingface.co/docs/safetensors/.

    This function reads data serialized in the safetensors format and constructs an
    Awkward Array (or low-level layout) from it. Buffers in the safetensors
    file are mapped to Awkward buffers using the `buffer_key` template, and optional
    behavior/attributes can be attached to the returned array.

    Optionally the result can be "virtual" (lazily referencing buffers rather than materializing them immediately).

    Args:
        source (str | os.PathLike | bytes | file-like): Path to a .safetensors file,
            raw bytes containing safetensors data, or a file-like object supporting
            read/seek.
        virtual (bool): If True, create a virtual (lazy) Awkward Array that references
            buffers without immediately materializing them. Defaults to False.
        buffer_key (str): Template used to construct buffer names for ak.from_buffers.
            The template may include the placeholders "{form_key}" (the safetensors form
            key) and "{attribute}" (the tensor attribute, e.g. "data" or a named field).
            Defaults to "{form_key}-{attribute}".
        backend (str): Backend identifier used to interpret raw buffers (for example
            "cpu" or a GPU backend string). Defaults to "cpu".
        byteorder (str): Byte order to assume when interpreting raw tensor bytes.
            Use "<" for little-endian (default) or ">" for big-endian.
        allow_noncanonical_form (bool): If True, attempt to convert non-canonical
            safetensors forms into a canonical Awkward form. If False, raise when
            a direct mapping is not possible. Defaults to False.
        highlevel (bool): If True, return a high-level ak.Array. If False, return the
            low-level Awkward layout object. Defaults to True.
        behavior (Mapping | None): Optional behavior mapping applied to the returned
            high-level array (see Awkward's behavior mechanism). If None, the default
            behavior is used.
        attrs (Mapping | None): Optional dictionary of attributes (metadata) to attach
            to the returned array; useful for preserving safetensors file metadata.

    Returns:
        ak.Array or ak.layout.Content: A high-level Awkward Array if `highlevel` is True,
        otherwise the corresponding low-level Awkward layout object. The array contains
        data reconstructed from the safetensors tensors; buffer names follow `buffer_key`.

    Raises:
        ValueError: If `byteorder` is not one of "<" or ">".
        FileNotFoundError: If `source` is a path that does not exist.
        TypeError: If `source` is not a supported type (neither path/bytes nor file-like).
        RuntimeError: If the safetensors data is malformed or tensors cannot be mapped to
            Awkward forms and `allow_noncanonical_form` is False.

    Here is a simple example:

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
        import safetensors
    except ImportError as err:
        raise ImportError(
            """to use ak.from_tensorflow, you must install the 'safetensors' package with:

        pip install safetensors
or
        conda install -c huggingface safetensors"""
        ) from err

    buffers = {}
    wrap = lambda x: (lambda: x) if virtual else x  # noqa: E731 # pylint: disable=C3001
    with safetensors.safe_open(source, framework="np") as f:
        metadata = f.metadata()
        for k in f.offset_keys():
            buffers[k] = wrap(f.get_tensor(k))

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
        highlevel=highlevel,
        behavior=behavior,
        attrs=attrs,
    )
