# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

__all__ = ("from_non_simplified_buffers",)

from awkward._dispatch import high_level_function
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpylike import NumpyMetadata
from awkward.operations.ak_from_buffers import from_buffers as from_buffers_impl

np = NumpyMetadata.instance()
numpy = Numpy.instance()


@high_level_function()
def from_non_simplified_buffers(
    form,
    length,
    container,
    buffer_key="{form_key}-{attribute}",
    *,
    backend="cpu",
    byteorder="<",
    highlevel=True,
    behavior=None,
):
    """
    Args:
        form (#ak.forms.Form or str/dict equivalent): The (maybe non-simplified)
            form of the Awkward Array to reconstitute from named buffers.
        length (int): Length of the array. (The output of this function is always
            single-partition.)
        container (Mapping, such as dict): The str \u2192 Python buffers that
            represent the decomposed Awkward Array. This `container` is only
            assumed to have a `__getitem__` method that accepts strings as keys.
        buffer_key (str or callable): Python format string containing
            `"{form_key}"` and/or `"{attribute}"` or a function that takes these
            as keyword arguments and returns a string to use as a key for a buffer
            in the `container`.
        backend (str): Library to use to generate values that are
            put into the new array. The default, cpu, makes NumPy
            arrays, which are in main memory (e.g. not GPU). If all the values in
            `container` have the same `backend` as this, they won't be copied.
        byteorder (`"<"`, `">"`): Endianness of buffers read from `container`.
            If the byteorder does not match the current system byteorder, the
            arrays will be copied.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Reconstitutes an Awkward Array from a Form, length, and a collection of memory
    buffers, so that data can be losslessly read from file formats and storage
    devices that only map names to binary blobs (such as a filesystem directory).

    Unlike #ak.from_buffers, this function readily accepts non-simplified forms,
    i.e. forms which will be simplified by Awkward Array into "canonical"
    representations, e.g. `option[option[...]]` → `option[...]`. As such, this
    function may produce an array whose form differs from the input form.

    The non-simplified forms that this function accepts can be produced by the
    low-level ArrayBuilder `snapshot()` method. In order for a non-simplified
    form to be considered valid, it should be one that the #ak.contents.Content
    layout classes could produce iff. the simplification rules were removed.

    For a complete description of the other arguments of this function, see
    #ak.from_buffers.

    See also #ak.from_buffers, and #ak.to_buffers for examples.
    """
    return from_buffers_impl(
        form,
        length,
        container,
        buffer_key,
        backend,
        byteorder,
        highlevel,
        behavior,
        True,
    )
