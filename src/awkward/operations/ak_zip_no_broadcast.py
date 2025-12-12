# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import reduce
from typing import Any

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext, ensure_same_backend
from awkward._namedaxis import _get_named_axis, _unify_named_axis
from awkward._nplikes.placeholder import PlaceholderArray
from awkward._nplikes.shape import UnknownLength

__all__ = ("zip_no_broadcast",)


@high_level_function()
def zip_no_broadcast(
    arrays: Mapping[str, Any] | Sequence[Any],
    *,
    parameters: dict | None = None,
    with_name: str | None = None,
    highlevel: bool = True,
    behavior: dict | None = None,
    attrs: dict | None = None,
):
    """
    Combine `arrays` into a single structure as the fields of a collection
    of records or the slots of a collection of tuples **without broadcasting**.

    This function is similar to :func:`ak.zip` but does *not* attempt to
    broadcast inputs together. It therefore requires that the provided
    arrays already share compatible layouts and lengths.

    Parameters
    ----------
    arrays
        Mapping or sequence of array-like objects that ``ak.to_layout``
        recognizes.
        If a mapping is provided, its keys are used as field names in
        the resulting record array.
        If a sequence is provided, a tuple-like record (no field names)
        is produced.
    parameters
        Optional parameters for the resulting RecordArray node.
    with_name
        If given, sets the ``__record__`` parameter on the result.
    highlevel
        If True, return an :class:`ak.Array`; otherwise return a low-level
        :class:`ak.contents.Content` subclass.
    behavior
        Optional behavior for the output array when ``highlevel`` is True.
    attrs
        Optional attributes for the output array when ``highlevel`` is True.

    Notes
    -----
    Only two kinds of layouts are supported:
    :class:`ak.contents.NumpyArray` and
    :class:`ak.contents.ListOffsetArray` (whose contents must be
    ``NumpyArray``). All inputs must have the same length. When
    ListOffsetArray inputs are provided, their offsets must be identical.
    """
    # Dispatch: provide the underlying arrays to the dispatcher
    if isinstance(arrays, Mapping):
        yield arrays.values()
    else:
        yield arrays

    return _impl(arrays, parameters, with_name, highlevel, behavior, attrs)


def _impl(
    arrays: Mapping[str, Any] | Sequence[Any],
    parameters: dict | None,
    with_name: str | None,
    highlevel: bool,
    behavior: dict | None,
    attrs: dict | None,
):
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        if isinstance(arrays, Mapping):
            values = list(arrays.values())
            layouts = ensure_same_backend(
                *(
                    ctx.unwrap(
                        x,
                        allow_record=False,
                        allow_unknown=False,
                        none_policy="pass-through",
                        primitive_policy="pass-through",
                    )
                    for x in values
                )
            )
            fields = list(arrays.keys())
            out_named_axis = (
                reduce(_unify_named_axis, map(_get_named_axis, values))
                if values
                else None
            )
        else:
            seq = list(arrays)
            layouts = ensure_same_backend(
                *(
                    ctx.unwrap(
                        x,
                        allow_record=False,
                        allow_unknown=False,
                        none_policy="pass-through",
                        primitive_policy="pass-through",
                    )
                    for x in seq
                )
            )
            fields = None
            out_named_axis = (
                reduce(_unify_named_axis, map(_get_named_axis, seq)) if seq else None
            )

    if not layouts:
        raise ValueError("zip_no_broadcast requires at least one array")

    backend = layouts[0].backend

    if with_name is not None:
        parameters = dict(parameters) if parameters is not None else {}
        parameters["__record__"] = with_name

    if all(isinstance(layout, ak.contents.NumpyArray) for layout in layouts):
        length = _check_equal_lengths(layouts)
        out = ak.contents.RecordArray(
            layouts,
            fields,
            length=length,
            parameters=parameters,
            backend=backend,
        )

    elif all(isinstance(layout, ak.contents.ListOffsetArray) for layout in layouts):
        contents: list[ak.contents.Content] = []
        for layout in layouts:
            if not isinstance(layout.content, ak.contents.NumpyArray):
                msg = (
                    "ak.zip_no_broadcast cannot (safely) zip ListOffsetArrays whose "
                    "contents are not NumpyArray. Use ak.zip instead."
                )
                raise NotImplementedError(msg)
            contents.append(layout.content)

        if backend.name == "typetracer":
            offsets = layouts[0].offsets
        else:
            comparable_offsets = [
                layout.offsets
                for layout in layouts
                if not isinstance(layout.offsets, PlaceholderArray)
            ]

            if comparable_offsets:
                first = comparable_offsets[0]
                if not all(
                    first.nplike.all(other.data == first.data)
                    for other in comparable_offsets
                ):
                    raise ValueError("all ListOffsetArrays must have the same offsets")
                offsets = first
            else:
                offsets = layouts[0].offsets

        length = _check_equal_lengths(contents)
        out = ak.contents.ListOffsetArray(
            offsets=offsets,
            content=ak.contents.RecordArray(
                contents,
                fields,
                length=length,
                parameters=parameters,
                backend=backend,
            ),
        )
    else:
        raise ValueError(
            "all array layouts must be either NumpyArrays or ListOffsetArrays"
        )

    wrapped_out = ctx.wrap(out, highlevel=highlevel)

    return ak.operations.ak_with_named_axis._impl(
        wrapped_out,
        named_axis=out_named_axis,
        highlevel=highlevel,
        behavior=ctx.behavior,
        attrs=ctx.attrs,
    )


def _check_equal_lengths(
    contents: Sequence[ak.contents.Content],
) -> int | UnknownLength:
    """
    Ensure all layouts in ``contents`` have the same length and return that
    length.

    ``UnknownLength`` is returned when lengths are not statically known
    (typetracer/placeholder scenarios).
    """
    if not contents:
        raise ValueError("_check_equal_lengths requires at least one content")

    length = contents[0].length
    for layout in contents:
        if layout.length != length:
            raise ValueError("all arrays must have the same length")

    return length
