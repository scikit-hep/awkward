# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import copy
from collections.abc import Mapping, MutableMapping
from numbers import Integral

import awkward as ak
from awkward._backends import Backend
from awkward.contents.content import ActionType, AxisMaybeNone, Content
from awkward.forms import form
from awkward.record import Record
from awkward.typing import Any

np = ak._nplikes.NumpyMetadata.instance()


def recursively_apply(
    layout: Content | Record,
    action: ActionType,
    behavior: dict | None = None,
    depth_context: dict[str, Any] | None = None,
    lateral_context: dict[str, Any] | None = None,
    allow_records: bool = True,
    keep_parameters: bool = True,
    numpy_to_regular: bool = True,
    return_simplified: bool = True,
    return_array: bool = True,
    function_name: str | None = None,
) -> Content | Record | None:

    if isinstance(layout, Content):
        return layout._recursively_apply(
            action,
            behavior,
            1,
            copy.copy(depth_context),
            lateral_context,
            {
                "allow_records": allow_records,
                "keep_parameters": keep_parameters,
                "numpy_to_regular": numpy_to_regular,
                "return_simplified": return_simplified,
                "return_array": return_array,
                "function_name": function_name,
            },
        )

    elif isinstance(layout, Record):
        out = recursively_apply(
            layout._array,
            action,
            behavior,
            depth_context,
            lateral_context,
            allow_records,
            keep_parameters,
            numpy_to_regular,
            return_simplified,
            return_array,
            function_name,
        )

        if return_array:
            return Record(out, layout.at)
        else:
            return None


def to_buffers(
    content: Content,
    container: MutableMapping[str, Any] | None = None,
    buffer_key="{form_key}-{attribute}",
    form_key: str | None = "node{id}",
    id_start: Integral = 0,
    backend: Backend = None,
) -> tuple[form.Form, int, Mapping[str, Any]]:
    if container is None:
        container = {}
    if backend is None:
        backend = content._backend
    if not backend.nplike.known_data:
        raise ak._errors.wrap_error(
            TypeError("cannot call 'to_buffers' on an array without concrete data")
        )

    if isinstance(buffer_key, str):

        def getkey(layout, form, attribute):
            return buffer_key.format(form_key=form.form_key, attribute=attribute)

    elif callable(buffer_key):

        def getkey(layout, form, attribute):
            return buffer_key(
                form_key=form.form_key,
                attribute=attribute,
                layout=layout,
                form=form,
            )

    else:
        raise ak._errors.wrap_error(
            TypeError(
                "buffer_key must be a string or a callable, not {}".format(
                    type(buffer_key)
                )
            )
        )

    if form_key is None:
        raise ak._errors.wrap_error(
            TypeError(
                "a 'form_key' must be supplied, to match Form elements to buffers in the 'container'"
            )
        )

    form = content.form_with_key(form_key=form_key, id_start=id_start)

    content._to_buffers(form, getkey, container, backend)

    return form, len(content), container


def axis_wrap_if_negative(
    layout: Content | Record, axis: AxisMaybeNone
) -> AxisMaybeNone:
    if isinstance(layout, Record):
        if axis == 0:
            raise ak._errors.wrap_error(
                np.AxisError("Record type at axis=0 is a scalar, not an array")
            )
        return axis_wrap_if_negative(layout._array, axis)

    else:
        if axis is None or axis >= 0:
            return axis

        mindepth, maxdepth = layout.minmax_depth
        depth = layout.purelist_depth
        if mindepth == depth and maxdepth == depth:
            posaxis = depth + axis
            if posaxis < 0:
                raise ak._errors.wrap_error(
                    np.AxisError(
                        f"axis={axis} exceeds the depth ({depth}) of this array"
                    )
                )
            return posaxis

        elif mindepth + axis == 0:
            raise ak._errors.wrap_error(
                np.AxisError(
                    "axis={} exceeds the depth ({}) of at least one record field (or union possibility) of this array".format(
                        axis, depth
                    )
                )
            )

        return axis
