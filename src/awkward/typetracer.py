# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from collections.abc import Callable, Mapping

import awkward.forms
from awkward._backends.typetracer import TypeTracerBackend
from awkward._do import touch_data as _touch_data
from awkward._layout import HighLevelContext, wrap_layout
from awkward._nplikes.numpy import NumpyMetadata
from awkward._nplikes.placeholder import PlaceholderArray
from awkward._nplikes.shape import unknown_length
from awkward._nplikes.typetracer import (
    MaybeNone,
    OneOf,
    TypeTracerArray,
    TypeTracerReport,
    is_unknown_array,
    is_unknown_scalar,
)
from awkward._nplikes.typetracer import (
    typetracer_with_report as _typetracer_with_report,
)
from awkward._typing import TYPE_CHECKING, Any, TypeVar
from awkward.contents import Content
from awkward.forms import Form
from awkward.forms.form import regularize_buffer_key
from awkward.highlevel import Array, Record
from awkward.types.numpytype import is_primitive

if TYPE_CHECKING:
    from numpy.typing import DTypeLike  # noqa: TID251

__all__ = [
    "is_unknown_array",
    "is_unknown_scalar",
    "TypeTracerReport",
    "typetracer_with_report",
    "PlaceholderArray",
    "unknown_length",
    "touch_data",
    "TypeTracerArray",
    "MaybeNone",
    "OneOf",
    "create_unknown_scalar",
]


np = NumpyMetadata

T = TypeVar("T", Array, Record)


def _length_0_1_if_typetracer(
    array,
    function,
    highlevel: bool,
    behavior: Mapping | None,
    attrs: Mapping[str, Any],
) -> T:
    typetracer_backend = TypeTracerBackend.instance()

    if type(array).__module__.startswith("dask_awkward."):
        raise TypeError(
            "ak.typetracer._length_0_1_if_typetracer cannot take a dask-awkward Array; pass its '_meta' instead"
        )

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(
            array,
            allow_unknown=False,
            allow_record=True,
            primitive_policy="error",
            none_policy="error",
            string_policy="as-characters",
        )

    if layout.backend is typetracer_backend:
        _touch_data(layout)
        layout = function(layout.form, highlevel=False)

    return ctx.wrap(layout, highlevel=highlevel)


def length_zero_if_typetracer(
    array: Any,
    *,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> T:
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Recursively touches the data of an array, before returning a length-zero
    NumPy-backed iff. the given array has a typetracer backend; otherwise, a
    shallow copy of the original array is returned.
    """
    return _length_0_1_if_typetracer(
        array, Form.length_zero_array, highlevel, behavior, attrs
    )


def length_one_if_typetracer(
    array: Any,
    *,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array | Record:
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Recursively touches the data of an array, before returning a length-one
    NumPy-backed iff. the given array has a typetracer backend; otherwise, a
    shallow copy of the original array is returned.
    """
    return _length_0_1_if_typetracer(
        array, Form.length_one_array, highlevel, behavior, attrs
    )


def touch_data(
    array: Any,
    *,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array | Record:
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Recursively touches the data and returns a shall copy of the given array.
    """

    if type(array).__module__.startswith("dask_awkward."):
        raise TypeError(
            "ak.typetracer.touch_data cannot take a dask-awkward Array; pass its '_meta' instead"
        )

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(
            array,
            allow_unknown=False,
            allow_record=True,
            primitive_policy="error",
            none_policy="error",
            string_policy="as-characters",
        )

    _touch_data(layout)
    return ctx.wrap(layout, highlevel=highlevel)


def typetracer_with_report(
    form: Form | str | Mapping,
    *,
    buffer_key: str | Callable = "{form_key}",
    highlevel: bool = False,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> tuple[Content, TypeTracerReport]:
    """
    Args:
        form (#ak.forms.Form or str/dict equivalent): The form of the Awkward
            Array to build a typetracer-backed array from.
        buffer_key (str or callable): Python format string containing
            `"{form_key}"` and/or `"{attribute}"` or a function that takes these
            as keyword arguments and returns a string to use as a `form_key`
            for low-level typetracer buffers.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Returns a typetracer array and associated report object built from a form
    with labelled form keys.
    """
    if isinstance(form, str):
        if is_primitive(form):
            form = awkward.forms.NumpyForm(form)
        else:
            form = awkward.forms.from_json(form)
    elif isinstance(form, dict):
        form = awkward.forms.from_dict(form)
    elif not isinstance(form, awkward.forms.Form):
        raise TypeError(
            "'form' argument must be a Form or its Python dict/JSON string representation"
        )

    getkey = regularize_buffer_key(buffer_key)

    layout, report = _typetracer_with_report(form, getkey=getkey)
    return wrap_layout(
        layout, behavior=behavior, highlevel=highlevel, attrs=attrs
    ), report


def typetracer_from_form(
    form: Form | str | Mapping,
    *,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array | Content:
    """
    Args:
        form (#ak.forms.Form or str/dict equivalent): The form of the Awkward
            Array to build a typetracer-backed array from.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Returns a typetracer array built from a form.
    """
    if isinstance(form, str):
        if is_primitive(form):
            form = awkward.forms.NumpyForm(form)
        else:
            form = awkward.forms.from_json(form)
    elif isinstance(form, Mapping):
        form = awkward.forms.from_dict(form)
    elif not isinstance(form, awkward.forms.Form):
        raise TypeError(
            "'form' argument must be a Form or its Python dict/JSON string representation"
        )

    layout = form.length_zero_array().to_typetracer(forget_length=True)

    return wrap_layout(layout, behavior=behavior, highlevel=highlevel, attrs=attrs)


def create_unknown_scalar(dtype: DTypeLike) -> TypeTracerArray:
    return TypeTracerArray._new(np.dtype(dtype), shape=())
