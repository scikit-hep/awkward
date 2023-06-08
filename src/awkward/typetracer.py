from __future__ import annotations

__all__ = [
    "is_unknown_array",
    "is_unknown_scalar",
    "empty_if_typetracer",
    "TypeTracerReport",
    "typetracer_with_report",
    "PlaceholderArray",
    "unknown_length",
]

from awkward._backends.typetracer import TypeTracerBackend
from awkward._behavior import behavior_of
from awkward._errors import deprecate
from awkward._layout import wrap_layout
from awkward._nplikes.placeholder import PlaceholderArray
from awkward._nplikes.shape import unknown_length
from awkward._nplikes.typetracer import (
    TypeTracerReport,
    is_unknown_array,
    is_unknown_scalar,
    typetracer_with_report,
)
from awkward._typing import TypeVar
from awkward.forms import Form
from awkward.highlevel import Array, Record
from awkward.operations.ak_to_layout import to_layout

T = TypeVar("T", Array, Record)


def _length_0_1_if_typetracer(array: T, function):
    typetracer_backend = TypeTracerBackend.instance()

    layout = to_layout(array, allow_other=False)
    behavior = behavior_of(array)

    if layout.backend is typetracer_backend:
        layout._touch_data(True)
        layout = function(layout.form, highlevel=False)

    return wrap_layout(layout, behavior=behavior)


def empty_if_typetracer(array: T) -> T:
    deprecate(
        "'empty_if_typetracer' is being replaced by 'length_zero_if_typetracer' (change name)",
        "2.4.0",
    )

    return length_zero_if_typetracer(array)


def length_zero_if_typetracer(array: T) -> T:
    return _length_0_1_if_typetracer(array, Form.length_zero_array)


def length_one_if_typetracer(array: T) -> T:
    return _length_0_1_if_typetracer(array, Form.length_one_array)
