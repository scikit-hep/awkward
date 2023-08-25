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
from awkward._do import touch_data as _touch_data
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


def _length_0_1_if_typetracer(array, function, highlevel: bool, behavior) -> T:
    typetracer_backend = TypeTracerBackend.instance()

    layout = to_layout(array, allow_other=False)
    behavior = behavior_of(array, behavior=behavior)

    if layout.backend is typetracer_backend:
        _touch_data(layout)
        layout = function(layout.form, highlevel=False)

    return wrap_layout(layout, behavior=behavior, highlevel=highlevel)


def empty_if_typetracer(array) -> T:
    deprecate(
        "'empty_if_typetracer' is being replaced by 'length_zero_if_typetracer' (change name)",
        "2.4.0",
    )

    return length_zero_if_typetracer(array)


def length_zero_if_typetracer(array, *, highlevel: bool = True, behavior=None) -> T:
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Recursively touches the data of an array, before returning a length-zero
    NumPy-backed iff. the given array has a typetracer backend; otherwise, a
    shallow copy of the original array is returned.
    """
    return _length_0_1_if_typetracer(array, Form.length_zero_array, highlevel, behavior)


def length_one_if_typetracer(array, *, highlevel: bool = True, behavior=None) -> T:
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Recursively touches the data of an array, before returning a length-one
    NumPy-backed iff. the given array has a typetracer backend; otherwise, a
    shallow copy of the original array is returned.
    """
    return _length_0_1_if_typetracer(array, Form.length_one_array, highlevel, behavior)


def touch_data(array, *, highlevel: bool = True, behavior=None) -> T:
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Recursively touches the data and returns a shall copy of the given array.
    """
    behavior = behavior_of(array, behavior=behavior)
    layout = to_layout(array, allow_other=False)
    _touch_data(layout)
    return wrap_layout(layout, behavior=behavior, highlevel=highlevel)
