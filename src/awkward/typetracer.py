from __future__ import annotations

__all__ = [
    "is_unknown_array",
    "is_unknown_scalar",
    "empty_if_typetracer",
    "TypeTracerReport",
    "typetracer_with_report",
]

from awkward._backends.typetracer import TypeTracerBackend
from awkward._behavior import behavior_of
from awkward._layout import wrap_layout
from awkward._nplikes.typetracer import (
    TypeTracerReport,
    is_unknown_array,
    is_unknown_scalar,
    typetracer_with_report,
)
from awkward._typing import TypeVar
from awkward.highlevel import Array, Record
from awkward.operations.ak_to_layout import to_layout

T = TypeVar("T", Array, Record)


def empty_if_typetracer(array: T) -> T:
    typetracer_backend = TypeTracerBackend.instance()

    layout = to_layout(array, allow_other=False)
    behavior = behavior_of(array)

    if layout.backend is typetracer_backend:
        layout._touch_data(True)
        layout = layout.form.length_zero_array(highlevel=False)

    return wrap_layout(layout, behavior=behavior)
