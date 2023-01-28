from __future__ import annotations

__all__ = ["is_unknown_array", "is_unknown_scalar"]

from awkward._nplikes.typetracer import is_unknown_array, is_unknown_scalar
from awkward.highlevel import Array, Record
from awkward.typing import TypeVar

T = TypeVar("T", Array, Record)


def empty_if_typetracer(array: T) -> T:
    from awkward._backends import TypeTracerBackend
    from awkward._util import behavior_of, wrap
    from awkward.operations.ak_to_layout import to_layout

    typetracer_backend = TypeTracerBackend.instance()

    layout = to_layout(array, allow_other=False)
    behavior = behavior_of(array)

    if layout.backend is typetracer_backend:
        layout._touch_data(True)
        return layout.form.length_zero_array(behavior=behavior)
    else:
        return wrap(layout, behavior=behavior)
