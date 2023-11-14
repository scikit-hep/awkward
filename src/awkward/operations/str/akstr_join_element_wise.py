# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._backends.typetracer import TypeTracerBackend
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext, ensure_same_backend

__all__ = ("join_element_wise",)

typetracer = TypeTracerBackend.instance()


@high_level_function(module="ak.str")
def join_element_wise(*arrays, highlevel=True, behavior=None, attrs=None):
    """
    Args:
        arrays: Array-like data (anything #ak.to_layout recognizes).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Broadcasts and concatenates all but the last array of strings in `arrays`;
    the last is used as a separator.

    Note: this function does not raise an error if the `array` does not
    contain any string or bytestring data.

    Requires the pyarrow library and calls
    [pyarrow.compute.binary_join_element_wise](https://arrow.apache.org/docs/python/generated/pyarrow.compute.binary_join_element_wise.html).

    Unlike Arrow's `binary_join_element_wise`, this function has no `null_handling`
    and `null_replacement` arguments. This function's behavior is like
    `null_handling="emit_null"` (Arrow's default). The other cases can be implemented
    with Awkward slices, #ak.drop_none, and #ak.fill_none.

    See also: #ak.str.join.
    """
    # Dispatch
    yield arrays

    # Implementation
    return _impl(arrays, highlevel, behavior, attrs)


def _impl(arrays, highlevel, behavior, attrs):
    from awkward._connect.pyarrow import import_pyarrow_compute
    from awkward.operations.str import _apply_through_arrow

    pc = import_pyarrow_compute("ak.str.join_element_wise")

    if len(arrays) < 1:
        raise TypeError("at least one array is required")

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layouts = ensure_same_backend(*(ctx.unwrap(x) for x in arrays))

    def action(layouts, **kwargs):
        if all(
            x.is_list and x.parameter("__array__") in ("string", "bytestring")
            for x in layouts
        ):
            return (_apply_through_arrow(pc.binary_join_element_wise, *layouts),)

    (out,) = ak._broadcasting.broadcast_and_apply(layouts, action)

    return ctx.wrap(out, highlevel=highlevel)
