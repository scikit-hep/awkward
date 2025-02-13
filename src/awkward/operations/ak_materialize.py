# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext
from awkward._nplikes.virtual import VirtualArray

__all__ = ("materialize",)


@high_level_function()
def materialize(
    array,
    highlevel=True,
    behavior=None,
    attrs=None,
):
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, highlevel, behavior, attrs)


def _impl(array, highlevel, behavior, attrs):
    if not isinstance(
        array,
        (
            ak.highlevel.Array,
            ak.highlevel.Record,
            ak.contents.Content,
            ak.record.Record,
        ),
    ):
        return array

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(
            array, allow_record=True, allow_unknown=True, primitive_policy="error"
        )

    def action(layout, backend, **kwargs):
        if isinstance(layout, ak.contents.NumpyArray):
            buffer = layout.data
            if isinstance(buffer, VirtualArray):
                out = buffer.materialize()
            else:
                out = buffer
            return ak.contents.NumpyArray(out, parameters=layout.parameters)
        else:
            return None

    out = ak._do.recursively_apply(
        layout,
        action,
        numpy_to_regular=False,
        return_simplified=False,
    )
    return ctx.wrap(out, highlevel=highlevel)
