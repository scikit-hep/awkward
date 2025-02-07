# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._nplikes.virtual import VirtualArray

__all__ = ("materialize",)

np = NumpyMetadata.instance()


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
    return _impl1(array, highlevel, behavior, attrs)


def _impl1(array, highlevel, behavior, attrs):
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=True, primitive_policy="error")

    def action(layout, backend, **kwargs):
        if isinstance(layout, ak.contents.NumpyArray):
            buffer = layout.data
            if isinstance(buffer, VirtualArray):
                out = buffer.materialize()
            else:
                out = buffer
            return ak.contents.NumpyArray(out)
        else:
            return None

    out = ak._do.recursively_apply(layout, action)
    return ctx.wrap(out, highlevel=highlevel)
