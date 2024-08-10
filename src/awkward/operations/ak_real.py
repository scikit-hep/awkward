# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext
from awkward._nplikes.numpy_like import NumpyMetadata

__all__ = ("real",)

np = NumpyMetadata.instance()


@ak._connect.numpy.implements("real")
@high_level_function()
def real(val, highlevel=True, behavior=None, attrs=None):
    """
    Args:
        val : array_like
            Input array.
        highlevel (bool, default is True): If True, return an #ak.Array;
            otherwise, return a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Returns the real components of the given array elements.
    If the arrays have complex elements, the returned arrays are floats.
    """
    # Dispatch
    yield (val,)

    # Implementation
    return _impl_real(val, highlevel, behavior, attrs)


def _impl_real(val, highlevel, behavior, attrs):
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(val, allow_record=False, primitive_policy="error")

    out = ak._do.recursively_apply(layout, _action_real)
    return ctx.wrap(out, highlevel=highlevel)


def _action_real(layout, backend, **kwargs):
    if isinstance(layout, ak.contents.NumpyArray):
        return ak.contents.NumpyArray(backend.nplike.real(layout.data))
    else:
        return None
