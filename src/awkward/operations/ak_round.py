# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._connect.numpy import UNSUPPORTED
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext
from awkward._nplikes.numpy_like import NumpyMetadata

__all__ = ("round",)

np = NumpyMetadata.instance()


@ak._connect.numpy.implements("around")
@ak._connect.numpy.implements("round")
@high_level_function()
def round(
    array,
    decimals: int = 0,
    out=UNSUPPORTED,
    highlevel=True,
    behavior=None,
    attrs=None,
):
    """
    Args:
        array : array_like
            Input array.
        decimals : int, optional
            Number of decimal places to round to (default: 0).  If
            decimals is negative, it specifies the number of positions to
            the left of the decimal point.
        out : unsupported optional argument
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
    yield (array,)

    # Implementation
    return _impl(array, decimals, highlevel, behavior, attrs)


def _impl(array, decimals, highlevel, behavior, attrs):
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False, primitive_policy="error")

    # A closure over deg:
    def action(layout, backend, **kwargs):
        if isinstance(layout, ak.contents.NumpyArray):
            return ak.contents.NumpyArray(backend.nplike.round(layout.data, decimals))
        else:
            return None

    out = ak._do.recursively_apply(layout, action)
    return ctx.wrap(out, highlevel=highlevel)
