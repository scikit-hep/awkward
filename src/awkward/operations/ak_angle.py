# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext
from awkward._nplikes.numpy_like import NumpyMetadata

__all__ = ("angle",)

np = NumpyMetadata.instance()


@ak._connect.numpy.implements("angle")
@high_level_function()
def angle(val, deg=False, highlevel=True, behavior=None, attrs=None):
    """
    Args:
        val : array_like
            Input array.
        deg (bool, default is False): If True, returns angles in degrees,
            otherwise in radians.
        highlevel (bool, default is True): If True, return an #ak.Array;
            otherwise, return a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Returns the counterclockwise angle from the positive real axis on the complex
    plane in the range ``(-pi, pi]``, with dtype as a float.
    """
    # Dispatch
    yield (val,)

    # Implementation
    return _impl_angle(val, deg, highlevel, behavior, attrs)


def _impl_angle(val, deg, highlevel, behavior, attrs):
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(val, allow_record=False, primitive_policy="error")

    # A closure over deg:
    def action_angle(layout, backend, **kwargs):
        if isinstance(layout, ak.contents.NumpyArray):
            return ak.contents.NumpyArray(backend.nplike.angle(layout.data, deg))
        else:
            return None

    out = ak._do.recursively_apply(layout, action_angle)
    return ctx.wrap(out, highlevel=highlevel)
