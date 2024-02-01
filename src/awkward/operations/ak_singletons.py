# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext, maybe_posaxis
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._regularize import is_integer, regularize_axis
from awkward.errors import AxisError

__all__ = ("singletons",)

np = NumpyMetadata.instance()


@high_level_function()
def singletons(array, axis=0, *, highlevel=True, behavior=None, attrs=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        axis (int): The dimension at which this operation is applied. The
            outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Returns a singleton list (length 1) wrapping each non-missing value and
    an empty list (length 0) in place of each missing value.

    For example,

        >>> array = ak.Array([1.1, 2.2, None, 3.3, None, None, 4.4, 5.5])
        >>> ak.singletons(array).show()
        [[1.1],
         [2.2],
         [],
         [3.3],
         [],
         [],
         [4.4],
         [5.5]]

    See #ak.firsts to invert this function.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, axis, highlevel, behavior, attrs)


def _impl(array, axis, highlevel, behavior, attrs):
    axis = regularize_axis(axis)
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False, primitive_policy="error")

    if not is_integer(axis):
        raise TypeError(f"'axis' must be an integer, not {axis!r}")

    def action(layout, depth, backend, **kwargs):
        posaxis = maybe_posaxis(layout, axis, depth)

        if posaxis is not None and posaxis + 1 == depth:
            if layout.is_union or layout.is_record:
                return None

            elif layout.is_option:
                offsets = backend.index_nplike.empty(layout.length + 1, dtype=np.int64)
                offsets[0] = 0

                backend.index_nplike.cumsum(
                    layout.mask_as_bool(valid_when=True), maybe_out=offsets[1:]
                )

                return ak.contents.ListOffsetArray(
                    ak.index.Index64(offsets), layout.project()
                )

            else:
                return ak.contents.RegularArray(layout, 1).to_ListOffsetArray64(True)

        elif layout.is_leaf:
            raise AxisError(f"axis={axis} exceeds the depth of this array ({depth})")

    out = ak._do.recursively_apply(layout, action, numpy_to_regular=True)

    return ctx.wrap(out, highlevel=highlevel)
