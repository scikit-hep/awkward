# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext, maybe_posaxis
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._regularize import is_integer, regularize_axis
from awkward.errors import AxisError

__all__ = ("firsts",)

np = NumpyMetadata.instance()


@high_level_function()
def firsts(array, axis=1, *, highlevel=True, behavior=None, attrs=None):
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

    Selects the first element of each non-empty list and inserts None for each
    empty list.

    For example,

        >>> array = ak.Array([[1.1], [2.2], [], [3.3], [], [], [4.4], [5.5]])
        >>> ak.firsts(array).show()
        [1.1,
         2.2,
         None,
         3.3,
         None,
         None,
         4.4,
         5.5]

    See #ak.singletons to invert this function.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, axis, highlevel, behavior, attrs)


def _impl(array, axis, highlevel, behavior, attrs):
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False)
    axis = regularize_axis(axis)

    if not is_integer(axis):
        raise TypeError(f"'axis' must be an integer, not {axis!r}")

    if maybe_posaxis(layout, axis, 1) == 0:
        # specialized logic; it's tested in test_0582-propagate-context-in-broadcast_and_apply.py
        # Build an integer-typed slice array, so that we can
        # ensure we have advanced indexing for both length==0
        # and length > 0 cases.
        backend = ak.backend(array)
        slicer = ak.to_backend(ak.from_iter([None, 0]), backend)
        if layout.length == 0:
            out = layout[slicer[[0]]][0]
        else:
            out = layout[slicer[[1]]][0]

    else:

        def action(layout, depth, backend, **kwargs):
            posaxis = maybe_posaxis(layout, axis, depth)

            if posaxis == depth and layout.is_list:
                # this is a copy of the raw array
                index = starts = backend.index_nplike.asarray(
                    layout.starts.data, dtype=np.int64, copy=True
                )

                # this might be a view
                stops = layout.stops.data

                empties = starts == stops
                index[empties] = -1

                return ak.contents.IndexedOptionArray.simplified(
                    ak.index.Index64(index), layout._content
                )

            elif layout.is_leaf:
                raise AxisError(
                    f"axis={axis} exceeds the depth of this array ({depth})"
                )

        out = ak._do.recursively_apply(layout, action, numpy_to_regular=True)

    return ctx.wrap(out, highlevel=highlevel, allow_other=True)
