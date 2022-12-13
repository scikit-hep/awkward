# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()


def firsts(array, axis=1, *, highlevel=True, behavior=None):
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
    with ak._errors.OperationErrorContext(
        "ak.firsts",
        dict(array=array, axis=axis, highlevel=highlevel, behavior=behavior),
    ):
        return _impl(array, axis, highlevel, behavior)


def _impl(array, axis, highlevel, behavior):
    layout = ak.operations.to_layout(array)
    behavior = ak._util.behavior_of(array, behavior=behavior)

    if not ak._util.is_integer(axis):
        raise ak._errors.wrap_error(
            TypeError(f"'axis' must be an integer, not {axis!r}")
        )

    if ak._util.maybe_posaxis(layout, axis, 1) == 0:
        # specialized logic; it's tested in test_0582-propagate-context-in-broadcast_and_apply.py
        # Build an integer-typed slice array, so that we can
        # ensure we have advanced indexing for both length==0
        # and length > 0 cases.
        slicer = ak.from_iter([None, 0])
        if layout.length == 0:
            out = layout[slicer[[0]]][0]
        else:
            out = layout[slicer[[1]]][0]

    else:

        def action(layout, depth, depth_context, **kwargs):
            posaxis = ak._util.maybe_posaxis(layout, axis, depth)

            if posaxis == depth and layout.is_list:
                nplike = layout._backend.index_nplike

                # this is a copy of the raw array
                index = starts = nplike.array(layout.starts.raw(nplike), dtype=np.int64)

                # this might be a view
                stops = layout.stops.raw(nplike)

                empties = starts == stops
                index[empties] = -1

                return ak.contents.IndexedOptionArray.simplified(
                    ak.index.Index64(index), layout._content
                )

            elif layout.is_leaf:
                raise ak._errors.wrap_error(
                    np.AxisError(
                        f"axis={axis} exceeds the depth of this array ({depth})"
                    )
                )

        out = ak._do.recursively_apply(layout, action, behavior, numpy_to_regular=True)

    return ak._util.wrap(out, behavior, highlevel)
