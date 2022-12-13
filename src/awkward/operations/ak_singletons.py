# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()


def singletons(array, axis=0, *, highlevel=True, behavior=None):
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
    with ak._errors.OperationErrorContext(
        "ak.singletons",
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

    def action(layout, depth, **kwargs):
        posaxis = ak._util.maybe_posaxis(layout, axis, depth)

        if posaxis is not None and posaxis + 1 == depth:
            if layout.is_union or layout.is_record:
                return None

            elif layout.is_option:
                nplike = layout._backend.index_nplike

                offsets = nplike.empty(layout.length + 1, dtype=np.int64)
                offsets[0] = 0

                nplike.cumsum(layout.mask_as_bool(valid_when=True), out=offsets[1:])

                return ak.contents.ListOffsetArray(
                    ak.index.Index64(offsets), layout.project()
                )

            else:
                return ak.contents.RegularArray(layout, 1).to_ListOffsetArray64(True)

        elif layout.is_leaf:
            raise ak._errors.wrap_error(
                np.AxisError(f"axis={axis} exceeds the depth of this array ({depth})")
            )

    out = ak._do.recursively_apply(layout, action, behavior, numpy_to_regular=True)

    return ak._util.wrap(out, behavior, highlevel)
