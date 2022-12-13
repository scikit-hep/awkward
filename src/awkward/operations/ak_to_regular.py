# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()


def to_regular(array, axis=1, *, highlevel=True, behavior=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        axis (int or None): The dimension at which this operation is applied.
            The outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc. If None, convert all
            variable dimensions into regular ones or raise a ValueError if that
            is not possible.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Converts a variable-length axis into a regular one, if possible.

        >>> irregular = ak.from_iter(np.arange(2*3*5).reshape(2, 3, 5))
        >>> irregular.type.show()
        2 * var * var * int64
        >>> ak.to_regular(irregular).type.show()
        2 * 3 * var * int64
        >>> ak.to_regular(irregular, axis=2).type.show()
        2 * var * 5 * int64
        >>> ak.to_regular(irregular, axis=-1).type.show()
        2 * var * 5 * int64

    But truly irregular data cannot be converted.

        >>> ak.to_regular(ak.Array([[1, 2, 3], [], [4, 5]]))
        ValueError: while calling
            ak.to_regular(
                array = <Array [[1, 2, 3], [], [4, 5]] type='3 * var * int64'>
                axis = 1
                highlevel = True
                behavior = None
            )
        Error details: cannot convert to RegularArray because subarray lengths are not regular

    See also #ak.from_regular.
    """
    with ak._errors.OperationErrorContext(
        "ak.to_regular",
        dict(array=array, axis=axis, highlevel=highlevel, behavior=behavior),
    ):
        return _impl(array, axis, highlevel, behavior)


def _impl(array, axis, highlevel, behavior):
    layout = ak.operations.to_layout(array)
    behavior = ak._util.behavior_of(array, behavior=behavior)

    if axis is None:

        def action(layout, continuation, **kwargs):
            if layout.is_list:
                return continuation().to_RegularArray()

        out = ak._do.recursively_apply(layout, action, behavior)

    elif ak._util.maybe_posaxis(layout, axis, 1) == 0:
        out = layout  # the top-level can only be regular (ArrayType)

    else:

        def action(layout, depth, **kwargs):
            posaxis = ak._util.maybe_posaxis(layout, axis, depth)
            if posaxis == depth and layout.is_list:
                return layout.to_RegularArray()

            elif layout.is_leaf:
                raise ak._errors.wrap_error(
                    np.AxisError(
                        f"axis={axis} exceeds the depth of this array ({depth})"
                    )
                )

        out = ak._do.recursively_apply(layout, action, behavior)

    return ak._util.wrap(out, behavior, highlevel)
