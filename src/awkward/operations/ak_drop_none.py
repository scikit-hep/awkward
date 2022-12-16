# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()


def drop_none(array, axis=None, highlevel=True, behavior=None):
    """
    Args:
        array: Data in which to remove Nones.
        axis (None or int): If None, the operation drops Nones at all levels of
            nesting, returning an array of the same dimension, but without Nones.
            Otherwise, it drops Nones at a specified depth.
            The outermost dimension is `0`, followed by `1`, etc.,
            and negative values count backward from the innermost: `-1` is the
            innermost dimension, `-2` is the next level up, etc.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Removes missing values (None) from a given array.

    For example, in the following `array`,

        >>> array = ak.Array([[[0]], [[None]], [[1], None], [[2, None]]])

    The None value will be removed, resulting in

        >>> ak.drop_none(array)
        <Array [[[0]], [[]], [[1]], [[2]]] type='4 * var * var * int64'>

    The default axis is None, however an axis can be specified:

        >>> ak.drop_none(array, axis=1)
        <Array [[[0]], [[None]], [[1]], [[2, None]]] type='4 * var * var * ?int64'>

    """
    with ak._errors.OperationErrorContext(
        "ak.drop_none",
        dict(array=array, axis=axis, highlevel=highlevel, behavior=behavior),
    ):
        return _impl(array, axis, highlevel, behavior)


def _impl(array, axis, highlevel, behavior):
    layout = ak.operations.to_layout(array, allow_record=False, allow_other=False)

    def drop_nones(layout, **kwargs):
        if layout.is_list:
            # only drop nones at list level in the recursion; this way ListArray -> ListOffsetArray with unprojected optiontype -> avoid offset mismatch
            return layout.drop_none()
        else:
            return layout

    if axis is None:
        # if the outer layout is_option, drop_nones without affecting offsets
        if layout.is_option:
            layout = layout.drop_none()

        def action(layout, continuation, **kwargs):
            return drop_nones(continuation())

    else:
        max_axis = layout.branch_depth[1] - 1
        if axis > max_axis:
            raise ak._errors.wrap_error(
                np.AxisError(
                    f"axis={axis} exceeds the depth ({max_axis}) of this array"
                )
            )

        def recompute_offsets(layout, depth, **kwargs):
            posaxis = ak._util.maybe_posaxis(layout, axis, depth)
            if (
                posaxis == 0
                and posaxis == depth - 1
                or posaxis == depth
                and layout.is_list
            ):
                none_indexes = options["none_indexes"].pop(0)
                out = layout._rebuild_without_nones(none_indexes, layout.content)
                return out

        def action(layout, depth, **kwargs):
            if layout.is_record:
                posaxises = {
                    ak._util.maybe_posaxis(x, axis, depth) for x in layout.contents
                }
                if len(posaxises) > 1 and any(x < depth for x in posaxises):
                    raise ak._errors.wrap_error(
                        np.AxisError(
                            f"axis={axis} implies different levels in records that might require part of a record to be dropped, which is impossible"
                        )
                    )
            posaxis = ak._util.maybe_posaxis(layout, axis, depth)
            if posaxis == 0:
                if not layout.is_option:
                    return layout
                else:
                    return layout.drop_none()
            if posaxis == depth - 1 and layout.is_option:
                _, _, none_indexes = layout._nextcarry_outindex(layout.backend)
                options["none_indexes"].append(none_indexes)
                return layout.drop_none()
            if posaxis == depth - 1 and layout.is_list and layout.content.is_option:
                return layout.drop_none()

    options = {"none_indexes": []}
    out = ak._do.recursively_apply(layout, action, behavior, options)

    if len(options["none_indexes"]) > 0:
        out = ak._do.recursively_apply(out, recompute_offsets, behavior, options)

    return ak._util.wrap(out, behavior, highlevel, like=behavior)
