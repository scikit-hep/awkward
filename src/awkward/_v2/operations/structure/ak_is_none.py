# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def is_none(array, axis=0, highlevel=True, behavior=None):
    """
    Args:
        array: Data to check for missing values (None).
        axis (int): The dimension at which this operation is applied. The
            outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Returns an array whose value is True where an element of `array` is None;
    False otherwise (at a given `axis` depth).
    """

    # Determine the (potentially nested) bytemask
    def getfunction(layout, depth, depth_context, **kwargs):
        depth_context["posaxis"] = layout.axis_wrap_if_negative(
            depth_context["posaxis"]
        )
        if depth_context["posaxis"] != depth - 1:
            return

        nplike = ak.nplike.of(layout)

        if layout.is_OptionType:
            return ak._v2.contents.NumpyArray(
                nplike.asarray(layout.bytemask()).view(np.bool_)
            )

        elif (
            layout.is_UnknownType
            or layout.is_ListType
            or layout.is_RecordType
            or layout.is_NumpyType
        ):
            return ak._v2.contents.NumpyArray(nplike.zeros(len(layout), dtype=np.bool_))

    layout = ak._v2.operations.convert.to_layout(array)
    behavior = ak._v2._util.behavior_of(array, behavior=behavior)
    out = layout.recursively_apply(getfunction, depth_context={"posaxis": axis})
    return ak._v2._util.wrap(out, behavior, highlevel)
