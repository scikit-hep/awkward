# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def from_regular(array, axis=1, highlevel=True, behavior=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        axis (int or None): The dimension at which this operation is applied.
            The outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc. If None, convert all
            regular dimensions into variable ones.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Converts a regular axis into an irregular one.

        >>> regular = ak.Array(np.arange(2*3*5).reshape(2, 3, 5))
        >>> print(regular.type)
        2 * 3 * 5 * int64
        >>> print(ak.from_regular(regular).type)
        2 * var * 5 * int64
        >>> print(ak.from_regular(regular, axis=2).type)
        2 * 3 * var * int64
        >>> print(ak.from_regular(regular, axis=-1).type)
        2 * 3 * var * int64

    See also #ak.to_regular.
    """
    layout = ak._v2.operations.convert.to_layout(array)
    posaxis = layout.axis_wrap_if_negative(axis)

    if axis is None:

        def action(layout, continuation, **kwargs):
            if layout.is_RegularType:
                return continuation().toListOffsetArray64(False)

        out = layout.recursively_apply(action, numpy_to_regular=True)

    elif posaxis == 0:
        out = layout  # the top-level is already regular (ArrayType)

    else:

        def action(layout, depth, depth_context, **kwargs):
            posaxis = layout.axis_wrap_if_negative(depth_context["posaxis"])
            if posaxis == depth and layout.is_RegularType:
                return layout.toListOffsetArray64(False)
            elif posaxis == depth and layout.is_ListType:
                return layout
            elif posaxis == 0:
                raise np.AxisError(
                    "axis={0} exceeds the depth of this array ({1})".format(axis, depth)
                )

            depth_context["posaxis"] = posaxis

        depth_context = {"posaxis": posaxis}
        out = layout.recursively_apply(action, depth_context, numpy_to_regular=True)

    return ak._v2._util.wrap(out, behavior, highlevel)
