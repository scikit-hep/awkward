# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._experimental import experimental
from awkward._layout import HighLevelContext, maybe_posaxis
from awkward._namedaxis import (
    NAMED_AXIS_KEY,
    _add_named_axis,
    _get_named_axis,
    _named_axis_to_positional_axis,
)
from awkward._regularize import regularize_axis
from awkward._stack import field_lists, parameters, walk
from awkward.errors import AxisError

__all__ = ("stack",)


@high_level_function()
@experimental
def stack(array, axis=-1, *, dropna=True, highlevel=True, behavior=None, attrs=None):
    """Converts tuple records to lists, preserving named records.

    This is a tuple/list structure conversion, not NumPy-style stacking of a
    sequence of arrays. See #ak.unstack for the reverse operation.

    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        axis (int or str): The array axis containing the tuples. Flat arrays of
            tuples use axis=0; tuples inside one level of lists use axis=1.
            Negative axes count from the innermost array dimension. The new
            list dimension is inserted immediately after this axis.
        dropna (bool): If True, omit missing tuple fields. If False, retain
            missing fields and pad shorter tuple types with None to the largest
            number of fields at this axis. Padding is independent for each
            named-record field. Missing tuples themselves remain None.
        highlevel (bool): If True, return an #ak.Array; otherwise return a
            low-level layout.
        behavior (None or dict): Custom behavior for the output.
        attrs (None or dict): Custom attributes for the output.

    Returns:
        An array with the selected tuples replaced by lists. Named records and
        other values are retained. Numeric fields may be promoted to a common
        dtype, and incompatible field types are represented by unions.

    Examples:
        >>> a = ak.Array([[(1, 2), (1, 3)], [], [(4, 5), (6, 7, 8)]])
        >>> ak.stack(a).to_list()
        [[[1, 2], [1, 3]], [], [[4, 5], [6, 7, 8]]]
        >>> ak.stack(a, dropna=False).to_list()
        [[[1, 2, None], [1, 3, None]], [], [[4, 5, None], [6, 7, 8]]]
        >>> ak.stack([{"x": (2, 3), "y": 1}]).to_list()
        [{'x': [2, 3], 'y': 1}]

    This operation preserves the order of tuple fields and does not convert
    the array to Python objects. To stack tuples whose fields themselves
    contain lists, select their position with an explicit non-negative axis.
    """
    yield (array,)
    return _impl(array, axis, dropna, highlevel, behavior, attrs)


def _impl(array, axis, dropna, highlevel, behavior, attrs):
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False, primitive_policy="error")
    named_axis = _get_named_axis(ctx)
    axis = regularize_axis(
        _named_axis_to_positional_axis(named_axis, axis), none_allowed=False
    )
    if axis >= layout.minmax_depth[1] or axis < -layout.minmax_depth[1]:
        raise AxisError(f"axis={axis} exceeds the depth of this array")
    widths = {}

    def selected(node, depth):
        return (
            node.is_record
            and node.is_tuple
            and maybe_posaxis(node, axis, depth) == depth - 1
        )

    def collect(node, depth, path):
        if selected(node, depth):
            widths[path] = max(widths.get(path, 0), len(node.fields))
            return node

    walk(layout, collect, rebuild=False)

    def action(node, depth, path):
        if not selected(node, depth):
            return None
        width = len(node.fields) if dropna else widths[path]
        contents = [node.content(i) for i in range(len(node.fields))]
        for _ in range(len(contents), width):
            contents.append(
                ak.contents.IndexedOptionArray(
                    ak.index.Index64(
                        node.backend.nplike.full(node.length, -1, dtype="int64")
                    ),
                    ak.contents.EmptyArray(backend=node.backend),
                )
            )
        if contents:
            out = ak.operations.ak_concatenate._impl(
                [
                    field_lists(content)
                    if dropna
                    else ak.contents.RegularArray(content, 1)
                    for content in contents
                ],
                1,
                False,
                highlevel=False,
                behavior=None,
                attrs=None,
            )
        else:
            out = ak.contents.RegularArray(
                ak.contents.EmptyArray(backend=node.backend),
                0,
                node.length,
            )
        return out.copy(parameters=parameters(node))

    out = walk(layout, action)
    if widths:
        named_axis = _add_named_axis(
            named_axis, axis + 1 if axis >= 0 else axis, layout.minmax_depth[1]
        )
    return ctx.with_attr(NAMED_AXIS_KEY, named_axis).wrap(out, highlevel=highlevel)
