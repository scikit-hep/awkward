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

__all__ = ("tuples_to_lists",)


@high_level_function()
@experimental
def tuples_to_lists(
    array,
    axis=-1,
    *,
    drop_none=False,
    pad=False,
    highlevel=True,
    behavior=None,
    attrs=None,
):
    """Converts tuple records to lists, preserving named records.

    This is a tuple/list structure conversion, not NumPy-style tuples_to_listsing of a
    sequence of arrays. See #ak.lists_to_tuples for the reverse operation.

    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        axis (int or str): The list dimension inserted in the result.
            Flat tuple arrays use axis=1; tuples inside one level of lists use
            axis=2. Negative axes count from the innermost input dimension.
            Raises AxisError if no tuple records are selected.
        drop_none (bool): If True, omit missing tuple fields. The default
            preserves field positions, including None values.
        pad (bool): If True, pad shorter tuple variants with None to the largest
            field count at this axis, independently within each named field.
            The default preserves each tuple's width. Padding is applied before
            dropping None fields. Missing tuples remain None.
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
        >>> ak.tuples_to_lists(a).to_list()
        [[[1, 2], [1, 3]], [], [[4, 5], [6, 7, 8]]]
        >>> ak.tuples_to_lists(a, pad=True).to_list()
        [[[1, 2, None], [1, 3, None]], [], [[4, 5, None], [6, 7, 8]]]
        >>> ak.tuples_to_lists([{"x": (2, 3), "y": 1}]).to_list()
        [{'x': [2, 3], 'y': 1}]

    This operation preserves the order of tuple fields and does not convert
    the array to Python objects. To tuples_to_lists tuples whose fields themselves
    contain lists, select their position with an explicit non-negative axis.
    Tuple fields share the enclosing tuple's array depth. When tuples are
    nested directly, the axis selects the outer tuple and leaves inner tuples
    as elements of the resulting lists.
    """
    yield (array,)
    return _impl(array, axis, drop_none, pad, highlevel, behavior, attrs)


def _impl(array, axis, drop_none, pad, highlevel, behavior, attrs):
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False, primitive_policy="error")
    named_axis = _get_named_axis(ctx)
    axis = regularize_axis(
        _named_axis_to_positional_axis(named_axis, axis), none_allowed=False
    )
    insertion_axis = axis
    if axis == 0:
        raise AxisError("axis=0 cannot insert a list inside an array element")
    if axis > 0:
        axis -= 1
    if axis >= layout.minmax_depth[1] or axis < -layout.minmax_depth[1]:
        raise AxisError(f"axis={insertion_axis} exceeds the depth of this array")
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
    if not widths:
        raise AxisError(f"no tuple records at axis={axis} to tuples_to_lists")

    def action(node, depth, path):
        if not selected(node, depth):
            return None
        width = len(node.fields) if not pad else widths[path]
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
                    if drop_none
                    else ak.contents.RegularArray(content, 1)
                    for content in contents
                ],
                1,
                True,
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
    named_axis = _add_named_axis(
        named_axis, axis + 1 if axis >= 0 else axis, layout.minmax_depth[1]
    )
    return ctx.with_attr(NAMED_AXIS_KEY, named_axis).wrap(out, highlevel=highlevel)
