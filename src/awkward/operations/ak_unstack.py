# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._experimental import experimental
from awkward._layout import HighLevelContext, maybe_posaxis
from awkward._namedaxis import (
    NAMED_AXIS_KEY,
    _get_named_axis,
    _named_axis_to_positional_axis,
    _remove_named_axis,
)
from awkward._nplikes.shape import unknown_length
from awkward._regularize import is_integer, regularize_axis
from awkward._stack import is_string, parameters, walk
from awkward.errors import AxisError

__all__ = ("unstack",)


@high_level_function()
@experimental
def unstack(array, axis=-1, *, size=None, highlevel=True, behavior=None, attrs=None):
    """Converts lists to tuple records, padding shorter lists with None.

    See #ak.stack for the reverse tuple/list structure conversion. Unlike
    #ak.unzip, the result is one array, rather than a Python tuple of arrays.

    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        axis (int or str): The list dimension to convert. The outermost nested
            lists are axis=1; negative axes count from the innermost dimension.
            The top-level array (axis=0) cannot be converted to a tuple record.
        size (None or int): The number of fields in the output tuples. If None,
            use the maximum list length at this axis, independently for each
            named-record field. With an explicit size, shorter lists are padded
            and longer lists are truncated. Regular list sizes can be inferred
            from metadata; ragged sizes require inspecting data on CPU. Supply
            size for ragged typetracer arrays and non-CPU backends to avoid a
            device-to-host read for width inference.
        highlevel (bool): If True, return an #ak.Array; otherwise return a
            low-level layout.
        behavior (None or dict): Custom behavior for the output.
        attrs (None or dict): Custom attributes for the output.

    Returns:
        An array whose selected lists are tuple records. Missing lists remain
        None, and named records and strings remain intact. Empty groups become
        tuples of None, or empty tuples if the inferred/explicit size is zero.

    Examples:
        >>> a = ak.Array([[[1, 2], [1, 3], [2, 3]], [], [[4, 5], [6]]])
        >>> ak.unstack(a).to_list()
        [[(1, 2), (1, 3), (2, 3)], [], [(4, 5), (6, None)]]
        >>> ak.unstack([[1], [], [2, 3]], size=2).to_list()
        [(1, None), (None, None), (2, 3)]

    Tuple width is part of the output type. An explicit size is therefore
    needed whenever ragged lengths are unavailable, such as in typetracer.
    Structural operations may still synchronize on GPU; size avoids the
    additional synchronization that inferring the number of fields would need.
    """
    yield (array,)
    return _impl(array, axis, size, highlevel, behavior, attrs)


def _impl(array, axis, size, highlevel, behavior, attrs):
    if size is not None:
        if not is_integer(size):
            raise TypeError("size must be a non-negative integer or None")
        size = int(size)
        if size < 0:
            raise ValueError("size must be non-negative")
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False, primitive_policy="error")
    named_axis = _get_named_axis(ctx)
    axis = regularize_axis(
        _named_axis_to_positional_axis(named_axis, axis), none_allowed=False
    )
    if axis == 0:
        raise AxisError("axis=0 is the top-level array, not a nested list")
    if axis >= layout.minmax_depth[1] or axis < -layout.minmax_depth[1]:
        raise AxisError(f"axis={axis} exceeds the depth of this array")

    # Width inference must not include unreachable backing data in sliced or
    # indexed inputs. Explicit sizes and regular typetracer sizes need no scan.
    if size is None and layout.backend.name == "cpu":

        def normalize_options(node, depth, path):
            if node.is_option and not isinstance(node, ak.contents.IndexedOptionArray):
                option = node.to_IndexedOptionArray64()
                return option.copy(
                    content=walk(option.content, normalize_options, depth, path)
                )

        layout = walk(layout, normalize_options).to_packed()
    widths = {}

    def selected(node, depth):
        return (
            node.is_list
            and not is_string(node)
            and maybe_posaxis(node, axis, depth) == depth
        )

    def collect(node, depth, path):
        if selected(node, depth):
            if size is not None:
                width = size
            elif node.is_regular and node.size is not unknown_length:
                width = node.size
            elif node.backend.name != "cpu":
                raise ValueError(
                    "unstack requires an explicit size for ragged lists on "
                    "typetracer and non-CPU backends"
                )
            else:
                counts = node.stops.data - node.starts.data
                width = int(node.backend.nplike.max(counts)) if counts.size else 0
            widths[path] = max(widths.get(path, 0), width)
            return node

    walk(layout, collect, rebuild=False)
    if not widths:
        raise AxisError(f"no list dimension at axis={axis} to unstack")

    def action(node, depth, path):
        if not selected(node, depth):
            return None
        width = widths[path]
        padded = ak.operations.ak_pad_none._impl(
            node, width, 1, True, False, None, None
        )
        return ak.contents.RecordArray(
            [padded[:, i] for i in range(width)],
            None,
            node.length,
            parameters=parameters(node),
            backend=node.backend,
        )

    out = walk(layout, action)
    named_axis = _remove_named_axis(named_axis, axis, layout.minmax_depth[1])
    return ctx.with_attr(NAMED_AXIS_KEY, named_axis).wrap(out, highlevel=highlevel)
