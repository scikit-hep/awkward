# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext, maybe_posaxis
from awkward._namedaxis import (
    _get_named_axis,
    _keep_named_axis,
    _named_axis_to_positional_axis,
    _remove_named_axis,
)
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._regularize import regularize_axis

__all__ = ("flatten",)

np = NumpyMetadata.instance()


@high_level_function()
def flatten(array, axis=1, *, highlevel=True, behavior=None, attrs=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        axis (None or int): If None, the operation flattens all levels of
            nesting, returning a 1-dimensional array. Otherwise, it flattens
            at a specified depth. The outermost dimension is `0`, followed
            by `1`, etc., and negative values count backward from the
            innermost: `-1` is the innermost dimension, `-2` is the next
            level up, etc.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Returns an array with one level of nesting removed by erasing the
    boundaries between consecutive lists. Since this operates on a level of
    nesting, `axis=0` is a special case that only removes values at the
    top level that are equal to None.

    Consider the following.

        >>> array = ak.Array([[[1.1, 2.2, 3.3],
        ...                    [],
        ...                    [4.4, 5.5],
        ...                    [6.6]],
        ...                   [],
        ...                   [[7.7],
        ...                    [8.8, 9.9]
        ...                   ]])

    At `axis=1`, the outer lists (length 4, length 0, length 2) become a single
    list (of length 6).

        >>> ak.flatten(array, axis=1).show()
        [[1.1, 2.2, 3.3],
         [],
         [4.4, 5.5],
         [6.6],
         [7.7],
         [8.8, 9.9]]

    At `axis=2`, the inner lists (lengths 3, 0, 2, 1, 1, and 2) become three
    lists (of lengths 6, 0, and 3).

        >>> ak.flatten(array, axis=2).show()
        [[1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
         [],
         [7.7, 8.8, 9.9]]

    There's also an option to completely flatten the array with `axis=None`.
    This is useful for passing the data to a function that doesn't care about
    nested structure, such as a plotting routine.

        >>> ak.flatten(array, axis=None).show()
        [1.1,
         2.2,
         3.3,
         4.4,
         5.5,
         6.6,
         7.7,
         8.8,
         9.9]

    Missing values are eliminated by flattening: there is no distinction
    between an empty list and a value of None at the level of flattening.

        >>> array = ak.Array([[1.1, 2.2, 3.3], None, [4.4], [], [5.5]])
        >>> ak.flatten(array, axis=1)
        <Array [1.1, 2.2, 3.3, 4.4, 5.5] type='5 * float64'>

    As a consequence, flattening at `axis=0` does only one thing: it removes
    None values from the top level.

        >>> ak.flatten(array, axis=0)
        <Array [[1.1, 2.2, 3.3], [4.4], [], [5.5]] type='4 * var * float64'>

    As a technical detail, the flattening operation can be trivial in a common
    case, #ak.contents.ListOffsetArray in which the first `offset` is `0`.
    In that case, the flattened data is simply the array node's `content`.

        >>> array = ak.Array([[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]])
        >>> array.layout
        <ListOffsetArray len='5'>
            <offsets><Index dtype='int64' len='6'>
                [ 0  3  3  5  6 10]
            </Index></offsets>
            <content><NumpyArray dtype='float64' len='10'>
                [0.  1.1 2.2 3.3 4.4 5.5 6.6 7.7 8.8 9.9]
            </NumpyArray></content>
        </ListOffsetArray>

        >>> ak.flatten(array).layout
        <NumpyArray dtype='float64' len='10'>
            [0.  1.1 2.2 3.3 4.4 5.5 6.6 7.7 8.8 9.9]
        </NumpyArray>

        >>> array.layout.content
        <NumpyArray dtype='float64' len='10'>
            [0.  1.1 2.2 3.3 4.4 5.5 6.6 7.7 8.8 9.9]
        </NumpyArray>

    However, it is important to keep in mind that this is a special case:
    #ak.flatten and `content` are not interchangeable!

        >>> array = ak.Array(
        ...     ak.contents.ListArray(
        ...         ak.index.Index64(np.array([ 9, 100, 5, 8, 1])),
        ...         ak.index.Index64(np.array([12, 100, 7, 9, 5])),
        ...         ak.contents.NumpyArray(
        ...             np.array([999, 6.6, 7.7, 8.8, 9.9, 3.3, 4.4, 999, 5.5, 0., 1.1, 2.2, 999])
        ...         ),
        ...     )
        ... )
        >>> array.show()
        [[0, 1.1, 2.2],
         [],
         [3.3, 4.4],
         [5.5],
         [6.6, 7.7, 8.8, 9.9]]

        >>> ak.flatten(array).show()
        [0,
         1.1,
         2.2,
         3.3,
         4.4,
         5.5,
         6.6,
         7.7,
         8.8,
         9.9]

        >>> ak.Array(array.layout.content).show()
        [999,
         6.6,
         7.7,
         8.8,
         9.9,
         3.3,
         4.4,
         999,
         5.5,
         0,
         1.1,
         2.2,
         999]
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, axis, highlevel, behavior, attrs)


def _impl(array, axis, highlevel, behavior, attrs):
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False, primitive_policy="error")

    # Handle named axis
    named_axis = _get_named_axis(ctx)
    # Step 1: Normalize named axis to positional axis
    axis = _named_axis_to_positional_axis(named_axis, axis)
    axis = regularize_axis(axis, none_allowed=True)
    # Step 2: propagate named axis from input to output,
    #   if axis == None: use strategy "remove all" (see: awkward._namedaxis)
    #   if axis == 0: use strategy "keep all" (see: awkward._namedaxis)
    #   if axis != 0: use strategy "remove one" (see: awkward._namedaxis)
    if axis is None:
        pass
    elif axis == 0 or maybe_posaxis(layout, axis, 1) == 0:
        out_named_axis = _keep_named_axis(named_axis, None)
    else:
        out_named_axis = _remove_named_axis(named_axis, axis, layout.minmax_depth[1])

    if axis is None:
        out = ak._do.remove_structure(layout, function_name="ak.flatten")
        assert isinstance(out, tuple) and all(
            isinstance(x, ak.contents.Content) for x in out
        )

        out = ak._do.mergemany(out)

    elif axis == 0 or maybe_posaxis(layout, axis, 1) == 0:

        def apply(layout):
            if layout.is_unknown:
                return apply(ak.contents.NumpyArray(layout.backend.nplike.asarray([])))

            elif layout.is_indexed:
                return apply(layout.project())

            elif layout.is_union:
                if not any(
                    x.is_option and not isinstance(x, ak.contents.UnmaskedArray)
                    for x in layout.contents
                ):
                    return layout

                tags = layout.tags.data
                index = layout.backend.nplike.asarray(layout.index.data, copy=True)
                big_mask = layout.backend.nplike.empty(
                    layout.index.length, dtype=np.bool_
                )
                for tag, content in enumerate(layout.contents):
                    if content.is_option and not isinstance(
                        content, ak.contents.UnmaskedArray
                    ):
                        big_mask[:] = False
                        big_mask[tags == tag] = content.mask_as_bool(
                            valid_when=False
                        ).view(np.bool_)
                        index[big_mask] = -1

                good = index >= 0
                return ak.contents.UnionArray(
                    ak.index.Index8(tags[good]),
                    ak.index.Index64(index[good]),
                    layout.contents,
                )

            elif layout.is_option:
                return layout.project()

            else:
                return layout

        out = apply(layout)
    else:
        out = ak._do.flatten(layout, axis)

    wrapped_out = ctx.wrap(
        out,
        highlevel=highlevel,
    )

    # propagate named axis to output
    #   if axis == None: use strategy "remove all" (see: awkward._namedaxis)
    if axis is None:
        return ak.operations.ak_without_named_axis._impl(
            wrapped_out,
            highlevel=highlevel,
            behavior=ctx.behavior,
            attrs=ctx.attrs,
        )
    #   if axis == 0: use strategy "keep all" (see: awkward._namedaxis)
    #   if axis != 0: use strategy "remove one" (see: awkward._namedaxis)
    return ak.operations.ak_with_named_axis._impl(
        wrapped_out,
        named_axis=out_named_axis,
        highlevel=highlevel,
        behavior=ctx.behavior,
        attrs=ctx.attrs,
    )
