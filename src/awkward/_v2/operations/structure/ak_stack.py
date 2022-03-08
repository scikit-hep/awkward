# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

from awkward._v2.operations.structure.ak_fill_none import fill_none

np = ak.nplike.NumpyMetadata.instance()


# @ak._v2._connect.numpy.implements("stack")
def stack(arrays, axis=0, merge=True, mergebool=True, highlevel=True, behavior=None):
    """
    Args:
        arrays: Arrays to concatenate along a new dimension.
        axis (int): The dimension in the result at which the arrays are stacked.
            The outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc.
        merge (bool): If True, combine data into the same buffers wherever
            possible, eliminating unnecessary #ak.layout.UnionArray8_64 types
            at the expense of materializing #ak.layout.VirtualArray nodes.
        mergebool (bool): If True, boolean and nummeric data can be combined
            into the same buffer, losing information about False vs `0` and
            True vs `1`; otherwise, they are kept in separate buffers with
            distinct types (using an #ak.layout.UnionArray8_64).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Returns an array with `arrays` stacked along a new axis. Above the axis `arrays`
    must be broadcastable.
    """
    with ak._v2._util.OperationErrorContext(
        "ak._v2.stack",
        dict(
            arrays=arrays,
            axis=axis,
            merge=merge,
            mergebool=mergebool,
            highlevel=highlevel,
            behavior=behavior,
        ),
    ):
        return _impl(arrays, axis, merge, mergebool, highlevel, behavior)


def _impl(arrays, axis, merge, mergebool, highlevel, behavior):
    layouts = [
        ak._v2.operations.convert.to_layout(
            x, allow_record=False if axis == 0 else True, allow_other=True
        )
        for x in arrays
    ]
    behavior = ak._v2._util.behavior_of(*arrays, behavior=behavior)
    nplike = ak.nplike.of(*layouts)

    contents = [x for x in layouts if isinstance(x, ak._v2.contents.Content)]
    if not contents:
        raise ak._v2._util.error(ValueError("need at least one array to concatenate"))

    posaxis = contents[0].axis_wrap_if_negative(axis)
    maxdepth = max(x.minmax_depth[1] for x in contents)
    if not 0 <= posaxis <= maxdepth:
        raise ak._v2._util.error(
            ValueError(
                "axis={} is beyond the depth of this array or the depth of this array "
                "is ambiguous".format(axis)
            )
        )
    if any(x.axis_wrap_if_negative(axis) != posaxis for x in contents):
        raise ak._v2._util.error(
            ValueError(
                "arrays to concatenate do not have the same depth for negative "
                "axis={}".format(axis)
            )
        )

    if posaxis == 0:
        length = max(len(x) for x in contents)

        nextinputs = []
        for x in layouts:
            if isinstance(x, ak._v2.contents.Content):
                nextinputs.append(x)
            else:
                nextinputs.append(
                    ak._v2.contents.NumpyArray(
                        nplike.broadcast_to(nplike.array([x]), (length,))
                    )
                )

        lengths = [len(x) for x in nextinputs]
        tags = nplike.repeat(nplike.arange(len(nextinputs)), lengths)
        index = nplike.concatenate([nplike.arange(x) for x in lengths])

        inner = ak._v2.contents.UnionArray(
            ak._v2.index.Index8(tags), ak._v2.index.Index64(index), nextinputs
        ).simplify_uniontype(merge=merge, mergebool=mergebool)

        offset = nplike.empty(len(lengths) + 1, dtype=np.int64)
        offset[0] = 0
        nplike.cumsum(lengths, out=offset[1:])
        out = ak._v2.contents.ListOffsetArray(ak._v2.index.Index64(offset), inner)

    else:

        def stack_contents(contents, length):
            tags = nplike.broadcast_to(
                nplike.arange(len(contents)), (length, len(contents))
            ).ravel()
            index = nplike.repeat(nplike.arange(length), len(contents))

            inner = ak._v2.contents.UnionArray(
                ak._v2.index.Index8(tags), ak._v2.index.Index64(index), contents
            ).simplify_uniontype(merge=merge, mergebool=mergebool)

            offset = nplike.arange(0, len(index) + 1, len(contents))
            return ak._v2.contents.ListOffsetArray(ak._v2.index.Index64(offset), inner)

        def action(inputs, depth, **kwargs):

            if depth == posaxis and any(
                isinstance(x, ak._v2.contents.Content) and x.is_OptionType
                for x in inputs
            ):
                nextinputs = []
                for x in inputs:
                    if x.is_OptionType and x.content.is_ListType:
                        nextinputs.append(fill_none(x, [], axis=0, highlevel=False))
                    else:
                        nextinputs.append(x)
                inputs = nextinputs

            # New axis above existing
            if depth == posaxis and all(
                isinstance(x, ak._v2.contents.Content)
                and x.is_ListType
                or not isinstance(x, ak._v2.contents.Content)
                for x in inputs
            ):
                length = max(
                    len(x) for x in inputs if isinstance(x, ak._v2.contents.Content)
                )
                nextinputs = []
                for x in inputs:
                    if isinstance(x, ak._v2.contents.Content):
                        nextinputs.append(x)
                    else:
                        nextinputs.append(
                            ak._v2.contents.ListOffsetArray(
                                ak._v2.index.Index64(
                                    nplike.arange(length + 1, dtype=np.int64)
                                ),
                                ak._v2.contents.NumpyArray(
                                    nplike.broadcast_to(nplike.array([x]), (length,))
                                ),
                            )
                        )

                return (stack_contents(nextinputs, length),)

            # New axis at end
            elif depth == posaxis and all(
                isinstance(x, ak._v2.contents.NumpyArray)
                or not isinstance(x, ak._v2.contents.Content)
                for x in inputs
            ):
                length = max(
                    len(x) for x in inputs if isinstance(x, ak._v2.contents.Content)
                )
                nextinputs = []
                for x in inputs:
                    if isinstance(x, ak._v2.contents.Content):
                        nextinputs.append(x)
                    else:
                        nextinputs.append(
                            ak._v2.contents.NumpyArray(
                                nplike.broadcast_to(nplike.array([x]), (length,))
                            )
                        )
                return (stack_contents(nextinputs, length),)

            elif any(
                x.purelist_depth == 1
                for x in inputs
                if isinstance(x, ak._v2.contents.Content)
            ):
                raise ak._v2._util.error(
                    ValueError(
                        "at least one array is not deep enough to concatenate at "
                        "axis={}".format(axis)
                    )
                )

            else:
                return None

        (out,) = ak._v2._broadcasting.broadcast_and_apply(
            layouts,
            action,
            behavior=behavior,
            numpy_to_regular=True,
            right_broadcast=False,
        )

    return ak._v2._util.wrap(out, behavior, highlevel)
