# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

from awkward._v2.operations.ak_fill_none import fill_none

np = ak.nplike.NumpyMetadata.instance()


# @ak._v2._connect.numpy.implements("concatenate")
def concatenate(
    arrays, axis=0, merge=True, mergebool=True, highlevel=True, behavior=None
):
    """
    Args:
        arrays: Arrays to concatenate along any dimension.
        axis (int): The dimension at which this operation is applied. The
            outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc.
        merge (bool): If True, combine data into the same buffers wherever
            possible, eliminating unnecessary #ak.layout.UnionArray8_64 types
            at the expense of materializing #ak.layout.VirtualArray nodes.
        mergebool (bool): If True, boolean and numeric data can be combined
            into the same buffer, losing information about False vs `0` and
            True vs `1`; otherwise, they are kept in separate buffers with
            distinct types (using an #ak.layout.UnionArray8_64).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Returns an array with `arrays` concatenated. For `axis=0`, this means that
    one whole array follows another. For `axis=1`, it means that the `arrays`
    must have the same lengths and nested lists are each concatenated,
    element for element, and similarly for deeper levels.
    """
    with ak._v2._util.OperationErrorContext(
        "ak._v2.concatenate",
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
    # Simple single-array, axis=0 fast-path
    single_nplike = ak.nplike.of(arrays)
    if (
        # Is an Awkward Content
        isinstance(arrays, ak._v2.contents.Content)
        # Is a NumPy Array
        or ak.nplike.is_numpy_buffer(arrays)
        # Is an array with a known NumpyLike
        or single_nplike is not ak.nplike.Numpy.instance()
    ):
        # Convert the array to a layout object
        content = ak._v2.operations.to_layout(
            arrays, allow_record=False, allow_other=False
        )
        # Only handle concatenation along `axis=0`
        # Let ambiguous depth arrays fall through
        if content.axis_wrap_if_negative(axis) == 0:
            return ak._v2.operations.ak_flatten._impl(content, 1, highlevel, behavior)

    content_or_others = [
        ak._v2.operations.to_layout(
            x, allow_record=False if axis == 0 else True, allow_other=True
        )
        for x in arrays
    ]

    contents = [x for x in content_or_others if isinstance(x, ak._v2.contents.Content)]
    if len(contents) == 0:
        raise ak._v2._util.error(ValueError("need at least one array to concatenate"))

    posaxis = contents[0].axis_wrap_if_negative(axis)
    maxdepth = max(
        x.minmax_depth[1]
        for x in content_or_others
        if isinstance(x, ak._v2.contents.Content)
    )
    if not 0 <= posaxis < maxdepth:
        raise ak._v2._util.error(
            ValueError(
                "axis={} is beyond the depth of this array or the depth of this array "
                "is ambiguous".format(axis)
            )
        )
    for x in content_or_others:
        if isinstance(x, ak._v2.contents.Content):
            if x.axis_wrap_if_negative(axis) != posaxis:
                raise ak._v2._util.error(
                    ValueError(
                        "arrays to concatenate do not have the same depth for negative "
                        "axis={}".format(axis)
                    )
                )

    if posaxis == 0:
        content_or_others = [
            x
            if isinstance(x, ak._v2.contents.Content)
            else ak._v2.operations.to_layout([x])
            for x in content_or_others
        ]
        batch = [content_or_others[0]]
        for x in content_or_others[1:]:
            if batch[-1].mergeable(x, mergebool=mergebool):
                batch.append(x)
            else:
                collapsed = batch[0].mergemany(batch[1:])
                batch = [collapsed.merge_as_union(x)]

        out = batch[0].mergemany(batch[1:])
        if isinstance(out, ak._v2.contents.unionarray.UnionArray):
            out = out.simplify_uniontype(merge=merge, mergebool=mergebool)

    else:

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

            if depth == posaxis:
                nplike = ak.nplike.of(*inputs)

                length = ak._v2._typetracer.UnknownLength
                for x in inputs:
                    if isinstance(x, ak._v2.contents.Content):
                        if not ak._v2._util.isint(length):
                            length = x.length
                        elif length != x.length and ak._v2._util.isint(x.length):
                            raise ak._v2._util.error(
                                ValueError(
                                    "all arrays must have the same length for "
                                    "axis={}".format(axis)
                                )
                            )

            if depth == posaxis and all(
                isinstance(x, ak._v2.contents.Content)
                and x.is_RegularType
                or (isinstance(x, ak._v2.contents.NumpyArray) and x.data.ndim > 1)
                or not isinstance(x, ak._v2.contents.Content)
                for x in inputs
            ):
                regulararrays = []
                sizes = []
                for x in inputs:
                    if isinstance(x, ak._v2.contents.RegularArray):
                        regulararrays.append(x)
                    elif isinstance(x, ak._v2.contents.NumpyArray):
                        regulararrays.append(x.toRegularArray())
                    else:
                        regulararrays.append(
                            ak._v2.contents.RegularArray(
                                ak._v2.contents.NumpyArray(
                                    nplike.broadcast_to(nplike.array([x]), (length,))
                                ),
                                1,
                            )
                        )
                    sizes.append(regulararrays[-1].size)

                prototype = nplike.empty(sum(sizes), np.int8)
                start = 0
                for tag, size in enumerate(sizes):
                    prototype[start : start + size] = tag
                    start += size

                tags = ak._v2.index.Index8(nplike.tile(prototype, length))
                index = ak._v2.contents.UnionArray.regular_index(tags)
                inner = ak._v2.contents.UnionArray(
                    tags, index, [x._content for x in regulararrays]
                )

                out = ak._v2.contents.RegularArray(
                    inner.simplify_uniontype(merge=merge, mergebool=mergebool),
                    len(prototype),
                )
                return (out,)

            elif depth == posaxis and all(
                isinstance(x, ak._v2.contents.Content)
                and x.is_ListType
                or (isinstance(x, ak._v2.contents.NumpyArray) and x.data.ndim > 1)
                or not isinstance(x, ak._v2.contents.Content)
                for x in inputs
            ):
                nextinputs = []
                for x in inputs:
                    if isinstance(x, ak._v2.contents.Content):
                        nextinputs.append(x)
                    else:
                        nextinputs.append(
                            ak._v2.contents.ListOffsetArray(
                                ak._v2.index.Index64(
                                    nplike.index_nplike.arange(
                                        length + 1, dtype=np.int64
                                    ),
                                    nplike=nplike,
                                ),
                                ak._v2.contents.NumpyArray(
                                    nplike.broadcast_to(nplike.array([x]), (length,))
                                ),
                            )
                        )

                counts = nplike.zeros(len(nextinputs[0]), dtype=np.int64)
                all_counts = []
                all_flatten = []

                for x in nextinputs:
                    o, f = x._offsets_and_flattened(1, 0)
                    o = nplike.index_nplike.asarray(o)
                    c = o[1:] - o[:-1]
                    nplike.add(counts, c, out=counts)
                    all_counts.append(c)
                    all_flatten.append(f)

                offsets = nplike.index_nplike.empty(
                    len(nextinputs[0]) + 1, dtype=np.int64
                )
                offsets[0] = 0
                nplike.index_nplike.cumsum(counts, out=offsets[1:])

                offsets = ak._v2.index.Index64(offsets, nplike=nplike)

                inner = ak._v2.contents.UnionArray(
                    ak._v2.index.Index8.empty(len(offsets) - 1, nplike),
                    ak._v2.index.Index64.empty(len(offsets) - 1, nplike),
                    all_flatten,
                )

                tags, index = inner._nested_tags_index(
                    offsets,
                    [ak._v2.index.Index64(x) for x in all_counts],
                )

                inner = ak._v2.contents.UnionArray(tags, index, all_flatten)

                out = ak._v2.contents.ListOffsetArray(
                    offsets,
                    inner.simplify_uniontype(merge=merge, mergebool=mergebool),
                )

                return (out,)

            elif any(
                x.minmax_depth == (1, 1)
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

        out = ak._v2._broadcasting.broadcast_and_apply(
            content_or_others,
            action,
            behavior=ak._v2._util.behavior_of(*arrays, behavior=behavior),
            allow_records=True,
            right_broadcast=False,
        )[0]

    return ak._v2._util.wrap(
        out, ak._v2._util.behavior_of(*arrays, behavior=behavior), highlevel
    )
