# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

from awkward._v2.operations.structure.ak_fill_none import fill_none

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
        mergebool (bool): If True, boolean and nummeric data can be combined
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

    contents = [
        ak._v2.operations.convert.to_layout(
            x, allow_record=False if axis == 0 else True, allow_other=True
        )
        for x in arrays
    ]
    if not any(isinstance(x, (ak._v2.contents.Content,)) for x in contents):
        raise ValueError("need at least one array to concatenate")

    first_content = [x for x in contents if isinstance(x, (ak._v2.contents.Content,))][
        0
    ]

    posaxis = first_content.axis_wrap_if_negative(axis)
    maxdepth = max(
        [
            x.minmax_depth[1]
            for x in contents
            if isinstance(
                x,
                (ak._v2.contents.Content,),
            )
        ]
    )
    if not 0 <= posaxis < maxdepth:
        raise ValueError(
            "axis={0} is beyond the depth of this array or the depth of this array "
            "is ambiguous".format(axis)
        )
    for x in contents:
        if isinstance(x, ak._v2.contents.Content):
            if x.axis_wrap_if_negative(axis) != posaxis:
                raise ValueError(
                    "arrays to concatenate do not have the same depth for negative "
                    "axis={0}".format(axis)
                )

    if posaxis == 0:
        contents = [
            x
            if isinstance(x, ak._v2.contents.Content)
            else ak._v2.operations.convert.to_layout([x])
            for x in contents
        ]
        batch = [contents[0]]
        for x in contents[1:]:
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

            if depth == posaxis and all(
                isinstance(x, ak._v2.contents.Content)
                and x.is_ListType
                or (isinstance(x, ak._v2.contents.NumpyArray) and x.data.ndim > 1)
                or not isinstance(x, ak._v2.contents.Content)
                for x in inputs
            ):

                nplike = ak.nplike.of(*inputs)

                length = max(
                    [len(x) for x in inputs if isinstance(x, ak._v2.contents.Content)]
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

                counts = nplike.zeros(len(nextinputs[0]), dtype=np.int64)
                all_counts = []
                all_flatten = []

                for x in nextinputs:
                    o, f = x._offsets_and_flattened(1, 0)
                    o = nplike.asarray(o)
                    c = o[1:] - o[:-1]
                    nplike.add(counts, c, out=counts)
                    all_counts.append(c)
                    all_flatten.append(f)

                offsets = nplike.empty(len(nextinputs[0]) + 1, dtype=np.int64)
                offsets[0] = 0
                nplike.cumsum(counts, out=offsets[1:])

                offsets = ak._v2.index.Index64(offsets)

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
                    offsets, inner.simplify_uniontype(merge=merge, mergebool=mergebool)
                )

                return (out,)

            elif any(
                x.minmax_depth == (1, 1)
                for x in inputs
                if isinstance(x, ak._v2.contents.Content)
            ):
                raise ValueError(
                    "at least one array is not deep enough to concatenate at "
                    "axis={0}".format(axis)
                )

            else:
                return None

        out = ak._v2._broadcasting.broadcast_and_apply(
            contents,
            action,
            behavior=ak._v2._util.behavior_of(*arrays, behavior=behavior),
            allow_records=True,
            right_broadcast=False,
        )[0]

    return ak._v2._util.wrap(
        out, ak._v2._util.behavior_of(*arrays, behavior=behavior), highlevel
    )
