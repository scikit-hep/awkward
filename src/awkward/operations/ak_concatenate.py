# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak
from awkward.operations.ak_fill_none import fill_none

np = ak._nplikes.NumpyMetadata.instance()
cpu = ak._backends.NumpyBackend.instance()


@ak._connect.numpy.implements("concatenate")
def concatenate(arrays, axis=0, *, mergebool=True, highlevel=True, behavior=None):
    """
    Args:
        arrays: Array-like data (anything #ak.to_layout recognizes).
        axis (int): The dimension at which this operation is applied. The
            outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc.
        mergebool (bool): If True, boolean and numeric data can be combined
            into the same buffer, losing information about False vs `0` and
            True vs `1`; otherwise, they are kept in separate buffers with
            distinct types (using an #ak.contents.UnionArray).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Returns an array with `arrays` concatenated. For `axis=0`, this means that
    one whole array follows another. For `axis=1`, it means that the `arrays`
    must have the same lengths and nested lists are each concatenated,
    element for element, and similarly for deeper levels.
    """
    with ak._errors.OperationErrorContext(
        "ak.concatenate",
        dict(
            arrays=arrays,
            axis=axis,
            mergebool=mergebool,
            highlevel=highlevel,
            behavior=behavior,
        ),
    ):
        return _impl(arrays, axis, mergebool, highlevel, behavior)


def _impl(arrays, axis, mergebool, highlevel, behavior):
    # Simple single-array, axis=0 fast-path
    behavior = ak._util.behavior_of(*arrays, behavior=behavior)
    if (
        # Is an Awkward Content
        isinstance(arrays, ak.contents.Content)
        # Is an array with a known NumpyLike
        or ak._nplikes.nplike_of(arrays, default=None) is not None
    ):
        # Convert the array to a layout object
        content = ak.operations.to_layout(arrays, allow_record=False, allow_other=False)
        # Only handle concatenation along `axis=0`
        # Let ambiguous depth arrays fall through
        if ak._util.maybe_posaxis(content, axis, 1) == 0:
            return ak.operations.ak_flatten._impl(content, 1, highlevel, behavior)

    content_or_others = [
        ak.operations.to_layout(
            x, allow_record=False if axis == 0 else True, allow_other=True
        )
        for x in arrays
    ]

    contents = [x for x in content_or_others if isinstance(x, ak.contents.Content)]
    if len(contents) == 0:
        raise ak._errors.wrap_error(
            ValueError("need at least one array to concatenate")
        )

    posaxis = ak._util.maybe_posaxis(contents[0], axis, 1)
    maxdepth = max(
        x.minmax_depth[1]
        for x in content_or_others
        if isinstance(x, ak.contents.Content)
    )
    if posaxis is None or not 0 <= posaxis < maxdepth:
        raise ak._errors.wrap_error(
            ValueError(
                "axis={} is beyond the depth of this array or the depth of this array "
                "is ambiguous".format(axis)
            )
        )
    for x in content_or_others:
        if isinstance(x, ak.contents.Content):
            if ak._util.maybe_posaxis(x, axis, 1) != posaxis:
                raise ak._errors.wrap_error(
                    ValueError(
                        "arrays to concatenate do not have the same depth for negative "
                        "axis={}".format(axis)
                    )
                )

    if posaxis == 0:
        content_or_others = [
            x if isinstance(x, ak.contents.Content) else ak.operations.to_layout([x])
            for x in content_or_others
        ]
        batch = [content_or_others[0]]
        for x in content_or_others[1:]:
            if ak._do.mergeable(batch[-1], x, mergebool=mergebool):
                batch.append(x)
            else:
                collapsed = ak._do.mergemany(batch)
                batch = [ak._do.merge_as_union(collapsed, x)]

        out = ak._do.mergemany(batch)

        if isinstance(out, ak.contents.UnionArray):
            out = type(out).simplified(
                out._tags,
                out._index,
                out._contents,
                parameters=out._parameters,
                mergebool=mergebool,
            )

    else:

        def action(inputs, depth, **kwargs):
            if depth == posaxis and any(
                isinstance(x, ak.contents.Content) and x.is_option for x in inputs
            ):
                nextinputs = []
                for x in inputs:
                    if x.is_option and x.content.is_list:
                        nextinputs.append(fill_none(x, [], axis=0, highlevel=False))
                    else:
                        nextinputs.append(x)
                inputs = nextinputs

            if depth == posaxis:
                backend = ak._backends.backend_of(*inputs, default=cpu)

                length = ak._typetracer.UnknownLength
                for x in inputs:
                    if isinstance(x, ak.contents.Content):
                        if not ak._util.is_integer(length):
                            length = x.length
                        elif length != x.length and ak._util.is_integer(x.length):
                            raise ak._errors.wrap_error(
                                ValueError(
                                    "all arrays must have the same length for "
                                    "axis={}".format(axis)
                                )
                            )

            if depth == posaxis and all(
                isinstance(x, ak.contents.Content)
                and x.is_regular
                or (isinstance(x, ak.contents.NumpyArray) and x.data.ndim > 1)
                or not isinstance(x, ak.contents.Content)
                for x in inputs
            ):
                regulararrays = []
                sizes = []
                for x in inputs:
                    if isinstance(x, ak.contents.RegularArray):
                        regulararrays.append(x)
                    elif isinstance(x, ak.contents.NumpyArray):
                        regulararrays.append(x.to_RegularArray())
                    else:
                        regulararrays.append(
                            ak.contents.RegularArray(
                                ak.contents.NumpyArray(
                                    backend.nplike.broadcast_to(
                                        backend.nplike.array([x]), (length,)
                                    )
                                ),
                                1,
                            )
                        )
                    sizes.append(regulararrays[-1].size)

                prototype = backend.index_nplike.empty(sum(sizes), np.int8)
                start = 0
                for tag, size in enumerate(sizes):
                    prototype[start : start + size] = tag
                    start += size

                tags = ak.index.Index8(backend.index_nplike.tile(prototype, length))
                index = ak.contents.UnionArray.regular_index(tags, backend=backend)
                inner = ak.contents.UnionArray.simplified(
                    tags,
                    index,
                    [x._content for x in regulararrays],
                    mergebool=mergebool,
                )

                return (ak.contents.RegularArray(inner, len(prototype)),)

            elif depth == posaxis and all(
                isinstance(x, ak.contents.Content)
                and x.is_list
                or (isinstance(x, ak.contents.NumpyArray) and x.data.ndim > 1)
                or not isinstance(x, ak.contents.Content)
                for x in inputs
            ):
                nextinputs = []
                for x in inputs:
                    if isinstance(x, ak.contents.Content):
                        nextinputs.append(x)
                    else:
                        nextinputs.append(
                            ak.contents.ListOffsetArray(
                                ak.index.Index64(
                                    backend.index_nplike.arange(
                                        length + 1, dtype=np.int64
                                    ),
                                    nplike=backend.index_nplike,
                                ),
                                ak.contents.NumpyArray(
                                    backend.nplike.broadcast_to(
                                        backend.nplike.array([x]), (length,)
                                    )
                                ),
                            )
                        )

                counts = backend.index_nplike.zeros(len(nextinputs[0]), dtype=np.int64)
                all_counts = []
                all_flatten = []

                for x in nextinputs:
                    o, f = x._offsets_and_flattened(1, 1)
                    o = backend.index_nplike.asarray(o)
                    c = o[1:] - o[:-1]
                    backend.index_nplike.add(counts, c, out=counts)
                    all_counts.append(c)
                    all_flatten.append(f)

                offsets = backend.index_nplike.empty(
                    len(nextinputs[0]) + 1, dtype=np.int64
                )
                offsets[0] = 0
                backend.index_nplike.cumsum(counts, out=offsets[1:])

                offsets = ak.index.Index64(offsets, nplike=backend.index_nplike)

                tags, index = ak.contents.UnionArray.nested_tags_index(
                    offsets,
                    [ak.index.Index64(x) for x in all_counts],
                    backend=backend,
                )

                inner = ak.contents.UnionArray.simplified(
                    tags, index, all_flatten, mergebool=mergebool
                )

                return (ak.contents.ListOffsetArray(offsets, inner),)

            elif any(
                x.minmax_depth == (1, 1)
                for x in inputs
                if isinstance(x, ak.contents.Content)
            ):
                raise ak._errors.wrap_error(
                    ValueError(
                        "at least one array is not deep enough to concatenate at "
                        "axis={}".format(axis)
                    )
                )

            else:
                return None

        out = ak._broadcasting.broadcast_and_apply(
            content_or_others,
            action,
            behavior=behavior,
            allow_records=True,
            right_broadcast=False,
        )[0]

    return ak._util.wrap(out, behavior, highlevel)
