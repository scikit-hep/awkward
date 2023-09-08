# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
__all__ = ("concatenate",)
import awkward as ak
from awkward._backends.dispatch import backend_of
from awkward._backends.numpy import NumpyBackend
from awkward._behavior import behavior_of
from awkward._dispatch import high_level_function
from awkward._layout import maybe_posaxis, wrap_layout
from awkward._nplikes.numpylike import NumpyMetadata
from awkward._nplikes.shape import unknown_length
from awkward._regularize import regularize_axis
from awkward._typing import Sequence
from awkward.contents import Content
from awkward.operations.ak_fill_none import fill_none

np = NumpyMetadata.instance()
cpu = NumpyBackend.instance()


@ak._connect.numpy.implements("concatenate")
@high_level_function()
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
    # Dispatch
    if (
        # Is an array with a known backend
        backend_of(arrays, default=None)
        is not None
    ):
        yield (arrays,)
    else:
        yield arrays

    # Implementation
    return _impl(arrays, axis, mergebool, highlevel, behavior)


def _impl(arrays, axis, mergebool, highlevel, behavior):
    axis = regularize_axis(axis)
    behavior = behavior_of(*arrays, behavior=behavior)
    # Simple single-array, axis=0 fast-path
    if (
        # Is an array with a known backend
        backend_of(arrays, default=None)
        is not None
    ):
        # Convert the array to a layout object
        content = ak.operations.to_layout(arrays, allow_record=False, allow_other=False)
        # Only handle concatenation along `axis=0`
        # Let ambiguous depth arrays fall through
        if maybe_posaxis(content, axis, 1) == 0:
            return ak.operations.ak_flatten._impl(content, 1, highlevel, behavior)

    # Now that we're sure `arrays` is not a singular array
    backend = backend_of(*arrays, default=cpu, coerce_to_common=True)
    content_or_others = [
        x.to_backend(backend) if isinstance(x, ak.contents.Content) else x
        for x in (
            ak.operations.to_layout(
                x, allow_record=False if axis == 0 else True, allow_other=True
            )
            for x in arrays
        )
    ]

    contents = [x for x in content_or_others if isinstance(x, ak.contents.Content)]
    if len(contents) == 0:
        raise ValueError("need at least one array to concatenate")

    posaxis = maybe_posaxis(contents[0], axis, 1)
    maxdepth = max(
        x.minmax_depth[1]
        for x in content_or_others
        if isinstance(x, ak.contents.Content)
    )
    if posaxis is None or not 0 <= posaxis < maxdepth:
        raise ValueError(
            f"axis={axis} is beyond the depth of this array or the depth of this array "
            "is ambiguous"
        )
    for x in content_or_others:
        if isinstance(x, ak.contents.Content):
            if maybe_posaxis(x, axis, 1) != posaxis:
                raise ValueError(
                    "arrays to concatenate do not have the same depth for negative "
                    f"axis={axis}"
                )

    if posaxis == 0:
        content_or_others = [
            x if isinstance(x, ak.contents.Content) else ak.operations.to_layout([x])
            for x in content_or_others
        ]
        batches = [[content_or_others[0]]]
        for x in content_or_others[1:]:
            batch = batches[-1]
            if ak._do.mergeable(batch[-1], x, mergebool=mergebool):
                batch.append(x)
            else:
                batches.append([x])

        contents = [ak._do.mergemany(b) for b in batches]
        if len(contents) > 1:
            out = _merge_as_union(contents)
        else:
            out = contents[0]

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
            if any(
                x.minmax_depth == (1, 1)
                for x in inputs
                if isinstance(x, ak.contents.Content)
            ):
                raise ValueError(
                    "at least one array is not deep enough to concatenate at "
                    f"axis={axis}"
                )

            if depth != posaxis:
                return

            if any(isinstance(x, ak.contents.Content) and x.is_option for x in inputs):
                nextinputs = []
                for x in inputs:
                    if x.is_option and x.content.is_list:
                        nextinputs.append(fill_none(x, [], axis=0, highlevel=False))
                    else:
                        nextinputs.append(x)
                inputs = nextinputs

            # Ensure the lengths agree, taking known lengths over unknown lengths
            length = None
            for x in inputs:
                if isinstance(x, ak.contents.Content):
                    if length is None:
                        length = x.length
                    elif x.length is unknown_length:
                        continue
                    elif length is unknown_length:
                        length = x.length
                    elif length != x.length:
                        raise ValueError(
                            f"all arrays must have the same length for axis={axis}"
                        )
            assert length is not None

            if all(
                (isinstance(x, ak.contents.Content) and x.is_regular)
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
                                        backend.nplike.asarray([x]), (length,)
                                    )
                                ),
                                1,
                            )
                        )
                    sizes.append(regulararrays[-1].size)

                prototype = backend.index_nplike.empty(sum(sizes), dtype=np.int8)
                start = 0
                for tag, size in enumerate(sizes):
                    prototype[start : start + size] = tag
                    start += size

                tags = ak.index.Index8(
                    backend.index_nplike.reshape(
                        backend.index_nplike.broadcast_to(
                            prototype, (length, prototype.size)
                        ),
                        (-1,),
                    )
                )
                index = ak.contents.UnionArray.regular_index(tags, backend=backend)
                inner = ak.contents.UnionArray.simplified(
                    tags,
                    index,
                    [x._content for x in regulararrays],
                    mergebool=mergebool,
                )

                return (ak.contents.RegularArray(inner, prototype.size),)

            elif all(
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
                                        backend.index_nplike.shape_item_as_index(
                                            length + 1
                                        ),
                                        dtype=np.int64,
                                    ),
                                    nplike=backend.index_nplike,
                                ),
                                ak.contents.NumpyArray(
                                    backend.nplike.broadcast_to(
                                        backend.nplike.asarray([x]), (length,)
                                    )
                                ),
                            )
                        )

                counts = backend.index_nplike.zeros(
                    nextinputs[0].length, dtype=np.int64
                )
                all_counts = []
                all_flatten = []

                for x in nextinputs:
                    o, f = x._offsets_and_flattened(1, 1)
                    o = backend.index_nplike.asarray(o)
                    c = o[1:] - o[:-1]
                    backend.index_nplike.add(counts, c, maybe_out=counts)
                    all_counts.append(c)
                    all_flatten.append(f)

                offsets = backend.index_nplike.empty(
                    nextinputs[0].length + 1, dtype=np.int64
                )
                offsets[0] = 0
                backend.index_nplike.cumsum(counts, maybe_out=offsets[1:])

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

            else:
                return None

        out = ak._broadcasting.broadcast_and_apply(
            content_or_others,
            action,
            behavior=behavior,
            allow_records=True,
            right_broadcast=False,
        )[0]

    return wrap_layout(out, behavior, highlevel)


def _merge_as_union(
    contents: Sequence[Content], parameters=None
) -> ak.contents.UnionArray:
    length = sum([c.length for c in contents])
    first = contents[0]
    tags = ak.index.Index8.empty(length, first.backend.index_nplike)
    index = ak.index.Index64.empty(length, first.backend.index_nplike)

    offset = 0
    for i, content in enumerate(contents):
        content.backend.maybe_kernel_error(
            content.backend["awkward_UnionArray_filltags_const", tags.dtype.type](
                tags.data, offset, content.length, i
            )
        )
        content.backend.maybe_kernel_error(
            content.backend["awkward_UnionArray_fillindex_count", index.dtype.type](
                index.data, offset, content.length
            )
        )
        offset += content.length

    return ak.contents.UnionArray.simplified(
        tags, index, contents, parameters=parameters
    )
