# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import awkward as ak
from awkward._nplikes.numpylike import NumpyMetadata

np = NumpyMetadata.instance()
cpu = ak._backends.NumpyBackend.instance()


def merge_union_of_lists(array, axis=-1, *, highlevel=True, behavior=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        axis (int): The dimension at which this operation is applied.
            The outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the  innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Simplifies unions of lists, e.g.

        >>> array = ak.concatenate(([["a", "b"]], [[1, 2, 3]]))

    into lists of unions, i.e.

        >>> ak.merge_union_of_lists(array, axis=0)
        <Array [['a', 'b'], [1, 2, 3]] type='2 * var * union[string, int64]'>
    """
    with ak._errors.OperationErrorContext(
        "ak.merge_union_of_lists",
        {"array": array, "axis": axis, "highlevel": highlevel, "behavior": behavior},
    ):
        return _impl(array, axis, highlevel, behavior)


def _impl(array, axis, highlevel, behavior):
    behavior = ak._util.behavior_of(array, behavior=behavior)
    layout = ak.to_layout(array, allow_record=False)

    def apply(layout, depth, backend, **kwargs):
        posaxis = ak._util.maybe_posaxis(layout, axis, depth)
        if depth < posaxis + 1 and layout.is_leaf:
            raise ak._errors.wrap_error(
                np.AxisError(f"axis={axis} exceeds the depth of this array ({depth})")
            )
        elif depth == posaxis + 1 and layout.is_union:
            if all(x.is_list for x in layout.contents):
                if not all(x.is_list for x in layout.contents):
                    raise ak._errors.wrap_error(NotImplementedError)

                contents = [
                    x.to_ListOffsetArray64(start_at_zero=True) for x in layout.contents
                ]
                index_nplike = backend.index_nplike

                # Compute new offsets of outermost list by taking union of sublist lengths!
                nums = index_nplike.asarray(
                    ak.contents.UnionArray.simplified(
                        layout.tags,
                        layout.index,
                        [
                            ak.contents.NumpyArray(x.stops.data - x.starts.data)
                            for x in contents
                        ],
                    ),
                    dtype=np.int64,
                )
                stops = index_nplike.cumsum(nums)
                offsets = index_nplike.empty(stops.size + 1, dtype=stops.dtype)
                offsets[0] = 0
                offsets[1:] = stops

                # The new tags are given by the original tags, repeated by the sublist counts
                new_tags = index_nplike.repeat(index_nplike.asarray(layout.tags), nums)

                # To compute the per-tag index, we first project each content, and then flatten it
                new_contents = [
                    ak._do.flatten(layout.project(tag), axis=1)
                    for tag, _ in enumerate(contents)
                ]
                # Now loop over each flattened array, and build a simple range index
                new_index = index_nplike.empty(new_tags.size, dtype=np.int64)
                for tag, flattened in enumerate(new_contents):
                    # Index into flattened content, and
                    index = index_nplike.arange(flattened.length, dtype=np.int64)
                    new_index[new_tags == tag] = index

                # Now we have a union over these contents
                content = ak.contents.UnionArray.simplified(
                    ak.index.Index8(new_tags), ak.index.Index64(new_index), new_contents
                )
                return ak.contents.ListOffsetArray(ak.index.Index64(offsets), content)

    out = ak._do.recursively_apply(layout, apply)
    return ak._util.wrap(out, highlevel=highlevel, behavior=behavior)
