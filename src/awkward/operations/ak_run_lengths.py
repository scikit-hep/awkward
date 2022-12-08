# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()
cpu = ak._backends.NumpyBackend.instance()


def run_lengths(array, *, highlevel=True, behavior=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Computes the lengths of sequences of identical values at the deepest level
    of nesting, returning an array with the same structure but with `int64` type.

    For example,

        >>> array = ak.Array([1.1, 1.1, 1.1, 2.2, 3.3, 3.3, 4.4, 4.4, 5.5])
        >>> ak.run_lengths(array)
        <Array [3, 1, 2, 2, 1] type='5 * int64'>

    There are 3 instances of 1.1, followed by 1 instance of 2.2, 2 instances of 3.3,
    2 instances of 4.4, and 1 instance of 5.5.

    The order and uniqueness of the input data doesn't matter,

        >>> array = ak.Array([1.1, 1.1, 1.1, 5.5, 4.4, 4.4, 1.1, 1.1, 5.5])
        >>> ak.run_lengths(array)
        <Array [3, 1, 2, 2, 1] type='5 * int64'>

    just the difference between each value and its neighbors.

    The data can be nested, but runs don't cross list boundaries.

        >>> array = ak.Array([[1.1, 1.1, 1.1, 2.2, 3.3], [3.3, 4.4], [4.4, 5.5]])
        >>> ak.run_lengths(array)
        <Array [[3, 1, 1], [1, 1], [1, 1]] type='3 * var * int64'>

    This function recognizes strings as distinguishable values.

        >>> array = ak.Array([["one", "one"], ["one", "two", "two"], ["three", "two", "two"]])
        >>> ak.run_lengths(array)
        <Array [[2], [1, 2], [1, 2]] type='3 * var * int64'>

    Note that this can be combined with #ak.argsort and #ak.unflatten to compute
    a "group by" operation:

        >>> array = ak.Array([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 1, "y": 1.1},
        ...                   {"x": 3, "y": 3.3}, {"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}])
        >>> sorted = array[ak.argsort(array.x)]
        >>> sorted.x
        <Array [1, 1, 1, 2, 2, 3] type='6 * int64'>
        >>> ak.run_lengths(sorted.x)
        <Array [3, 2, 1] type='3 * int64'>
        >>> ak.unflatten(sorted, ak.run_lengths(sorted.x)).show()
        [[{x: 1, y: 1.1}, {x: 1, y: 1.1}, {x: 1, y: 1.1}],
         [{x: 2, y: 2.2}, {x: 2, y: 2.2}],
         [{x: 3, y: 3.3}]]

    Unlike a database "group by," this operation can be applied in bulk to many sublists
    (though the run lengths need to be fully flattened to be used as `counts` for
    #ak.unflatten, and you need to specify `axis=-1` as the depth).

        >>> array = ak.Array([[{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 1, "y": 1.1}],
        ...                   [{"x": 3, "y": 3.3}, {"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}]])
        >>> sorted = array[ak.argsort(array.x)]
        >>> sorted.x
        <Array [[1, 1, 2], [1, 2, 3]] type='2 * var * int64'>
        >>> ak.run_lengths(sorted.x)
        <Array [[2, 1], [1, 1, 1]] type='2 * var * int64'>
        >>> counts = ak.flatten(ak.run_lengths(sorted.x), axis=None)
        >>> ak.unflatten(sorted, counts, axis=-1).show()
        [[[{x: 1, y: 1.1}, {x: 1, y: 1.1}], [{x: 2, y: 2.2}]],
         [[{x: 1, y: 1.1}], [{x: 2, y: 2.2}], [{x: 3, y: 3.3}]]]

    See also #ak.num, #ak.argsort, #ak.unflatten.
    """
    with ak._errors.OperationErrorContext(
        "ak.run_lengths",
        dict(
            array=array,
            highlevel=highlevel,
            behavior=behavior,
        ),
    ):
        return _impl(array, highlevel, behavior)


def _impl(array, highlevel, behavior):
    backend = ak._backends.backend_of(array, default=cpu)

    def lengths_of(data, offsets):
        if len(data) == 0:
            return backend.index_nplike.empty(0, np.int64), offsets
        else:
            diffs = data[1:] != data[:-1]

            if isinstance(diffs, ak.highlevel.Array):
                diffs = backend.nplike.asarray(diffs)
            # Do we have list boundaries to consider?
            if offsets is not None:
                # When checking to see whether one element equals its following neighbour
                # we also want to break runs at list boundaries. The comparison for offset `i`
                # occurs at `i-1` in `diffs`. Consider this example with an empty sublist:
                #                                data = [1 1 2 2 2 3 4 4 5  ]
                #                             offsets = [0           6   9 9]
                #                        (data) diffs = [  0 1 0 0 1 1 0 1  ]
                #                        diffs index  = [  0 1 2 3 4 5 6 7  ]
                #                                      boundary diff ^
                # To consider only the interior boundaries, we ignore the start and end
                # offset values. These can be repeated with empty sublists, so we mask them out.
                is_interior = backend.nplike.logical_and(
                    0 < offsets, offsets < len(data) - 1
                )
                interior_offsets = offsets[is_interior]
                diffs[interior_offsets - 1] = True
            positions = backend.index_nplike.nonzero(diffs)[0]
            full_positions = backend.index_nplike.empty(len(positions) + 2, np.int64)
            full_positions[0] = 0
            full_positions[-1] = len(data)
            full_positions[1:-1] = positions + 1

            nextcontent = full_positions[1:] - full_positions[:-1]
            if offsets is None:
                nextoffsets = None
            else:
                nextoffsets = backend.index_nplike.searchsorted(
                    full_positions, offsets, side="left"
                )
            return nextcontent, nextoffsets

    def action(layout, **kwargs):
        if layout.branch_depth == (False, 1):
            if layout.is_indexed:
                layout = layout.project()

            if (
                layout.parameter("__array__") == "string"
                or layout.parameter("__array__") == "bytestring"
            ):
                nextcontent, _ = lengths_of(ak.highlevel.Array(layout), None)
                return ak.contents.NumpyArray(nextcontent)

            if not isinstance(layout, (ak.contents.NumpyArray, ak.contents.EmptyArray)):
                raise ak._errors.wrap_error(
                    NotImplementedError("run_lengths on " + type(layout).__name__)
                )

            nextcontent, _ = lengths_of(backend.nplike.asarray(layout), None)
            return ak.contents.NumpyArray(nextcontent)

        elif layout.branch_depth == (False, 2):
            if layout.is_indexed:
                layout = layout.project()

            if not layout.is_list:
                raise ak._errors.wrap_error(
                    NotImplementedError("run_lengths on " + type(layout).__name__)
                )

            if (
                layout.content.parameter("__array__") == "string"
                or layout.content.parameter("__array__") == "bytestring"
            ):
                # We also want to trim the _upper_ bound of content,
                # so we manually convert the list type to zero-based
                listoffsetarray = layout.to_ListOffsetArray64(False)
                offsets = backend.index_nplike.asarray(listoffsetarray.offsets)
                content = listoffsetarray.content[offsets[0] : offsets[-1]]

                if content.is_indexed:
                    content = content.project()

                nextcontent, nextoffsets = lengths_of(
                    ak.highlevel.Array(content), offsets - offsets[0]
                )
                return ak.contents.ListOffsetArray(
                    ak.index.Index64(nextoffsets),
                    ak.contents.NumpyArray(nextcontent),
                )

            listoffsetarray = layout.to_ListOffsetArray64(False)
            offsets = backend.index_nplike.asarray(listoffsetarray.offsets)
            content = listoffsetarray.content[offsets[0] : offsets[-1]]

            if content.is_indexed:
                content = content.project()

            if not isinstance(
                content, (ak.contents.NumpyArray, ak.contents.EmptyArray)
            ):
                raise ak._errors.wrap_error(
                    NotImplementedError(
                        "run_lengths on "
                        + type(layout).__name__
                        + " with content "
                        + type(content).__name__
                    )
                )

            nextcontent, nextoffsets = lengths_of(
                backend.nplike.asarray(content), offsets - offsets[0]
            )
            return ak.contents.ListOffsetArray(
                ak.index.Index64(nextoffsets),
                ak.contents.NumpyArray(nextcontent),
            )
        else:
            return None

    layout = ak.operations.to_layout(array, allow_record=False, allow_other=False)
    behavior = ak._util.behavior_of(array, behavior=behavior)

    out = ak._do.recursively_apply(layout, action, behavior)
    return ak._util.wrap(out, behavior, highlevel)
