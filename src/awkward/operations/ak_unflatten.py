# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numbers

import awkward as ak

np = ak.nplikes.NumpyMetadata.instance()


def unflatten(array, counts, axis=0, highlevel=True, behavior=None):
    """
    Args:
        array: Data to create an array with an additional level from.
        counts (int or array): Number of elements the new level should have.
            If an integer, the new level will be regularly sized; otherwise,
            it will consist of variable-length lists with the given lengths.
        axis (int): The dimension at which this operation is applied. The
            outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Returns an array with an additional level of nesting. This is roughly the
    inverse of #ak.flatten, where `counts` were obtained by #ak.num (both with
    `axis=1`).

    For example,

        >>> original = ak.Array([[0, 1, 2], [], [3, 4], [5], [6, 7, 8, 9]])
        >>> counts = ak.num(original)
        >>> array = ak.flatten(original)
        >>> counts
        <Array [3, 0, 2, 1, 4] type='5 * int64'>
        >>> array
        <Array [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] type='10 * int64'>
        >>> ak.unflatten(array, counts)
        <Array [[0, 1, 2], [], ... [5], [6, 7, 8, 9]] type='5 * var * int64'>

    An inner dimension can be unflattened by setting the `axis` parameter, but
    operations like this constrain the `counts` more tightly.

    For example, we can subdivide an already divided list:

        >>> original = ak.Array([[1, 2, 3, 4], [], [5, 6, 7], [8, 9]])
        >>> print(ak.unflatten(original, [2, 2, 1, 2, 1, 1], axis=1))
        [[[1, 2], [3, 4]], [], [[5], [6, 7]], [[8], [9]]]

    But the counts have to add up to the lengths of those lists. We can't mix
    values from the first `[1, 2, 3, 4]` with values from the next `[5, 6, 7]`.

        >>> print(ak.unflatten(original, [2, 1, 2, 2, 1, 1], axis=1))
        Traceback (most recent call last):
        ...
        ValueError: structure imposed by 'counts' does not fit in the array at axis=1

    Also note that new lists created by this function cannot cross partitions
    (which is only possible at `axis=0`, anyway).

    See also #ak.num and #ak.flatten.
    """
    with ak._errors.OperationErrorContext(
        "ak.unflatten",
        dict(
            array=array,
            counts=counts,
            axis=axis,
            highlevel=highlevel,
            behavior=behavior,
        ),
    ):
        return _impl(array, counts, axis, highlevel, behavior)


def _impl(array, counts, axis, highlevel, behavior):
    nplike = ak.nplikes.nplike_of(array)

    layout = ak.operations.to_layout(array, allow_record=False, allow_other=False)
    behavior = ak._util.behavior_of(array, behavior=behavior)

    if isinstance(counts, (numbers.Integral, np.integer)):
        current_offsets = None
    else:
        counts = ak.operations.to_layout(counts, allow_record=False, allow_other=False)

        if counts.is_option:
            mask = counts.mask_as_bool(valid_when=False)
            counts = counts.to_numpy(allow_missing=True)
            counts = ak.nplikes.numpy.ma.filled(counts, 0)
        elif counts.is_numpy or counts.is_unknown:
            counts = counts.to_numpy(allow_missing=False)
            mask = False

        if counts.ndim != 1:
            raise ak._errors.wrap_error(ValueError("counts must be one-dimensional"))
        if not issubclass(counts.dtype.type, np.integer):
            raise ak._errors.wrap_error(ValueError("counts must be integers"))

        current_offsets = [nplike.index_nplike.empty(len(counts) + 1, np.int64)]
        current_offsets[0][0] = 0
        nplike.index_nplike.cumsum(counts, out=current_offsets[0][1:])

    def doit(layout):
        if isinstance(counts, (numbers.Integral, np.integer)):
            if counts < 0 or counts > len(layout):
                raise ak._errors.wrap_error(
                    ValueError("too large counts for array or negative counts")
                )
            out = ak.contents.RegularArray(layout, counts)

        else:
            position = (
                nplike.index_nplike.searchsorted(
                    current_offsets[0],
                    nplike.index_nplike.array([len(layout)]),
                    side="right",
                )[0]
                - 1
            )
            if position >= len(current_offsets[0]) or current_offsets[0][
                position
            ] != len(layout):
                raise ak._errors.wrap_error(
                    ValueError(
                        "structure imposed by 'counts' does not fit in the array or partition "
                        "at axis={}".format(axis)
                    )
                )

            offsets = current_offsets[0][: position + 1]
            current_offsets[0] = current_offsets[0][position:] - len(layout)

            out = ak.contents.ListOffsetArray(ak.index.Index64(offsets), layout)
            if not isinstance(mask, (bool, np.bool_)):
                index = ak.index.Index8(
                    nplike.asarray(mask).astype(np.int8), nplike=nplike
                )
                out = ak.contents.ByteMaskedArray(index, out, valid_when=False)

        return out

    if axis == 0 or layout.axis_wrap_if_negative(axis) == 0:
        out = doit(layout)

    else:

        def transform(layout, depth, posaxis):
            # Pack the current layout. This ensures that the `counts` array,
            # which is computed with these layouts applied, aligns with the
            # internal layout to be unflattened (#910)
            layout = layout.packed()

            posaxis = layout.axis_wrap_if_negative(posaxis)
            if posaxis == depth and layout.is_list:
                # We are one *above* the level where we want to apply this.
                listoffsetarray = layout.toListOffsetArray64(True)
                outeroffsets = nplike.index_nplike.asarray(listoffsetarray.offsets)

                content = doit(listoffsetarray.content[: outeroffsets[-1]])
                if isinstance(content, ak.contents.ByteMaskedArray):
                    inneroffsets = nplike.index_nplike.asarray(content.content.offsets)
                elif isinstance(content, ak.contents.RegularArray):
                    inneroffsets = nplike.index_nplike.asarray(
                        content.toListOffsetArray64(True).offsets
                    )
                else:
                    inneroffsets = nplike.index_nplike.asarray(content.offsets)

                positions = (
                    nplike.index_nplike.searchsorted(
                        inneroffsets, outeroffsets, side="right"
                    )
                    - 1
                )
                if not nplike.index_nplike.array_equal(
                    inneroffsets[positions], outeroffsets
                ):
                    raise ak._errors.wrap_error(
                        ValueError(
                            "structure imposed by 'counts' does not fit in the array or partition "
                            "at axis={}".format(axis)
                        )
                    )
                positions[0] = 0

                return ak.contents.ListOffsetArray(ak.index.Index64(positions), content)

            else:
                return layout

        out = transform(layout, depth=1, posaxis=axis)

    if current_offsets is not None and not (
        len(current_offsets[0]) == 1 and current_offsets[0][0] == 0
    ):
        raise ak._errors.wrap_error(
            ValueError(
                "structure imposed by 'counts' does not fit in the array or partition "
                "at axis={}".format(axis)
            )
        )

    return ak._util.wrap(out, behavior, highlevel)
