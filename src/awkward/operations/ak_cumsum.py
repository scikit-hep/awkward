# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak.nplikes.NumpyMetadata.instance()


@ak._connect.numpy.implements("cumsum")
def cumsum(
    array, axis=None, keepdims=False, mask_identity=False, flatten_records=False
):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        axis (None or int): If None, compute the cumulative sum over the
            flattened array; if an int, compute the cumsum along that axis:
            `0` is the outermost, `1` is the first level of nested lists, etc.,
            and negative `axis` counts from the innermost: `-1` is the innermost,
            `-2` is the next level up, etc.

    Performs the cumulative sum over `array` (many types supported, including all
    Awkward Arrays and Records). This operation is the same as NumPy's
    [cumsum](https://docs.scipy.org/doc/numpy/reference/generated/numpy.cumsum.html)
    if all lists at a given dimension have the same length and no None values,
    but it generalizes to cases where they do not.

    For example, consider this `array`, in which all lists at a given dimension
    have the same length.

        ak.Array([[1, 10],
                  [2, 20],
                  [3, 30]])

    A cumsum over `axis=-1` accumulates over the inner lists:

        >>> ak.cumsum(array, axis=-1)
        <Array [[1, 11], [2, 22], [3, 33]] type='3 * 2 * int64'>

    while a cumsum over `axis=0` accumulates over the outer lists:

        >>> ak.cumsum(array, axis=0)
        <Array [[1, 10], [3, 30], [6, 60]] type='3 * 2 * int64'>

    Now with some values missing,

        ak.Array([[1,    10],
                  [None, 20],
                  [3,  None]])

    The cumsum over `axis=-1` results in

        >>> ak.cumsum(array, axis=-1)
        <Array [[1, 11], [None, 20], [3, None]] type='3 * var * ?int64'>

    and the cumsum over `axis=0` results in

        >>> ak.cumsum(array, axis=0)
        <Array [[1, 10], [None, 30], [4, None]] type='3 * var * ?int64'>

    See also #ak.nansum.
    """
    with ak._errors.OperationErrorContext(
        "ak.cumsum",
        dict(
            array=array,
            axis=axis,
        ),
    ):
        return _impl(array, axis)


def _impl(array, axis):
    layout = ak.operations.to_layout(array, allow_record=False, allow_other=False)

    if axis is None:
        if not layout.nplike.known_data or not layout.nplike.known_shape:
            raise ak._errors.wrap_error(NotImplementedError)  # TODO: FIXME

        else:

            def map(x):
                return layout.nplike.cumsum(x)

        def reduce(xs):
            if len(xs) == 1:
                return xs[0]
            else:
                return layout.nplike.add(xs[0], reduce(xs[1:]))

        return layout.nplike.concatenate(
            [
                map(x)
                for x in layout.completely_flatten(
                    function_name="ak.sum", flatten_records=False
                )
            ]
        )

    else:
        behavior = ak._util.behavior_of(array)
        out = layout.cumsum(axis=axis, behavior=behavior)
        if isinstance(out, (ak.contents.Content, ak.record.Record)):
            return ak._util.wrap(out, behavior)
        else:
            return out
