# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()


@ak._connect.numpy.implements("argsort")
def argsort(
    array, axis=-1, *, ascending=True, stable=True, highlevel=True, behavior=None
):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        axis (int): The dimension at which this operation is applied. The
            outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc.
        ascending (bool): If True, the first value in each sorted group
            will be smallest, the last value largest; if False, the order
            is from largest to smallest.
        stable (bool): If True, use a stable sorting algorithm (introsort:
            a hybrid of quicksort, heapsort, and insertion sort); if False,
            use a sorting algorithm that is not guaranteed to be stable
            (heapsort).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Returns an array of integer indexes that would sort the array if applied
    as an integer-array slice.

    For example,

        >>> ak.argsort(ak.Array([[7.7, 5.5, 7.7], [], [2.2], [8.8, 2.2]]))
        <Array [[1, 0, 2], [], [0], [1, 0]] type='4 * var * int64'>

    The result of this function can be used to index other arrays with the
    same shape:

        >>> data = ak.Array([[7, 5, 7], [], [2], [8, 2]])
        >>> index = ak.argsort(data)
        >>> index
        <Array [[1, 0, 2], [], [0], [1, 0]] type='4 * var * int64'>
        >>> data[index]
        <Array [[5, 7, 7], [], [2], [2, 8]] type='4 * var * int64'>
    """
    with ak._errors.OperationErrorContext(
        "ak.argsort",
        dict(
            array=array,
            axis=axis,
            ascending=ascending,
            stable=stable,
            highlevel=highlevel,
            behavior=behavior,
        ),
    ):
        return _impl(array, axis, ascending, stable, highlevel, behavior)


def _impl(array, axis, ascending, stable, highlevel, behavior):
    layout = ak.operations.to_layout(array, allow_record=False, allow_other=False)
    out = ak._do.argsort(layout, axis, ascending, stable)
    return ak._util.wrap(out, behavior, highlevel, like=array)
