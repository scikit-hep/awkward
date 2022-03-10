# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


# @ak._v2._connect.numpy.implements("cumsum")
def cumsum(array, axis=None, highlevel=True, behavior=None):
    """
    Args:
        array: Data to accumulate, possibly within nested lists.
        axis (int): The dimension at which this operation is applied. The
            outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Returns the cumulative sum of an array along a given axis.
    """
    with ak._v2._util.OperationErrorContext(
        "ak._v2.cumsum",
        dict(array=array, axis=axis, highlevel=highlevel, behavior=behavior),
    ):
        return _impl(array, axis, highlevel, behavior)


def _impl(array, axis, highlevel, behavior):
    layout = ak._v2.operations.convert.to_layout(
        array, allow_record=False, allow_other=False
    )
    nplike = ak.nplike.of(layout)

    if axis is None:
        flattened = layout.completely_flatten(function_name="ak.cumsum")
        assert isinstance(flattened, tuple) and all(
            isinstance(x, nplike.ndarray) for x in flattened
        )
        flattened = ak._v2.contents.NumpyArray(nplike.concatenate(flattened))
        out = flattened.cumsum()

    else:
        out = layout.cumsum(axis)

    return ak._v2._util.wrap(out, behavior, highlevel)
