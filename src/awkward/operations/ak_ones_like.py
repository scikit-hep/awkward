# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()


@ak._connect.numpy.implements("ones_like")
def ones_like(array, *, dtype=None, highlevel=True, behavior=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        dtype (None or NumPy dtype): Overrides the data type of the result.
        highlevel (bool, default is True): If True, return an #ak.Array;
            otherwise, return a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    This is the equivalent of NumPy's `np.ones_like` for Awkward Arrays.

    See #ak.full_like for details, and see also #ak.zeros_like.

    (There is no equivalent of NumPy's `np.empty_like` because Awkward Arrays
    are immutable.)
    """
    with ak._errors.OperationErrorContext(
        "ak.ones_like",
        dict(array=array, dtype=dtype, highlevel=highlevel, behavior=behavior),
    ):
        return _impl(array, highlevel, behavior, dtype)


def _impl(array, highlevel, behavior, dtype):
    return ak.operations.ak_full_like._impl(array, 1, highlevel, behavior, dtype)
