# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


# @ak._v2._connect.numpy.implements("ones_like")
def ones_like(array, highlevel=True, behavior=None, dtype=None):
    """
    Args:
        array: Array to use as a model for a replacement that contains only `1`.
        highlevel (bool, default is True): If True, return an #ak.Array;
            otherwise, return a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        dtype (None or NumPy dtype): Overrides the data type of the result.

    This is the equivalent of NumPy's `np.ones_like` for Awkward Arrays.

    See #ak.full_like for details, and see also #ak.zeros_like.

    (There is no equivalent of NumPy's `np.empty_like` because Awkward Arrays
    are immutable.)
    """
    with ak._v2._util.OperationErrorContext(
        "ak._v2.ones_like",
        dict(array=array, highlevel=highlevel, behavior=behavior, dtype=dtype),
    ):
        return _impl(array, highlevel, behavior, dtype)


def _impl(array, highlevel, behavior, dtype):
    return ak._v2.operations.ak_full_like._impl(array, 1, highlevel, behavior, dtype)
