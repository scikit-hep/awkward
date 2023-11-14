# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._connect.numpy import UNSUPPORTED
from awkward._dispatch import high_level_function
from awkward._nplikes.numpy_like import NumpyMetadata

__all__ = ("zeros_like",)

np = NumpyMetadata.instance()

_ZEROS = object()


@high_level_function()
def zeros_like(
    array,
    *,
    dtype=None,
    including_unknown=False,
    highlevel=True,
    behavior=None,
    attrs=None,
):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        dtype (None or NumPy dtype): Overrides the data type of the result.
        including_unknown (bool): If True, the `unknown` type is considered
            a value type and is converted to a zero-length array of the
            specified dtype; if False, `unknown` will remain `unknown`.
        highlevel (bool, default is True): If True, return an #ak.Array;
            otherwise, return a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    This is the equivalent of NumPy's `np.zeros_like` for Awkward Arrays.

    See #ak.full_like for details, and see also #ak.ones_like.

    (There is no equivalent of NumPy's `np.empty_like` because Awkward Arrays
    are immutable.)
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, highlevel, behavior, dtype, including_unknown, attrs)


def _impl(array, highlevel, behavior, dtype, including_unknown, attrs):
    if dtype is not None:
        return ak.operations.ak_full_like._impl(
            array, 0, highlevel, behavior, dtype, including_unknown, attrs
        )
    else:
        return ak.operations.ak_full_like._impl(
            array, _ZEROS, highlevel, behavior, dtype, including_unknown, attrs
        )


@ak._connect.numpy.implements("zeros_like")
def _nep_18_impl(
    a, dtype=None, order=UNSUPPORTED, subok=UNSUPPORTED, shape=UNSUPPORTED
):
    return zeros_like(a, dtype=dtype)
