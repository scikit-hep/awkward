# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext, ensure_same_backend
from awkward._nplikes.numpy_like import NumpyMetadata

__all__ = ("mask",)

np = NumpyMetadata.instance()


@high_level_function()
def mask(array, mask, *, valid_when=True, highlevel=True, behavior=None, attrs=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        mask (array of booleans): The mask that overlays elements in the
            `array` with None. Must have the same length as `array`.
        valid_when (bool): If True, True values in `mask` are considered
            valid (passed from `array` to the output); if False, False
            values in `mask` are considered valid.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Returns an array for which

        output[i] = array[i] if mask[i] == valid_when else None

    Unlike filtering data with #ak.Array.__getitem__, this `output` has the
    same length as the original `array` and can therefore be used in
    calculations with it, such as
    [universal functions](https://docs.scipy.org/doc/numpy/reference/ufuncs.html).

    For example, with

        >>> array = ak.Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    with a boolean selection of `good` elements like

        >>> good = (array % 2 == 1)
        >>> good
        <Array [False, True, False, True, ..., True, False, True] type='10 * bool'>

    could be used to filter the original `array` (or another with the same
    length).

        >>> array[good]
        <Array [1, 3, 5, 7, 9] type='5 * int64'>

    However, this eliminates information about which elements were dropped and
    where they were. If we instead use #ak.mask,

        >>> ak.mask(array, good)
        <Array [None, 1, None, 3, None, 5, None, 7, None, 9] type='10 * ?int64'>

    this information and the length of the array is preserved, and it can be
    used in further calculations with the original `array` (or another with
    the same length).

        >>> ak.mask(array, good) + array
        <Array [None, 2, None, 6, None, 10, None, 14, None, 18] type='10 * ?int64'>

    In particular, successive filters can be applied to the same array.

    Even if the `array` and/or the `mask` is nested,

        >>> array = ak.Array([[[0, 1, 2], [], [3, 4], [5]], [[6, 7, 8], [9]]])
        >>> good = (array % 2 == 1)
        >>> good
        <Array [[[False, True, False], ..., [True]], ...] type='2 * var * var * bool'>

    it can still be used with #ak.mask because the `array` and `mask`
    parameters are broadcasted.

        >>> ak.mask(array, good)
        <Array [[[None, 1, None], [], ..., [5]], ...] type='2 * var * var * ?int64'>

    See #ak.broadcast_arrays for details about broadcasting and the generalized
    set of broadcasting rules.

    Another syntax for

        ak.mask(array, array_of_booleans)

    is

        array.mask[array_of_booleans]

    (which is 5 characters away from simply filtering the `array`).
    """
    # Dispatch
    yield array, mask

    # Implementation
    return _impl(array, mask, valid_when, highlevel, behavior, attrs)


def _impl(array, mask, valid_when, highlevel, behavior, attrs):
    def action(inputs, backend, **kwargs):
        layoutarray, layoutmask = inputs
        if layoutmask.is_numpy:
            m = backend.nplike.asarray(layoutmask.data)
            if not issubclass(m.dtype.type, (bool, np.bool_)):
                raise ValueError(f"mask must have boolean type, not {m.dtype!r}")
            bytemask = ak.index.Index8(m.view(np.int8))
            return (
                ak.contents.ByteMaskedArray.simplified(
                    bytemask, layoutarray, valid_when=valid_when
                ),
            )
        else:
            return None

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layouts = ensure_same_backend(
            ctx.unwrap(array, allow_record=False, primitive_policy="error"),
            ctx.unwrap(mask, allow_record=False, primitive_policy="error"),
        )

    out = ak._broadcasting.broadcast_and_apply(
        layouts, action, numpy_to_regular=True, right_broadcast=False
    )
    assert isinstance(out, tuple) and len(out) == 1
    return ctx.wrap(out[0], highlevel=highlevel)
