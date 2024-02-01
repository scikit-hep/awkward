# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._connect.numpy import UNSUPPORTED
from awkward._dispatch import high_level_function
from awkward._layout import (
    HighLevelContext,
    maybe_highlevel_to_lowlevel,
    maybe_posaxis,
)
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._regularize import regularize_axis

__all__ = ("ptp",)

np = NumpyMetadata.instance()


@high_level_function()
def ptp(
    array,
    axis=None,
    *,
    keepdims=False,
    mask_identity=True,
    highlevel=True,
    behavior=None,
    attrs=None,
):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        axis (None or int): If None, combine all values from the array into
            a single scalar result; if an int, group by that axis: `0` is the
            outermost, `1` is the first level of nested lists, etc., and
            negative `axis` counts from the innermost: `-1` is the innermost,
            `-2` is the next level up, etc.
        keepdims (bool): If False, this reducer decreases the number of
            dimensions by 1; if True, the reduced values are wrapped in a new
            length-1 dimension so that the result of this operation may be
            broadcasted with the original array.
        mask_identity (bool): If True, reducing over empty lists results in
            None (an option type); otherwise, reducing over empty lists
            results in the operation's identity of 0.

    Returns the range of values in each group of elements from `array` (many
    types supported, including all Awkward Arrays and Records). The range of
    an empty list is None, unless `mask_identity=False`, in which case it is 0.
    This operation is the same as NumPy's
    [ptp](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ptp.html)
    if all lists at a given dimension have the same length and no None values,
    but it generalizes to cases where they do not.

    For example, with

        >>> array = ak.Array([[0, 1, 2, 3],
        ...                   [          ],
        ...                   [4, 5      ]])

    The range of the innermost lists is

        >>> ak.ptp(array, axis=-1)
        <Array [3, None, 1] type='3 * ?int64'>

    because there are three lists, the first has a range of `3`, the second is
    `None` because the list is empty, and the third has a range of `1`. Similarly,

        >>> ak.ptp(array, axis=-1, mask_identity=False)
        <Array [3, 0, 1] type='3 * float64'>

    The second value is `0` because the list is empty.

    See #ak.sum for a more complete description of nested list and missing
    value (None) handling in reducers.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, axis, keepdims, mask_identity, highlevel, behavior, attrs)


def _impl(array, axis, keepdims, mask_identity, highlevel, behavior, attrs):
    axis = regularize_axis(axis)
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False, primitive_policy="error")

    with np.errstate(invalid="ignore", divide="ignore"):
        maxi = ak.operations.ak_max._impl(
            layout,
            axis,
            True,
            None,
            mask_identity,
            highlevel=True,
            behavior=ctx.behavior,
            attrs=ctx.attrs,
        )
        mini = ak.operations.ak_min._impl(
            layout,
            axis,
            True,
            None,
            True,
            highlevel=True,
            behavior=ctx.behavior,
            attrs=ctx.attrs,
        )
        out = maxi - mini

        # Check that removed code was not needed!
        assert maxi is not None and mini is not None

        if not mask_identity:
            out = ak.operations.fill_none(
                out, 0, axis=-1, behavior=ctx.behavior, attrs=ctx.attrs, highlevel=True
            )

        if axis is None:
            if not keepdims:
                out = out[(0,) * out.ndim]
        else:
            if not keepdims:
                posaxis = maybe_posaxis(out.layout, axis, 1)
                out = out[(slice(None, None),) * posaxis + (0,)]

        return ctx.wrap(
            maybe_highlevel_to_lowlevel(out), highlevel=highlevel, allow_other=True
        )


@ak._connect.numpy.implements("ptp")
def _nep_18_impl(a, axis=None, out=UNSUPPORTED, keepdims=False):
    return ptp(a, axis=axis, keepdims=keepdims)
