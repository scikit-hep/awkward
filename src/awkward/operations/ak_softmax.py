# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import (
    HighLevelContext,
    maybe_highlevel_to_lowlevel,
    maybe_posaxis,
)
from awkward._nplikes import ufuncs
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._regularize import regularize_axis

__all__ = ("softmax",)

np = NumpyMetadata.instance()


@high_level_function()
def softmax(
    x,
    axis=None,
    *,
    keepdims=False,
    mask_identity=False,
    highlevel=True,
    behavior=None,
    attrs=None,
):
    """
    Args:
        x: The data on which to compute the softmax (anything #ak.to_layout recognizes).
        axis (None or int): If None, combine all values from the array into
            a single scalar result; if an int, group by that axis. Only `axis`
            arguments equivalent to `-1` (softmax reduction along the innermost
            dimension) is supported.
        keepdims (bool): If False, this function decreases the number of
            dimensions by 1; if True, the output values are wrapped in a new
            length-1 dimension so that the result of this operation may be
            broadcasted with the original array.
        mask_identity (bool): If True, the application of this function on
            empty lists results in None (an option type); otherwise, the
            calculation is followed through with the reducers' identities,
            usually resulting in floating-point `nan`.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Computes the softmax in each group of elements from `x` (many
    types supported, including all Awkward Arrays and Records). The grouping
    is performed the same way as for reducers, though this operation is not a
    reducer and has no identity.

    This function has no NumPy equivalent.

    Passing all arguments to the reducers, the softmax is calculated as

        np.exp(x) / ak.sum(np.exp(x))

    See #ak.sum for a complete description of handling nested lists and
    missing values (None) in reducers, and #ak.mean for an example with another
    non-reducer.
    """
    # Dispatch
    yield (x,)

    # Implementation
    return _impl(x, axis, keepdims, mask_identity, highlevel, behavior, attrs)


def _impl(x, axis, keepdims, mask_identity, highlevel, behavior, attrs):
    original_axis = axis
    axis = regularize_axis(axis)

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        x_layout = ctx.unwrap(x, allow_record=False, primitive_policy="error")
    x = ctx.wrap(x_layout)

    if maybe_posaxis(x_layout, axis, 1) != maybe_posaxis(x_layout, -1, 1):
        raise NotImplementedError(
            f"ak.softmax is only defined for axis=-1, but axis={original_axis}; see https://github.com/scikit-hep/awkward/issues/2760#issuecomment-2034749982"
        )

    with np.errstate(invalid="ignore", divide="ignore"):
        expx = ufuncs.exp(x)
        denom = ak.operations.ak_sum._impl(
            expx,
            axis,
            keepdims,
            mask_identity,
            highlevel=True,
            behavior=ctx.behavior,
            attrs=ctx.attrs,
        )
        return ctx.wrap(
            maybe_highlevel_to_lowlevel(expx / denom),
            highlevel=highlevel,
            allow_other=True,
        )
