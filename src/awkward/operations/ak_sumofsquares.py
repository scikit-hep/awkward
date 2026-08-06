# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._layout import HighLevelContext
from awkward._namedaxis import (
    _get_named_axis,
    _keep_named_axis,
    _named_axis_to_positional_axis,
    _remove_named_axis,
)
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._regularize import regularize_axis

# Internal-only: no public `ak.sumofsquares`. This backs the squared term of
# ak.var/ak.std/ak.moment, accumulating `sum(x**2)` in float64 directly from the
# input (no `x*x` buffer, no integer overflow). Mirrors ak_sum._impl so the
# named-axis / masking / wrapping behaviour matches the reduction it replaces.
__all__ = ()

np = NumpyMetadata.instance()


def _impl(array, axis, keepdims, mask_identity, highlevel, behavior, attrs):
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False, primitive_policy="error")

    named_axis = _get_named_axis(ctx)
    axis = _named_axis_to_positional_axis(named_axis, axis)
    out_named_axis = _keep_named_axis(named_axis, None)
    if not keepdims:
        out_named_axis = _remove_named_axis(
            named_axis=out_named_axis,
            axis=axis,
            total=layout.minmax_depth[1],
        )

    axis = regularize_axis(axis, none_allowed=True)

    reducer = ak._reducers.SumOfSquares()

    out = ak._do.reduce(
        layout,
        reducer,
        axis=axis,
        mask=mask_identity,
        keepdims=keepdims,
        behavior=ctx.behavior,
    )

    wrapped_out = ctx.wrap(
        out,
        highlevel=highlevel,
        allow_other=True,
    )

    return ak.operations.ak_with_named_axis._impl(
        wrapped_out,
        named_axis=out_named_axis,
        highlevel=highlevel,
        behavior=ctx.behavior,
        attrs=ctx.attrs,
    )
