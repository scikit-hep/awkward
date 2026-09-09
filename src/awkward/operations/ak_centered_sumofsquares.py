# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


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

# Internal-only: no public op. Backs the two-pass variance numerator of
# ak.var/ak.std for a grouped (non-None) axis, computing sum((x - mean)**2) per
# segment in one pass -- no `x - mean` deviation buffer and no mean back-broadcast.
# `means` is a 1-D float64 array with one value per output bin, in bin order
# (i.e. ak.ravel(ak.mean(array, axis, keepdims=True))); the same _do.reduce(axis)
# descent produces the bins here, so the ordering matches. Mirrors ak_sumofsquares
# for named-axis / masking / wrapping.
__all__ = ()

np = NumpyMetadata.instance()


def _impl(array, means, axis, keepdims, mask_identity, highlevel, behavior, attrs):
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False, primitive_policy="error")
        means_layout = ctx.unwrap(means, allow_record=False, primitive_policy="error")

    # One float64 mean per output bin, raw (aligned to the reducer's outlength).
    means_data = means_layout.data

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

    reducer = ak._reducers.CenteredSumOfSquares(means_data)

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
