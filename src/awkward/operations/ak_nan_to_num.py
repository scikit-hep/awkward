# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from functools import reduce

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext, ensure_same_backend
from awkward._namedaxis import NAMED_AXIS_KEY, NamedAxesWithDims, _unify_named_axis
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._typing import Mapping

__all__ = ("nan_to_num",)

np = NumpyMetadata.instance()


@high_level_function()
def nan_to_num(
    array,
    copy: bool = True,
    nan=0.0,
    posinf=None,
    neginf=None,
    *,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping | None = None,
):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        copy (bool): Ignored (Awkward Arrays are immutable).
        nan (int, float, broadcastable array): Value to be used to fill `NaN` values.
        posinf (None, int, float, broadcastable array): Value to be used to fill positive infinity
            values. If None, positive infinities are replaced with a very large number.
        neginf (None, int, float, broadcastable array): Value to be used to fill negative infinity
            values. If None, negative infinities are replaced with a very small number.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Implements [np.nan_to_num](https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html)
    for Awkward Arrays, which replaces NaN ("not a number") or infinity with specified values.

    See also #ak.nan_to_none to convert NaN to None, i.e. missing values with option-type.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, copy, nan, posinf, neginf, highlevel, behavior, attrs)


def _impl(
    array,
    copy: bool,
    nan,
    posinf,
    neginf,
    highlevel: bool,
    behavior: Mapping | None,
    attrs: Mapping | None,
):
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout, nan_layout, posinf_layout, neginf_layout = ensure_same_backend(
            ctx.unwrap(array),
            ctx.unwrap(
                nan,
                allow_unknown=False,
                none_policy="pass-through",
                primitive_policy="pass-through",
                allow_record=False,
            ),
            ctx.unwrap(
                posinf,
                allow_unknown=False,
                none_policy="pass-through",
                primitive_policy="pass-through",
                allow_record=False,
            ),
            ctx.unwrap(
                neginf,
                allow_unknown=False,
                none_policy="pass-through",
                primitive_policy="pass-through",
                allow_record=False,
            ),
        )

    broadcasting_ids = {}
    broadcasting = [layout]
    arrays_to_broadcast = [array]
    if isinstance(nan_layout, ak.contents.Content):
        broadcasting_ids[id(nan)] = len(broadcasting)
        broadcasting.append(nan_layout)
        arrays_to_broadcast.append(nan)
    if isinstance(posinf_layout, ak.contents.Content):
        broadcasting_ids[id(posinf)] = len(broadcasting)
        broadcasting.append(posinf_layout)
        arrays_to_broadcast.append(posinf)
    if isinstance(neginf_layout, ak.contents.Content):
        broadcasting_ids[id(neginf)] = len(broadcasting)
        broadcasting.append(neginf_layout)
        arrays_to_broadcast.append(neginf)

    if len(broadcasting) == 1:

        def action(layout, backend, **kwargs):
            if isinstance(layout, ak.contents.NumpyArray):
                return ak.contents.NumpyArray(
                    backend.nplike.nan_to_num(
                        layout.data,
                        nan=nan,
                        posinf=posinf,
                        neginf=neginf,
                    )
                )
            else:
                return None

        out = ak._do.recursively_apply(layout, action)

    else:

        def action(inputs, backend, **kwargs):
            if all(isinstance(x, ak.contents.NumpyArray) for x in inputs):
                tmp_layout = inputs[0].data
                if id(nan) in broadcasting_ids:
                    tmp_nan = inputs[broadcasting_ids[id(nan)]].to_backend_array()
                else:
                    tmp_nan = nan
                if id(posinf) in broadcasting_ids:
                    tmp_posinf = inputs[broadcasting_ids[id(posinf)]].to_backend_array()
                else:
                    tmp_posinf = posinf
                if id(neginf) in broadcasting_ids:
                    tmp_neginf = inputs[broadcasting_ids[id(neginf)]].to_backend_array()
                else:
                    tmp_neginf = neginf
                return (
                    ak.contents.NumpyArray(
                        backend.nplike.nan_to_num(
                            tmp_layout,
                            nan=tmp_nan,
                            posinf=tmp_posinf,
                            neginf=tmp_neginf,
                        )
                    ),
                )
            else:
                return None

        depth_context, lateral_context = NamedAxesWithDims.prepare_contexts(
            arrays_to_broadcast
        )
        out = ak._broadcasting.broadcast_and_apply(
            broadcasting,
            action,
            depth_context=depth_context,
            lateral_context=lateral_context,
        )
        assert isinstance(out, tuple) and len(out) == 1

        # Unify named axes propagated through the broadcast
        out_named_axis = reduce(
            _unify_named_axis, lateral_context[NAMED_AXIS_KEY].named_axis
        )
        wrapped_out = ctx.wrap(out[0], highlevel=highlevel)
        return ak.operations.ak_with_named_axis._impl(
            wrapped_out,
            named_axis=out_named_axis,
            highlevel=highlevel,
            behavior=ctx.behavior,
            attrs=ctx.attrs,
        )

    return ctx.wrap(out, highlevel=highlevel)


@ak._connect.numpy.implements("nan_to_num")
def _nep_18_impl(x, copy=True, nan=0.0, posinf=None, neginf=None):
    return nan_to_num(x, copy=copy, nan=nan, posinf=posinf, neginf=neginf)
