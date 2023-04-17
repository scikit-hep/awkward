# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
__all__ = ("nan_to_num",)
import awkward as ak
from awkward._behavior import behavior_of
from awkward._layout import wrap_layout
from awkward._nplikes.numpylike import NumpyMetadata
from awkward._util import unset
from awkward.operations.ak_to_layout import _to_layout_detailed

np = NumpyMetadata.instance()


def nan_to_num(
    array,
    copy=True,
    nan=0.0,
    posinf=None,
    neginf=None,
    *,
    highlevel=True,
    behavior=None,
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

    Implements [np.nan_to_num](https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html)
    for Awkward Arrays, which replaces NaN ("not a number") or infinity with specified values.

    See also #ak.nan_to_none to convert NaN to None, i.e. missing values with option-type.
    """
    with ak._errors.OperationErrorContext(
        "ak.nan_to_num",
        {
            "array": array,
            "copy": copy,
            "nan": nan,
            "posinf": posinf,
            "neginf": neginf,
            "highlevel": highlevel,
            "behavior": behavior,
        },
    ):
        return _impl(array, copy, nan, posinf, neginf, highlevel, behavior)


def _impl(array, copy, nan, posinf, neginf, highlevel, behavior):
    behavior = behavior_of(array, behavior=behavior)
    assert array is not None
    assert nan is not None

    inputs = []
    input_is_scalar = []
    for obj in (array, nan, posinf, neginf):
        layout, layout_is_scalar = _to_layout_detailed(
            unset if obj is None else obj,
            allow_record=True,
            allow_other=True,
            regulararray=True,
        )
        inputs.append(layout)
        input_is_scalar.append(layout_is_scalar)

    def action(inputs, backend, **kwargs):
        layout, nan_layout, posinf_layout, neginf_layout = inputs
        if all(
            x.purelist_depth == 1 for x in inputs if isinstance(x, ak.contents.Content)
        ):
            return (
                ak.contents.NumpyArray(
                    backend.nplike.nan_to_num(
                        layout.data,
                        nan=nan_layout.data,
                        posinf=None if posinf_layout is unset else posinf_layout.data,
                        neginf=None if neginf_layout is unset else neginf_layout.data,
                    )
                ),
            )
        else:
            return None

    out = ak._broadcasting.broadcast_and_apply(
        inputs,
        action,
        behavior,
        return_scalar=all(input_is_scalar),
    )
    assert isinstance(out, tuple) and len(out) == 1
    out = out[0]

    return wrap_layout(out, behavior, highlevel)


@ak._connect.numpy.implements("nan_to_num")
def _nep_18_impl(x, copy=True, nan=0.0, posinf=None, neginf=None):
    return nan_to_num(x, copy=copy, nan=nan, posinf=posinf, neginf=neginf)
