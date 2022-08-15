# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


# @ak._v2._connect.numpy.implements("nan_to_num")
def nan_to_num(
    array, copy=True, nan=0.0, posinf=None, neginf=None, highlevel=True, behavior=None
):

    """
    Args:
        array: Array whose `NaN` values should be converted to a number.
        copy (bool): Ignored (Awkward Arrays are immutable).
        nan (int, float, broadcastable array): Value to be used to fill `NaN` values.
        posinf (None, int, float, broadcastable array): Value to be used to fill positive infinity
            values. If None, positive infinities are replaced with a very large number.
        neginf (None, int, float, broadcastable array): Value to be used to fill negative infinity
            values. If None, negative infinities are replaced with a very small number.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Implements [np.nan_to_num](https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html)
    for Awkward Arrays, which replaces NaN ("not a number") or infinity with specified values.

    See also #ak.nan_to_none to convert NaN to None, i.e. missing values with option-type.
    """
    with ak._v2._util.OperationErrorContext(
        "ak._v2.nan_to_num",
        dict(
            array=array,
            copy=copy,
            nan=nan,
            posinf=posinf,
            neginf=neginf,
            highlevel=highlevel,
            behavior=behavior,
        ),
    ):
        return _impl(array, copy, nan, posinf, neginf, highlevel, behavior)


def _impl(array, copy, nan, posinf, neginf, highlevel, behavior):
    behavior = ak._v2._util.behavior_of(array, behavior=behavior)

    broadcasting_ids = {}
    broadcasting = []

    layout = ak._v2.operations.to_layout(array)
    broadcasting.append(layout)

    nan_layout = ak._v2.operations.to_layout(nan, allow_other=True)
    if isinstance(nan_layout, ak._v2.contents.Content):
        broadcasting_ids[id(nan)] = len(broadcasting)
        broadcasting.append(nan_layout)

    posinf_layout = ak._v2.operations.to_layout(posinf, allow_other=True)
    if isinstance(posinf_layout, ak._v2.contents.Content):
        broadcasting_ids[id(posinf)] = len(broadcasting)
        broadcasting.append(posinf_layout)

    neginf_layout = ak._v2.operations.to_layout(neginf, allow_other=True)
    if isinstance(neginf_layout, ak._v2.contents.Content):
        broadcasting_ids[id(neginf)] = len(broadcasting)
        broadcasting.append(neginf_layout)

    nplike = ak.nplike.of(layout)

    if len(broadcasting) == 1:

        def action(layout, **kwargs):
            if isinstance(layout, ak._v2.contents.NumpyArray):
                return ak._v2.contents.NumpyArray(
                    nplike.nan_to_num(
                        nplike.asarray(layout),
                        nan=nan,
                        posinf=posinf,
                        neginf=neginf,
                    )
                )
            else:
                return None

        out = layout.recursively_apply(action, behavior)

    else:

        def action(inputs, **kwargs):
            if all(isinstance(x, ak._v2.contents.NumpyArray) for x in inputs):
                tmp_layout = nplike.asarray(inputs[0])
                if id(nan) in broadcasting_ids:
                    tmp_nan = nplike.asarray(inputs[broadcasting_ids[id(nan)]])
                else:
                    tmp_nan = nan
                if id(posinf) in broadcasting_ids:
                    tmp_posinf = nplike.asarray(inputs[broadcasting_ids[id(posinf)]])
                else:
                    tmp_posinf = posinf
                if id(neginf) in broadcasting_ids:
                    tmp_neginf = nplike.asarray(inputs[broadcasting_ids[id(neginf)]])
                else:
                    tmp_neginf = neginf
                return (
                    ak._v2.contents.NumpyArray(
                        nplike.nan_to_num(
                            tmp_layout,
                            nan=tmp_nan,
                            posinf=tmp_posinf,
                            neginf=tmp_neginf,
                        )
                    ),
                )
            else:
                return None

        out = ak._v2._broadcasting.broadcast_and_apply(broadcasting, action, behavior)
        assert isinstance(out, tuple) and len(out) == 1
        out = out[0]

    return ak._v2._util.wrap(out, behavior, highlevel)
