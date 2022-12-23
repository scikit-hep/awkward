# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak
from awkward._util import unset

np = ak._nplikes.NumpyMetadata.instance()


@ak._connect.numpy.implements("max")
def max(
    array,
    axis=None,
    *,
    keepdims=False,
    initial=None,
    mask_identity=True,
    flatten_records=unset,
    highlevel=True,
    behavior=None
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
        initial (None or number): The minimum value of an output element, as
            an alternative to the numeric type's natural identity (e.g. negative
            infinity for floating-point types, a minimum integer for integer types).
            If you use `initial`, you might also want `mask_identity=False`.
        mask_identity (bool): If True, reducing over empty lists results in
            None (an option type); otherwise, reducing over empty lists
            results in the operation's identity.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Returns the maximum value in each group of elements from `array` (many
    types supported, including all Awkward Arrays and Records). The identity
    of maximization is `-inf` if floating-point or the smallest integer value
    if applied to integers. This identity is usually masked: the maximum of
    an empty list is None, unless `mask_identity=False`.
    This operation is the same as NumPy's
    [amax](https://docs.scipy.org/doc/numpy/reference/generated/numpy.amax.html)
    if all lists at a given dimension have the same length and no None values,
    but it generalizes to cases where they do not.

    See #ak.sum for a more complete description of nested list and missing
    value (None) handling in reducers.

    See also #ak.nanmax.
    """
    with ak._errors.OperationErrorContext(
        "ak.max",
        dict(
            array=array,
            axis=axis,
            keepdims=keepdims,
            initial=initial,
            mask_identity=mask_identity,
            highlevel=highlevel,
            behavior=behavior,
        ),
    ):
        if flatten_records is not unset:
            message = (
                "`flatten_records` is no longer a supported argument for reducers. "
                "Instead, use `ak.ravel(array)` first to remove the record structure "
                "and flatten the array."
            )
            if flatten_records:
                raise ak._errors.wrap_error(ValueError(message))
            else:
                ak._errors.deprecate(message, "2.2.0")
        return _impl(
            array,
            axis,
            keepdims,
            initial,
            mask_identity,
            highlevel,
            behavior,
        )


@ak._connect.numpy.implements("nanmax")
def nanmax(
    array,
    axis=None,
    *,
    keepdims=False,
    initial=None,
    mask_identity=True,
    flatten_records=unset,
    highlevel=True,
    behavior=None
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
        initial (None or number): The minimum value of an output element, as
            an alternative to the numeric type's natural identity (e.g. negative
            infinity for floating-point types, a minimum integer for integer types).
            If you use `initial`, you might also want `mask_identity=False`.
        mask_identity (bool): If True, reducing over empty lists results in
            None (an option type); otherwise, reducing over empty lists
            results in the operation's identity.

    Like #ak.max, but treating NaN ("not a number") values as missing.

    Equivalent to

        ak.max(ak.nan_to_none(array))

    with all other arguments unchanged.

    See also #ak.max.
    """
    with ak._errors.OperationErrorContext(
        "ak.nanmax",
        dict(
            array=array,
            axis=axis,
            keepdims=keepdims,
            initial=initial,
            mask_identity=mask_identity,
            highlevel=highlevel,
            behavior=behavior,
        ),
    ):
        if flatten_records is not unset:
            message = (
                "`flatten_records` is no longer a supported argument for reducers. "
                "Instead, use `ak.ravel(array)` first to remove the record structure "
                "and flatten the array."
            )
            if flatten_records:
                raise ak._errors.wrap_error(ValueError(message))
            else:
                ak._errors.deprecate(message, "2.2.0")
        array = ak.operations.ak_nan_to_none._impl(array, False, None)

        return _impl(
            array,
            axis,
            keepdims,
            initial,
            mask_identity,
            highlevel,
            behavior,
        )


def _impl(array, axis, keepdims, initial, mask_identity, highlevel, behavior):
    layout = ak.operations.to_layout(array, allow_record=False, allow_other=False)
    behavior = ak._util.behavior_of(array, behavior=behavior)
    reducer = ak._reducers.Max(initial)

    out = ak._do.reduce(
        layout,
        reducer,
        axis=axis,
        mask=mask_identity,
        keepdims=keepdims,
        behavior=behavior,
    )
    if isinstance(out, (ak.contents.Content, ak.record.Record)):
        return ak._util.wrap(out, behavior, highlevel)
    else:
        return out
