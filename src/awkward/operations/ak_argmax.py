# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
__all__ = ("argmax",)
import awkward as ak
from awkward._behavior import behavior_of
from awkward._connect.numpy import unsupported
from awkward._layout import wrap_layout
from awkward._nplikes.numpylike import NumpyMetadata
from awkward._regularize import regularize_axis
from awkward._util import unset

np = NumpyMetadata.instance()


def argmax(
    array,
    axis=None,
    *,
    keepdims=False,
    mask_identity=True,
    flatten_records=unset,
    highlevel=True,
    behavior=None,
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
            results in the operation's identity.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Returns the index position of the maximum value in each group of elements
    from `array` (many types supported, including all Awkward Arrays and
    Records). The identity of maximization would be negative infinity, but
    argmax must return the position of the maximum element, which has no value
    for empty lists. Therefore, the identity should be masked: the argmax of
    an empty list is None. If `mask_identity=False`, the result would be `-1`,
    which is distinct from all valid index positions, but care should be taken
    that it is not misinterpreted as "the last element of the list."

    This operation is the same as NumPy's
    [argmax](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html)
    if all lists at a given dimension have the same length and no None values,
    but it generalizes to cases where they do not.

    See #ak.sum for a more complete description of nested list and missing
    value (None) handling in reducers.

    See also #ak.nanargmax.
    """
    with ak._errors.OperationErrorContext(
        "ak.argmax",
        {
            "array": array,
            "axis": axis,
            "keepdims": keepdims,
            "mask_identity": mask_identity,
            "highlevel": highlevel,
            "behavior": behavior,
        },
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
        return _impl(array, axis, keepdims, mask_identity, highlevel, behavior)


def nanargmax(
    array,
    axis=None,
    *,
    keepdims=False,
    mask_identity=True,
    flatten_records=unset,
    highlevel=True,
    behavior=None,
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
            results in the operation's identity.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Like #ak.argmax, but treating NaN ("not a number") values as missing.

    Equivalent to

        ak.argmax(ak.nan_to_none(array))

    with all other arguments unchanged.

    See also #ak.argmax.
    """
    with ak._errors.OperationErrorContext(
        "ak.nanargmax",
        {
            "array": array,
            "axis": axis,
            "keepdims": keepdims,
            "mask_identity": mask_identity,
            "highlevel": highlevel,
            "behavior": behavior,
        },
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

        return _impl(array, axis, keepdims, mask_identity, highlevel, behavior)


def _impl(array, axis, keepdims, mask_identity, highlevel, behavior):
    axis = regularize_axis(axis)
    layout = ak.operations.to_layout(array, allow_record=False, allow_other=False)
    behavior = behavior_of(array, behavior=behavior)
    reducer = ak._reducers.ArgMax()

    out = ak._do.reduce(
        layout,
        reducer,
        axis=axis,
        mask=mask_identity,
        keepdims=keepdims,
        behavior=behavior,
    )
    if isinstance(out, (ak.contents.Content, ak.record.Record)):
        return wrap_layout(out, behavior, highlevel)
    else:
        return out


@ak._connect.numpy.implements("argmax")
def _nep_18_impl_argmax(a, axis=None, out=unsupported, *, keepdims=False):
    return argmax(a, axis=axis, keepdims=keepdims)


@ak._connect.numpy.implements("nanargmax")
def _nep_18_impl_nanargmax(a, axis=None, out=unsupported, *, keepdims=False):
    return nanargmax(a, axis=axis, keepdims=keepdims)