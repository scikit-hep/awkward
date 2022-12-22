# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak
from awkward._util import unset

np = ak._nplikes.NumpyMetadata.instance()


@ak._connect.numpy.implements("ptp")
def ptp(array, axis=None, *, keepdims=False, mask_identity=True, flatten_records=unset):
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
    with ak._errors.OperationErrorContext(
        "ak.ptp",
        dict(
            array=array,
            axis=axis,
            keepdims=keepdims,
            mask_identity=mask_identity,
        ),
    ):
        if flatten_records is not unset:
            raise ak._errors.wrap_error(
                ValueError(
                    "`flatten_records` is no longer a supported argument for reducers. "
                    "Instead, use `ak.ravel(array)` first to remove the record structure "
                    "and flatten the array."
                )
            )
        return _impl(array, axis, keepdims, mask_identity)


def _impl(array, axis, keepdims, mask_identity):
    behavior = ak._util.behavior_of(array)
    array = ak.highlevel.Array(
        ak.operations.to_layout(array, allow_record=False, allow_other=False),
        behavior=behavior,
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        if axis is None:
            out = ak.operations.ak_max._impl(
                array,
                axis,
                keepdims,
                None,
                mask_identity,
                highlevel=True,
                behavior=None,
            ) - ak.operations.ak_min._impl(
                array,
                axis,
                keepdims,
                None,
                mask_identity,
                highlevel=True,
                behavior=None,
            )
            if not mask_identity and out is None:
                out = 0

        else:
            maxi = ak.operations.ak_max._impl(
                array,
                axis,
                True,
                None,
                mask_identity,
                highlevel=True,
                behavior=None,
            )
            mini = ak.operations.ak_min._impl(
                array,
                axis,
                True,
                None,
                True,
                highlevel=True,
                behavior=None,
            )

            if maxi is None or mini is None:
                out = None

            else:
                out = maxi - mini

                if not mask_identity:
                    out = ak.highlevel.Array(ak.operations.fill_none(out, 0, axis=-1))

                if not keepdims:
                    posaxis = ak._util.maybe_posaxis(out.layout, axis, 1)
                    out = out[(slice(None, None),) * posaxis + (0,)]

        return out
