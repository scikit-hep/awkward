# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


# @ak._v2._connect.numpy.implements("ptp")
def ptp(array, axis=None, keepdims=False, mask_identity=True, flatten_records=False):
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
        flatten_records (bool): If True, axis=None combines fields from different
            records; otherwise, records raise an error.

    Returns the range of values in each group of elements from `array` (many
    types supported, including all Awkward Arrays and Records). The range of
    an empty list is None, unless `mask_identity=False`, in which case it is 0.
    This operation is the same as NumPy's
    [ptp](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ptp.html)
    if all lists at a given dimension have the same length and no None values,
    but it generalizes to cases where they do not.

    For example, with an `array` like

        ak.Array([[0, 1, 2, 3],
                  [          ],
                  [4, 5      ]])

    The range of the innermost lists is

        >>> ak.ptp(array, axis=-1)
        <Array [3, None, 1] type='3 * ?float64'>

    because there are three lists, the first has a range of `3`, the second is
    `None` because the list is empty, and the third has a range of `1`. Similarly,

        >>> ak.ptp(array, axis=-1, mask_identity=False)
        <Array [3, 0, 1] type='3 * float64'>

    The second value is `0` because the list is empty.

    See #ak.sum for a more complete description of nested list and missing
    value (None) handling in reducers.
    """
    with ak._v2._util.OperationErrorContext(
        "ak._v2.ptp",
        dict(
            array=array,
            axis=axis,
            keepdims=keepdims,
            mask_identity=mask_identity,
            flatten_records=flatten_records,
        ),
    ):
        return _impl(array, axis, keepdims, mask_identity, flatten_records)


def _impl(array, axis, keepdims, mask_identity, flatten_records):
    array = ak._v2.highlevel.Array(
        ak._v2.operations.to_layout(array, allow_record=False, allow_other=False)
    )

    if axis is None:
        out = ak._v2.operations.ak_max._impl(
            array, axis, keepdims, None, mask_identity, flatten_records
        ) - ak._v2.operations.ak_min._impl(
            array, axis, keepdims, None, mask_identity, flatten_records
        )
        if not mask_identity and out is None:
            out = 0

    else:
        maxi = ak._v2.operations.ak_max._impl(
            array, axis, True, None, mask_identity, flatten_records
        )
        mini = ak._v2.operations.ak_min._impl(
            array, axis, True, None, True, flatten_records
        )

        if maxi is None or mini is None:
            out = None

        else:
            out = maxi - mini

            if not mask_identity:
                out = ak._v2.highlevel.Array(
                    ak._v2.operations.fill_none(out, 0, axis=-1)
                )

            if not keepdims:
                posaxis = out.layout.axis_wrap_if_negative(axis)
                out = out[(slice(None, None),) * posaxis + (0,)]

    return out
