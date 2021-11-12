# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


# @ak._v2._connect.numpy.implements("any")
def any(array, axis=None, keepdims=False, mask_identity=False, flatten_records=False):
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
        flatten_records (bool): If True, axis=None combines fields from different
            records; otherwise, records raise an error.

    Returns True in each group of elements from `array` (many types supported,
    including all Awkward Arrays and Records) if any values are True; False
    otherwise. Thus, it represents reduction over the "logical or" operation,
    whose identity is False (i.e. asking if there are any True values in an
    empty list results in False). This operation is the same as NumPy's
    [any](https://docs.scipy.org/doc/numpy/reference/generated/numpy.any.html)
    if all lists at a given dimension have the same length and no None values,
    but it generalizes to cases where they do not.

    See #ak.sum for a more complete description of nested list and missing
    value (None) handling in reducers.
    """
    layout = ak._v2.operations.convert.to_layout(
        array, allow_record=False, allow_other=False
    )

    if axis is None:
        return layout.nplike.any(
            layout.completely_flatten(
                function_name="ak.any", flatten_records=flatten_records
            )
        )

    else:
        behavior = ak._v2._util.behavior_of(array)
        return ak._v2._util.wrap(
            layout.any(axis=axis, mask=mask_identity, keepdims=keepdims), behavior
        )
