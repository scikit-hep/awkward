# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


# @ak._v2._connect.numpy.implements("argmin")
def argmin(array, axis=None, keepdims=False, mask_identity=True):
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

    Returns the index position of the minimum value in each group of elements
    from `array` (many types supported, including all Awkward Arrays and
    Records). The identity of minimization would be infinity, but argmin
    must return the position of the minimum element, which has no value for
    empty lists. Therefore, the identity should be masked: the argmin of
    an empty list is None. If `mask_identity=False`, the result would be `-1`,
    which is distinct from all valid index positions, but care should be taken
    that it is not misinterpreted as "the last element of the list."

    This operation is the same as NumPy's
    [argmin](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmin.html)
    if all lists at a given dimension have the same length and no None values,
    but it generalizes to cases where they do not.

    See #ak.sum for a more complete description of nested list and missing
    value (None) handling in reducers.
    """
    layout = ak._v2.operations.convert.to_layout(
        array, allow_record=False, allow_other=False
    )

    if axis is None:
        if isinstance(layout, ak.partition.PartitionedArray):  # NO PARTITIONED ARRAY
            start = 0
            best_index = None
            best_value = None
            for partition in layout.partitions:
                for tmp in ak._v2._util.completely_flatten(partition):
                    out = ak.nplike.of(tmp).argmin(tmp, axis=None)
                    if best_index is None or tmp[out] < best_value:
                        best_index = start + out
                        best_value = tmp[out]
                start += len(partition)
            return best_index

        else:
            best_index = None
            best_value = None
            for tmp in ak._v2._util.completely_flatten(layout):
                out = ak.nplike.of(tmp).argmin(tmp, axis=None)
                if best_index is None or tmp[out] < best_value:
                    best_index = out
                    best_value = tmp[out]
            return best_index

    else:
        behavior = ak._v2._util.behavior_of(array)
        return ak._v2._util.wrap(
            layout.argmin(axis=axis, mask=mask_identity, keepdims=keepdims), behavior
        )
