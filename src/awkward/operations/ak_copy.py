# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import copy as _copy

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()


@ak._connect.numpy.implements("copy")
def copy(array):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).

    Returns a deep copy of the array (no memory shared with original).

    This is identical to `np.copy` and `copy.deepcopy`.

    It's only useful to explicitly copy an array if you're going to change it
    in-place. This doesn't come up often because Awkward Arrays are immutable.
    That is to say, the Awkward Array library doesn't have any operations that
    change an array in-place, but the data in the array might be owned by another
    library that can change it in-place.

    For example, if the array comes from NumPy:

        >>> underlying_array = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
        >>> wrapper = ak.Array(underlying_array)
        >>> duplicate = ak.copy(wrapper)
        >>> underlying_array[2] = 123
        >>> underlying_array
        array([  1.1,   2.2, 123. ,   4.4,   5.5])
        >>> wrapper
        <Array [1.1, 2.2, 123, 4.4, 5.5] type='5 * float64'>
        >>> duplicate
        <Array [1.1, 2.2, 3.3, 4.4, 5.5] type='5 * float64'>

    There is an exception to this rule: you can add fields to records in an
    #ak.Array in-place. However, this changes the #ak.Array wrapper without
    affecting the underlying layout data (it *replaces* its layout), so a
    shallow copy will do:

        >>> import copy
        >>> original = ak.Array([{"x": 1}, {"x": 2}, {"x": 3}])
        >>> shallow_copy = copy.copy(original)
        >>> shallow_copy["y"] = original.x**2
        >>> shallow_copy
        <Array [{x: 1, y: 1}, {...}, {x: 3, y: 9}] type='3 * {x: int64, y: int64}'>
        >>> original
        <Array [{x: 1}, {x: 2}, {x: 3}] type='3 * {x: int64}'>

    This is key to Awkward Array's efficiency (memory and speed): operations that
    only change part of a structure re-use pieces from the original ("structural
    sharing"). Changing data in-place would result in many surprising long-distance
    changes, so we don't support it. However, an #ak.Array's data might come from
    a mutable third-party library, so this function allows you to make a true copy.
    """
    with ak._errors.OperationErrorContext(
        "ak.fill_none",
        dict(array=array),
    ):
        return _impl(array)


def _impl(array):
    return _copy.deepcopy(array)
