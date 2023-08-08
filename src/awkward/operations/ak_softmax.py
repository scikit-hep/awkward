# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
__all__ = ("softmax",)
import awkward as ak
from awkward._behavior import behavior_of
from awkward._dispatch import high_level_function
from awkward._nplikes import ufuncs
from awkward._nplikes.numpylike import NumpyMetadata
from awkward._regularize import regularize_axis

np = NumpyMetadata.instance()


@high_level_function()
def softmax(x, axis=None, *, keepdims=False, mask_identity=False):
    """
    Args:
        x: The data on which to compute the softmax (anything #ak.to_layout recognizes).
        axis (None or int): If None, combine all values from the array into
            a single scalar result; if an int, group by that axis: `0` is the
            outermost, `1` is the first level of nested lists, etc., and
            negative `axis` counts from the innermost: `-1` is the innermost,
            `-2` is the next level up, etc.
        keepdims (bool): If False, this function decreases the number of
            dimensions by 1; if True, the output values are wrapped in a new
            length-1 dimension so that the result of this operation may be
            broadcasted with the original array.
        mask_identity (bool): If True, the application of this function on
            empty lists results in None (an option type); otherwise, the
            calculation is followed through with the reducers' identities,
            usually resulting in floating-point `nan`.

    Computes the softmax in each group of elements from `x` (many
    types supported, including all Awkward Arrays and Records). The grouping
    is performed the same way as for reducers, though this operation is not a
    reducer and has no identity.

    This function has no NumPy equivalent.

    Passing all arguments to the reducers, the softmax is calculated as

        np.exp(x) / ak.sum(np.exp(x))

    See #ak.sum for a complete description of handling nested lists and
    missing values (None) in reducers, and #ak.mean for an example with another
    non-reducer.
    """
    # Dispatch
    yield (x,)

    # Implementation
    return _impl(x, axis, keepdims, mask_identity)


def _impl(x, axis, keepdims, mask_identity):
    axis = regularize_axis(axis)
    behavior = behavior_of(x)
    x = ak.highlevel.Array(
        ak.operations.to_layout(x, allow_record=False, allow_other=False),
        behavior=behavior,
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        expx = ufuncs.exp(x)
        denom = ak.operations.ak_sum._impl(
            expx,
            axis,
            keepdims,
            mask_identity,
            highlevel=True,
            behavior=behavior,
        )
        return expx / denom
