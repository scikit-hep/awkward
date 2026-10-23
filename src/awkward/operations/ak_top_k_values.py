# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from awkward._dispatch import high_level_function
from awkward._experimental import experimental
from awkward.operations.ak_top_k import _impl

__all__ = ("top_k_values",)


@high_level_function()
@experimental
def top_k_values(
    array, k, /, *, axis=-1, mode="largest", highlevel=True, behavior=None, attrs=None
):
    """Return the largest or smallest values along an axis.

    Equivalent to ``ak.top_k(array, k, ...).values`` returning only values.
    Arguments, ordering, missing-value handling, and axis restrictions are the
    same as #ak.top_k. This convenience function extends the Array API proposal,
    whose final specification includes only `top_k`.
    """
    yield (array,)
    return _impl(array, k, axis, mode, highlevel, behavior, attrs, "values")
