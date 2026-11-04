# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from awkward._dispatch import high_level_function
from awkward._experimental import experimental
from awkward.operations.ak_top_k import _impl

__all__ = ("top_k_indices",)


@high_level_function()
@experimental
def top_k_indices(
    array, k, /, *, axis=-1, mode="largest", highlevel=True, behavior=None, attrs=None
):
    """Return indices of the largest or smallest values along an axis.

    Equivalent to ``ak.top_k(array, k, ...).indices`` returning only indices.
    Arguments, ordering, missing-value handling, and axis restrictions are the
    same as #ak.top_k. Indices are local to the selected axis, as in #ak.argsort.
    This convenience function extends the Array API proposal, whose final
    specification includes only `top_k`.
    """
    yield (array,)
    return _impl(array, k, axis, mode, highlevel, behavior, attrs, "indices")
