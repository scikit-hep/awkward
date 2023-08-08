# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
__all__ = ("is_valid",)
import awkward as ak
from awkward._dispatch import high_level_function


@high_level_function()
def is_valid(array, *, exception=False):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        exception (bool): If True, validity errors raise exceptions.

    Returns True if there are no errors and False if there is an error.

    Checks for errors in the structure of the array, such as indexes that run
    beyond the length of a node's `content`, etc. Either an error is raised or
    the function returns a boolean.

    See also #ak.validity_error.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, exception)


def _impl(array, exception):
    out = ak.operations.validity_error(array, exception=exception)
    return out in (None, "")
