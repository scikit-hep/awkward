# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
__all__ = ("is_tuple",)
import awkward as ak
from awkward._dispatch import high_level_function


@high_level_function()
def is_tuple(array):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).

    If `array` is a record, this returns True if the record is a tuple.
    If `array` is an array, this returns True if the outermost record is a tuple.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array)


def _impl(array):
    layout = ak.operations.ak_to_layout._impl(
        array, allow_record=True, allow_other=False, regulararray=True
    )

    return layout.is_tuple
